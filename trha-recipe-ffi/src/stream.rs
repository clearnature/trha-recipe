//! Stream sessions: drive the deepseek-recipe `StreamProcessor` behind
//! handle-based C entry points.
//!
//! `recipe_stream_open` converts a protocol request, builds the protocol's
//! chunk generator, and spawns the parse pipeline on an internal tokio
//! runtime (2 workers). `recipe_stream_feed` pushes [`InferenceChunk`]s and
//! drains the serialized protocol chunks that became ready.
//! `recipe_stream_close` ends the inference side, waits for the pipeline to
//! flush, and returns everything still queued.
//!
//! Chunk JSON dialect (feed):
//!
//! | type | fields |
//! |------|--------|
//! | `ready`  | `system_fingerprint?`, `prompt_tokens`, `prompt_cache_hit_tokens` |
//! | `text`   | `content`, `token_count` |
//! | `tokens` | `token_ids: [u32]` |
//! | `finish` | `finish_reason: "stop"\|"length"\|"content_filter"` |

use std::collections::VecDeque;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokenizers::Tokenizer;

use deepseek_recipe::anthropic::messages::request::MessagesRequest;
use deepseek_recipe::openai::responses::request::ResponsesRequest;
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::ProtocolRequest;
use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamError,
    StreamProcessor,
};
use tokio_stream::StreamExt;

/// Internal tokio runtime — small on purpose: this lives inside the host
/// process and only drives chunk parsing, not inference.
/// image.rs 复用同一 runtime 跑异步 resolve。
pub(crate) fn block_on<F: Future>(fut: F) -> F::Output {
    runtime().block_on(fut)
}

fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build stream runtime")
    })
}

/// Session registry — monotonically increasing handles.
static STREAMS: OnceLock<Mutex<std::collections::HashMap<u64, StreamSession>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn streams() -> &'static Mutex<std::collections::HashMap<u64, StreamSession>> {
    STREAMS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

struct StreamSession {
    tx: mpsc::Sender<InferenceChunk>,
    out: Arc<Mutex<VecDeque<Value>>>,
    error: Arc<Mutex<Option<String>>>,
    task: tokio::task::JoinHandle<()>,
}

/// Run one pipeline to completion, queueing serialized protocol chunks.
async fn run_pipeline<G>(
    processor: StreamProcessor<G>,
    rx: mpsc::Receiver<InferenceChunk>,
    out: Arc<Mutex<VecDeque<Value>>>,
    error: Arc<Mutex<Option<String>>>,
) where
    G: ChunkGenerator,
    G::Chunk: Serialize + Send,
{
    let inference = tokio_stream::wrappers::ReceiverStream::new(rx);
    let protocol_stream = processor.into_protocol_stream(inference);
    let mut protocol = std::pin::pin!(protocol_stream);
    while let Some(item) = protocol.next().await {
        match item {
            Ok(chunk) => {
                let Ok(v) = serde_json::to_value(&chunk) else {
                    break;
                };
                out.lock().expect("out lock").push_back(v);
            }
            Err(e) => {
                let msg = match &e {
                    StreamError::MissingTokenizer => "missing tokenizer".to_string(),
                    StreamError::Decode { detail } => format!("decode: {detail}"),
                    StreamError::LogprobCountMismatch { detail } => {
                        format!("logprob count mismatch: {detail}")
                    }
                };
                *error.lock().expect("error lock") = Some(msg);
                break;
            }
        }
    }
}

fn parse_inference_chunk(v: &Value) -> Result<InferenceChunk, String> {
    let ty = v
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or("chunk missing type")?;
    let num = |key: &str| -> Result<usize, String> {
        v.get(key)
            .and_then(|n| n.as_u64())
            .map(|n| n as usize)
            .ok_or(format!("chunk missing {key}"))
    };
    match ty {
        "ready" => Ok(InferenceChunk::Ready {
            system_fingerprint: v
                .get("system_fingerprint")
                .and_then(|s| s.as_str())
                .map(str::to_string),
            prompt_usage: PromptUsage {
                prompt_tokens: num("prompt_tokens")?,
                prompt_cache_hit_tokens: num("prompt_cache_hit_tokens")?,
            },
        }),
        "text" => Ok(InferenceChunk::Text {
            content: v
                .get("content")
                .and_then(|c| c.as_str())
                .ok_or("text chunk missing content")?
                .to_string(),
            token_count: num("token_count")?,
        }),
        "tokens" => Ok(InferenceChunk::Tokens {
            token_ids: v
                .get("token_ids")
                .and_then(|t| t.as_array())
                .ok_or("tokens chunk missing token_ids")?
                .iter()
                .map(|n| n.as_u64().map(|n| n as u32).ok_or("token id not u32"))
                .collect::<Result<Vec<u32>, _>>()?,
            logprobs: None,
        }),
        "finish" => {
            let reason = v
                .get("finish_reason")
                .and_then(|r| r.as_str())
                .ok_or("finish chunk missing finish_reason")?;
            let fr = match reason {
                "stop" => InferenceFinishReason::Stop,
                "length" => InferenceFinishReason::Length,
                "content_filter" => InferenceFinishReason::ContentFilter,
                other => return Err(format!("unknown finish_reason: {other}")),
            };
            Ok(InferenceChunk::Finish { finish_reason: fr })
        }
        other => Err(format!("unknown chunk type: {other}")),
    }
}

/// Drain the queued protocol chunks of a session.
fn drain(out: &Arc<Mutex<VecDeque<Value>>>, error: &Arc<Mutex<Option<String>>>) -> Value {
    let mut q = out.lock().expect("out lock");
    let drained: VecDeque<Value> = std::mem::take(&mut q);
    drop(q);
    let chunks: Vec<Value> = drained.into();
    let err = error.lock().expect("error lock").clone();
    match err {
        Some(m) => json!({"chunks": chunks, "error": m}),
        None => json!({"chunks": chunks}),
    }
}

fn open_session(
    kind: &str,
    request_json: &str,
    tokenizer_path: Option<String>,
    response_id: &str,
    model: &str,
) -> Result<u64, String> {
    // 协议转换 + generator 构造（与 dispatch 共用转换逻辑）
    let opts = deepseek_recipe::request::ConversionOptions::default();
    let converted = match kind {
        "cc" => {
            let req: ChatCompletionRequest = serde_json::from_str(request_json)
                .map_err(|e| format!("deserialize: {e}"))?;
            req.convert(opts).map_err(|e| format!("convert: {e}"))
        }
        "responses" => {
            let req: ResponsesRequest = serde_json::from_str(request_json)
                .map_err(|e| format!("deserialize: {e}"))?;
            req.convert(opts).map_err(|e| format!("convert: {e}"))
        }
        "anthropic" => {
            let req: MessagesRequest = serde_json::from_str(request_json)
                .map_err(|e| format!("deserialize: {e}"))?;
            req.convert(opts).map_err(|e| format!("convert: {e}"))
        }
        other => return Err(format!("unknown protocol kind: {other}")),
    }?;

    // tokenizer 可选：text 流不需要；tokens 流没有它会在解析时报
    // MissingTokenizer（库语义）
    let tokenizer: Option<Tokenizer> = match tokenizer_path {
        Some(ref p) if !p.is_empty() => {
            Some(Tokenizer::from_file(p).map_err(|e| format!("load tokenizer: {e}"))?)
        }
        _ => None,
    };

    let (tx, rx) = mpsc::channel::<InferenceChunk>(64);
    let out: Arc<Mutex<VecDeque<Value>>> = Arc::new(Mutex::new(VecDeque::new()));
    let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    macro_rules! spawn {
        ($req_ty:ty) => {{
            let generator = <$req_ty as ProtocolRequest>::chunk_generator(
                &converted,
                response_id.to_string(),
                model.to_string(),
            );
            let mut processor = StreamProcessor::new(generator, converted.parsing_options.clone());
            if let Some(tok) = tokenizer {
                processor = processor.with_tokenizer(tok);
            }
            let task = runtime().spawn(run_pipeline(
                processor,
                rx,
                out.clone(),
                error.clone(),
            ));
            task
        }};
    }

    let task = match kind {
        "cc" => spawn!(ChatCompletionRequest),
        "responses" => spawn!(ResponsesRequest),
        "anthropic" => spawn!(MessagesRequest),
        _ => unreachable!("checked above"),
    };

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    streams()
        .lock()
        .expect("streams lock")
        .insert(id, StreamSession { tx, out, error, task });
    Ok(id)
}

/// Feed one inference chunk; returns the protocol chunks that became ready.
fn feed_session(id: u64, chunk_json: &str) -> Result<Value, String> {
    let chunk = parse_inference_chunk(
        &serde_json::from_str::<Value>(chunk_json).map_err(|e| format!("parse chunk: {e}"))?,
    )?;
    let (tx, out, error, _task) = {
        let mut streams = streams().lock().expect("streams lock");
        let s = streams
            .get_mut(&id)
            .ok_or_else(|| format!("unknown stream handle: {id}"))?;
        // 拿引用逐字段克隆出来，避免长期持有注册表锁
        let tx = s.tx.clone();
        let out = s.out.clone();
        let error = s.error.clone();
        (tx, out, error, ())
    };
    runtime()
        .block_on(async {
            tx.send(chunk)
                .await
                .map_err(|_| "pipeline dropped the sender (already finished)")?;
            // 等管道消化完这一块（chunk 一定会产出至少一个协议块）；
            // 有界等待——管道异常时也不挂死调用方
            let deadline = std::time::Duration::from_millis(500);
            let mut slept = std::time::Duration::ZERO;
            while out.lock().expect("out lock").is_empty() && slept < deadline {
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                slept += std::time::Duration::from_millis(2);
            }
            Ok::<(), String>(())
        })?;
    Ok(drain(&out, &error))
}

/// End the inference side, wait for the pipeline to flush, return everything.
fn close_session(id: u64, finish_reason: Option<&str>) -> Result<Value, String> {
    let (tx, out, error, task) = {
        let mut streams = streams().lock().expect("streams lock");
        let s = streams
            .remove(&id)
            .ok_or_else(|| format!("unknown stream handle: {id}"))?;
        (s.tx, s.out, s.error, s.task)
    };
    let reason = match finish_reason {
        Some("length") => InferenceFinishReason::Length,
        Some("content_filter") => InferenceFinishReason::ContentFilter,
        _ => InferenceFinishReason::Stop,
    };
    // 显式 Finish 后丢弃 tx；管道在处理完 Finish 后自然结束
    let _ = runtime().block_on(async {
        let _ = tx.send(InferenceChunk::Finish { finish_reason: reason }).await; // 接收端已死也无妨
        drop(tx);
        tokio::time::timeout(std::time::Duration::from_secs(5), task).await
    });
    Ok(drain(&out, &error))
}

/// Abort a session without flushing (drop path).
fn abort_session(id: u64) -> Result<Value, String> {
    let s = streams()
        .lock()
        .expect("streams lock")
        .remove(&id)
        .ok_or_else(|| format!("unknown stream handle: {id}"))?;
    s.task.abort();
    Ok(drain(&s.out, &s.error))
}

/// Total completion usage accumulated so far is reported by the protocol's
/// finish chunk; nothing extra to expose here.

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

use std::ffi::{c_char, CStr};

/// Open a stream session.
/// `tokenizer_path` may be NULL (text-only streams).
/// Returns `{"handle": N}` or `{"error": "..."}`.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_stream_open(
    kind: *const c_char,
    request_json: *const c_char,
    tokenizer_path: *const c_char,
    response_id: *const c_char,
    model: *const c_char,
) -> *mut c_char {
    let s = |p: *const c_char| -> String {
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    };
    let result = open_session(
        &s(kind),
        &s(request_json),
        {
            let p = s(tokenizer_path);
            if p.is_empty() { None } else { Some(p) }
        },
        &s(response_id),
        &s(model),
    );
    let v = match result {
        Ok(id) => json!({"handle": id}),
        Err(e) => json!({"error": e}),
    };
    crate::to_cstr_pub(v.to_string())
}

/// Feed one inference chunk. Returns `{"chunks": [...], "error"?: "..."}`.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_stream_feed(
    handle: u64,
    chunk_json: *const c_char,
) -> *mut c_char {
    let chunk_json = unsafe { CStr::from_ptr(chunk_json) }.to_string_lossy();
    let v = match feed_session(handle, &chunk_json) {
        Ok(v) => v,
        Err(e) => json!({"error": e}),
    };
    crate::to_cstr_pub(v.to_string())
}

/// Close a session: send Finish, flush, return remaining chunks.
/// `finish_reason` may be NULL (defaults to "stop").
#[unsafe(no_mangle)]
pub extern "C" fn recipe_stream_close(
    handle: u64,
    finish_reason: *const c_char,
) -> *mut c_char {
    let reason = if finish_reason.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(finish_reason) }.to_string_lossy().into_owned())
    };
    let v = match close_session(handle, reason.as_deref()) {
        Ok(v) => v,
        Err(e) => json!({"error": e}),
    };
    crate::to_cstr_pub(v.to_string())
}

/// Abort a session without flushing.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_stream_abort(handle: u64) -> *mut c_char {
    let v = match abort_session(handle) {
        Ok(v) => v,
        Err(e) => json!({"error": e}),
    };
    crate::to_cstr_pub(v.to_string())
}
