//! C ABI bridge for deepseek-recipe: JSON-in/JSON-out.
//!
//! Haskell (trha) calls these functions via `foreign import ccall`. All
//! complex types are serialized as JSON strings — serde_json on the Rust
//! side, Aeson on the Haskell side. Memory allocated by this library must
//! be freed with [`recipe_free_string`].
//!
//! # Capability surface
//!
//! The deepseek-recipe capability set is exposed two ways:
//!
//! * Legacy direct entry points (`recipe_init`, `recipe_render_v41`,
//!   `recipe_encode_v41`, `recipe_count_tokens`) — kept for the existing
//!   Haskell binding.
//! * [`recipe_dispatch`] — one entry point, an operation name plus a JSON
//!   payload, covering protocol conversion (Chat Completions / Responses /
//!   Anthropic), V4 and V4.1 rendering, token encoding, inference limit
//!   resolution (`max_tokens`, `top_p`), and the special-token table.
//!
//! # Thread safety
//!
//! The encoding state (V4.1 tokenizer + render logic) is held in a global
//! `Mutex` — all calls serialise on it. The tokenizer file (6.4 MB JSON) is
//! loaded once by [`recipe_init`] and reused.

use std::ffi::{c_char, CStr, CString};
use std::sync::{Mutex, MutexGuard, OnceLock};

mod image;
mod stream;

pub use stream::{
    recipe_stream_abort, recipe_stream_close, recipe_stream_feed, recipe_stream_open,
};

use deepseek_recipe::anthropic::messages::request::MessagesRequest;
use deepseek_recipe::openai::responses::request::ResponsesRequest;
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{
    ConversionOptions, ConversationRequest, ModelVersion, ProtocolRequest,
};
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_core::special_tokens::SpecialTokensMap;
use deepseek_recipe_encoding::PromptTokenizer;
use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::PromptEncoding;
use tokenizers::Tokenizer;

/// Global encoding state — initialised by `recipe_init`, reused by all calls.
static STATE: OnceLock<Mutex<FFIState>> = OnceLock::new();

/// V4.1 encoding with tokenizer.
struct FFIState {
    v41: DeepseekV41Encoding,
    /// 裸 tokenizer：encode_text（纯文本 → token IDs，无对话框架）用。
    /// CPU 产生器 / 真后端 tokenizer 都走它。
    tokenizer: Option<Tokenizer>,
}

/// Convert a Rust string to a C string pointer for FFI return.
fn to_cstr(s: String) -> *mut c_char {
    to_cstr_pub(s)
}

/// `to_cstr` for the sibling `stream` module.
pub(crate) fn to_cstr_pub(s: String) -> *mut c_char {
    let cleaned = s.replace('\0', "\\0");
    match CString::new(cleaned) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a string returned by this library.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}

/// Initialise the encoding with a tokenizer file path (JSON file).
/// Returns JSON: `{"ok": true}` or `{"ok": false, "error": "..."}`.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_init(tokenizer_path: *const c_char) -> *mut c_char {
    let path = unsafe { CStr::from_ptr(tokenizer_path) };
    let path_str = path.to_string_lossy();

    let init_result = match Tokenizer::from_file(path_str.as_ref()) {
        Ok(tok) => {
            let prompt_tok = PromptTokenizer::new(&tok);
            let encoding = DeepseekV41Encoding::new().with_tokenizer(prompt_tok);
            let _ = STATE.set(Mutex::new(FFIState { v41: encoding, tokenizer: Some(tok) }));
            serde_json::json!({"ok": true})
        }
        Err(e) => serde_json::json!({"ok": false, "error": e.to_string()}),
    };

    to_cstr(init_result.to_string())
}

/// Check if the encoding is initialised.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_is_init() -> bool {
    STATE.get().is_some()
}

// ---------------------------------------------------------------------------
// 分发核心（recipe_dispatch 与旧入口共用）
// ---------------------------------------------------------------------------

fn state_v41() -> Result<MutexGuard<'static, FFIState>, serde_json::Value> {
    match STATE.get() {
        Some(m) => m.lock().map_err(|_| {
            serde_json::json!({"error": "state mutex poisoned"})
        }),
        None => Err(serde_json::json!({"error": "not initialised, call recipe_init first"})),
    }
}

/// Deserialize a protocol request of `kind` and convert it to a
/// `ConversationRequest`. `kind` is `"cc"`, `"responses"`, or `"anthropic"`.
fn convert_request(
    kind: &str,
    json_str: &str,
) -> Result<ConversationRequest, serde_json::Value> {
    let opts = ConversionOptions::default();
    let conv_err = |stage: &str, e: String| {
        serde_json::json!({"error": format!("{}: {}", stage, e)})
    };
    match kind {
        "cc" => {
            let req: ChatCompletionRequest = serde_json::from_str(json_str)
                .map_err(|e| conv_err("deserialize", e.to_string()))?;
            req.convert(opts)
                .map_err(|e| conv_err("convert", e.to_string()))
        }
        "responses" => {
            let req: ResponsesRequest = serde_json::from_str(json_str)
                .map_err(|e| conv_err("deserialize", e.to_string()))?;
            req.convert(opts)
                .map_err(|e| conv_err("convert", e.to_string()))
        }
        "anthropic" => {
            let req: MessagesRequest = serde_json::from_str(json_str)
                .map_err(|e| conv_err("deserialize", e.to_string()))?;
            req.convert(opts)
                .map_err(|e| conv_err("convert", e.to_string()))
        }
        other => Err(serde_json::json!({
            "error": format!("unknown protocol kind: {} (want cc|responses|anthropic)", other)
        })),
    }
}

/// Serialize a `Conversation` to JSON. `Conversation` has no serde derives,
/// so this walks the tree manually — a faithful round-trip of every field.
fn conversation_to_json(conv: &Conversation) -> serde_json::Value {
    use serde_json::json;

    let message_json = |m: &InputMessage| match m {
        InputMessage::System { content, image_sources, tools } => {
            let mut o = json!({
                "role": "system",
                "content": content.to_string(),
                "image_sources": image_sources_to_json(image_sources),
            });
            if let Some(t) = tools {
                o["tools"] = json!(t.iter().map(tool_to_json).collect::<Vec<_>>());
            }
            o
        }
        InputMessage::User { content, image_sources } => json!({
            "role": "user",
            "content": content.to_string(),
            "image_sources": image_sources_to_json(image_sources),
        }),
        InputMessage::Assistant { content, reasoning_content, tool_calls, prefix } => {
            let mut o = json!({
                "role": "assistant",
                "content": content,
                "prefix": prefix,
            });
            if let Some(rc) = reasoning_content {
                o["reasoning_content"] = json!(rc);
            }
            if let Some(tc) = tool_calls {
                o["tool_calls"] = json!(tc.iter().map(tool_call_to_json).collect::<Vec<_>>());
            }
            o
        }
        InputMessage::Tool { content, image_sources, tool_call_id } => {
            let mut o = json!({
                "role": "tool",
                "content": content.to_string(),
                "image_sources": image_sources_to_json(image_sources),
            });
            if let Some(id) = tool_call_id {
                o["tool_call_id"] = json!(id);
            }
            o
        }
        InputMessage::LatestReminder { content } => json!({
            "role": "latest_reminder",
            "content": content,
        }),
    };

    json!({
        "thinking_mode": conv.thinking_mode,
        "messages": conv.messages.iter().map(message_json).collect::<Vec<_>>(),
        "tools": conv.tools.iter().map(tool_to_json).collect::<Vec<_>>(),
        "tool_choice": format!("{:?}", conv.tool_choice),
        "reasoning_effort": conv.reasoning_effort.map(|e| format!("{e:?}")),
        "response_format": format!("{:?}", conv.response_format),
    })
}

fn tool_to_json(t: &deepseek_recipe_core::tools::ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
        "strict": t.strict,
        "defer_loading": t.defer_loading,
    })
}

fn tool_call_to_json(tc: &ToolCall) -> serde_json::Value {
    serde_json::json!({"id": tc.id, "name": tc.name, "arguments": tc.arguments})
}

fn image_sources_to_json(
    sources: &[deepseek_recipe_core::multimodal::ImageSource],
) -> serde_json::Value {
    use deepseek_recipe_core::multimodal::ImageSource;
    serde_json::Value::Array(
        sources
            .iter()
            .map(|s| match s {
                ImageSource::DataUrl { data_url, detail, path } => serde_json::json!({
                    "kind": "data_url", "data_url": data_url,
                    "detail": format!("{:?}", detail), "path": path,
                }),
                ImageSource::Url { url, detail, .. } => serde_json::json!({
                    "kind": "url", "url": url, "detail": format!("{:?}", detail),
                }),
                ImageSource::Bytes { data, detail, path } => serde_json::json!({
                    "kind": "bytes", "bytes_len": data.len(),
                    "detail": format!("{:?}", detail), "path": path,
                }),
            })
            .collect(),
    )
}

fn special_ranges_to_json(ranges: &[std::ops::Range<usize>]) -> Vec<serde_json::Value> {
    ranges
        .iter()
        .map(|r| serde_json::json!({"start": r.start, "end": r.end}))
        .collect()
}

// ---------------------------------------------------------------------------
// Legacy direct entry points（Haskell 绑定已在使用）
// ---------------------------------------------------------------------------

/// Render a Chat Completions request as a DeepSeek V4.1 prompt.
/// Input: ChatCompletionRequest JSON (the standard OpenAI format).
/// Output: {"prompt": "...", "special_ranges": [...], "image_sources": [...]}
#[unsafe(no_mangle)]
pub extern "C" fn recipe_render_v41(request_json: *const c_char) -> *mut c_char {
    let lock = match state_v41() {
        Ok(l) => l,
        Err(e) => return to_cstr(e.to_string()),
    };

    let input = unsafe { CStr::from_ptr(request_json) };
    let json_str = input.to_string_lossy();

    let converted = match convert_request("cc", &json_str) {
        Ok(c) => c,
        Err(e) => return to_cstr(e.to_string()),
    };

    let rendered = lock.v41.render_conversation(&converted.conversation);

    let image_sources: Vec<serde_json::Value> = rendered
        .image_sources
        .iter()
        .map(|src| serde_json::json!({"detail": format!("{:?}", src.detail())}))
        .collect();

    to_cstr(
        serde_json::json!({
            "prompt": rendered.prompt,
            "special_ranges": special_ranges_to_json(&rendered.special_ranges),
            "image_sources": image_sources,
        })
        .to_string(),
    )
}

/// Encode a Chat Completions request into token IDs (V4.1).
/// Input: ChatCompletionRequest JSON.
/// Output: {"token_ids": [N, ...], "count": N}
#[unsafe(no_mangle)]
pub extern "C" fn recipe_encode_v41(request_json: *const c_char) -> *mut c_char {
    let lock = match state_v41() {
        Ok(l) => l,
        Err(e) => return to_cstr(e.to_string()),
    };

    let input = unsafe { CStr::from_ptr(request_json) };
    let json_str = input.to_string_lossy();

    let converted = match convert_request("cc", &json_str) {
        Ok(c) => c,
        Err(e) => return to_cstr(e.to_string()),
    };

    match lock.v41.encode(&converted.conversation) {
        Ok(ids) => to_cstr(
            serde_json::json!({"token_ids": ids, "count": ids.len()}).to_string(),
        ),
        Err(e) => to_cstr(serde_json::json!({"error": e.to_string()}).to_string()),
    }
}

/// Count tokens in raw text (without conversation rendering).
/// Input: plain text (NOT JSON).
/// Output: {"count": N}
#[unsafe(no_mangle)]
pub extern "C" fn recipe_count_tokens(text_ptr: *const c_char) -> *mut c_char {
    let lock = match state_v41() {
        Ok(l) => l,
        Err(e) => return to_cstr(e.to_string()),
    };

    let text = unsafe { CStr::from_ptr(text_ptr) };
    let text_str = text.to_string_lossy();

    // 最小 Conversation（单条 user 消息）→ encode → token 数
    let conv = Conversation {
        messages: vec![InputMessage::User {
            content: Prompt::plain(text_str.as_ref()),
            image_sources: vec![],
        }],
        ..Default::default()
    };
    let count = match lock.v41.encode(&conv) {
        Ok(ids) => ids.len(),
        Err(_) => 0,
    };

    to_cstr(serde_json::json!({"count": count}).to_string())
}

/// Get library version.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_version() -> *mut c_char {
    to_cstr(env!("CARGO_PKG_VERSION").to_string())
}

// ---------------------------------------------------------------------------
// 统一分发入口
// ---------------------------------------------------------------------------

/// One entry point for the full capability surface.
///
/// Operations (payload is JSON unless noted):
///
/// | op               | input                                                | output keys |
/// |------------------|------------------------------------------------------|-------------|
/// | `convert`        | `{"kind","request"}`                                 | `conversation`, `model_version`, `stream`, `top_p` |
/// | `render`         | `{"kind","request","version"?}`                      | `prompt`, `special_ranges`, `model_version` |
/// | `encode`         | `{"kind","request","version"?}`                      | `token_ids`, `count`, `model_version` |
/// | `count_text`     | `{"text"}`                                           | `count` |
/// | `resolve_limits` | `{"kind","request","prompt_tokens","context_limit"}` | `max_tokens`, `top_p`, `model_version` |
/// | `special_tokens` | `{}`                                                 | `special_tokens`, `count` |
/// | `model_versions` | `{}`                                                 | `model_versions` |
/// | `version`        | `{}`                                                 | `version` |
///
/// `kind` selects the protocol (`cc` / `responses` / `anthropic`); `version`
/// overrides the render target (`v4` / `v41`), defaulting to the version the
/// conversion resolved from the model name.
#[unsafe(no_mangle)]
pub extern "C" fn recipe_dispatch(
    operation: *const c_char,
    payload: *const c_char,
) -> *mut c_char {
    let op = unsafe { CStr::from_ptr(operation) }.to_string_lossy();
    let payload = unsafe { CStr::from_ptr(payload) }.to_string_lossy();

    let result = dispatch(&op, &payload);
    match result {
        Ok(v) => to_cstr(v.to_string()),
        Err(e) => {
            eprintln!("[recipe-ffi] dispatch error ({}): {}", op, e);
            to_cstr(e.to_string())
        }
    }
}

fn kind_of(v: &serde_json::Value) -> &str {
    v.get("kind").and_then(|k| k.as_str()).unwrap_or("cc")
}

fn dispatch(op: &str, payload: &str) -> Result<serde_json::Value, serde_json::Value> {
    let parse_payload = || -> Result<serde_json::Value, serde_json::Value> {
        serde_json::from_str(payload)
            .map_err(|e| serde_json::json!({"error": format!("parse payload: {e}")}))
    };

    let request_of = |v: &serde_json::Value| -> Result<String, serde_json::Value> {
        let request = v
            .get("request")
            .ok_or_else(|| serde_json::json!({"error": "missing request"}))?;
        Ok(if request.is_string() {
            request.as_str().unwrap_or("").to_string()
        } else {
            request.to_string()
        })
    };

    match op {
        "init" => {
            let v = parse_payload()?;
            let path = v
                .get("tokenizer_path")
                .and_then(|t| t.as_str())
                .ok_or_else(|| serde_json::json!({"error": "missing tokenizer_path"}))?;
            let path_c = CString::new(path)
                .map_err(|e| serde_json::json!({"error": format!("tokenizer_path: {e}")}))?;
            let ptr = recipe_init(path_c.as_ptr());
            let s = unsafe { CStr::from_ptr(ptr) };
            let parsed: serde_json::Value = serde_json::from_str(s.to_str().unwrap_or("{}"))
                .unwrap_or_else(|_| serde_json::json!({"error": "init returned non-JSON"}));
            recipe_free_string(ptr);
            Ok(parsed)
        }
        "convert" => {
            let v = parse_payload()?;
            let converted = convert_request(kind_of(&v), &request_of(&v)?)?;
            let top_p = converted.model_version.resolve_top_p(
                None,
                converted.conversation.thinking_mode,
                !converted.conversation.tools.is_empty(),
            );
            Ok(serde_json::json!({
                "conversation": conversation_to_json(&converted.conversation),
                "model_version": format!("{:?}", converted.model_version),
                "stream": converted.stream,
                "top_p": top_p,
            }))
        }
        "render" | "encode" => {
            let v = parse_payload()?;
            let converted = convert_request(kind_of(&v), &request_of(&v)?)?;

            // 版本选择：显式 v4/v41 优先，否则按转换出的 model_version
            let explicit = v.get("version").and_then(|k| k.as_str());
            let use_v41 = match explicit {
                Some("v4") => false,
                Some("v41") => true,
                Some(other) => {
                    return Err(serde_json::json!({
                        "error": format!("unknown version: {other} (want v4|v41)")
                    }))
                }
                None => converted.model_version == ModelVersion::DeepseekV41Flash,
            };

            if op == "encode" {
                let ids = if use_v41 {
                    let lock = state_v41().map_err(|e| e)?;
                    lock.v41.encode(&converted.conversation)
                } else {
                    DeepseekV4Encoding::new().encode(&converted.conversation)
                }
                .map_err(|e| serde_json::json!({"error": e.to_string()}))?;
                // ids=false（默认）只回 count——50 万 token 的响应不该默认全量回传
                let want_ids = v.get("ids").and_then(|b| b.as_bool()).unwrap_or(false);
                let mut out = serde_json::json!({
                    "count": ids.len(),
                    "model_version": format!("{:?}", converted.model_version),
                });
                if want_ids {
                    out["token_ids"] = serde_json::json!(ids);
                }
                return Ok(out);
            }

            let rendered = if use_v41 {
                let lock = state_v41().map_err(|e| e)?;
                lock.v41.render_conversation(&converted.conversation)
            } else {
                DeepseekV4Encoding::new().render_conversation(&converted.conversation)
            };
            Ok(serde_json::json!({
                "prompt": rendered.prompt,
                "special_ranges": special_ranges_to_json(&rendered.special_ranges),
                "model_version": format!("{:?}", converted.model_version),
            }))
        }
        "count_text" => {
            let v = parse_payload()?;
            let text = v
                .get("text")
                .and_then(|t| t.as_str())
                .ok_or_else(|| serde_json::json!({"error": "missing text"}))?;
            let lock = state_v41().map_err(|e| e)?;
            let conv = Conversation {
                messages: vec![InputMessage::User {
                    content: Prompt::plain(text),
                    image_sources: vec![],
                }],
                ..Default::default()
            };
            let count = lock.v41.encode(&conv).map(|ids| ids.len()).unwrap_or(0);
            Ok(serde_json::json!({"count": count}))
        }
        "resolve_limits" | "limits" => {
            let v = parse_payload()?;
            let converted = convert_request(kind_of(&v), &request_of(&v)?)?;
            let as_u32 = |key: &str| -> Result<u32, serde_json::Value> {
                v.get(key)
                    .and_then(|t| t.as_u64())
                    .map(|n| n.min(u32::MAX as u64) as u32)
                    .ok_or_else(|| serde_json::json!({"error": format!("missing {key}")}))
            };
            let prompt_tokens = as_u32("prompt_tokens")?;
            let context_limit = as_u32("context_limit")?;

            // top_p：适配器转换时已按请求值解析进 inference_options；缺失
            // （协议层未给）再按模型版本+模式重算默认
            let top_p = converted
                .inference_options
                .top_p
                .unwrap_or_else(|| {
                    converted.model_version.resolve_top_p(
                        None,
                        converted.conversation.thinking_mode,
                        !converted.conversation.tools.is_empty(),
                    )
                });
            let max_tokens = converted
                .resolve_max_tokens(prompt_tokens, context_limit)
                .map_err(|e| serde_json::json!({"error": e.to_string()}))?;
            Ok(serde_json::json!({
                "max_tokens": max_tokens,
                "top_p": top_p,
                "model_version": format!("{:?}", converted.model_version),
            }))
        }
        "special_tokens" => {
            let map = SpecialTokensMap::default();
            let tokens: Vec<serde_json::Value> = vec![
                ("Bos", map.bos), ("Eos", map.eos),
                ("ThinkingStart", map.thinking_start), ("ThinkingEnd", map.thinking_end),
                ("System", map.system), ("User", map.user),
                ("Assistant", map.assistant), ("LatestReminder", map.latest_reminder),
                ("Dsml", map.dsml), ("Image", map.image),
                ("Ref", map.r#ref), ("RefEnd", map.ref_end),
                ("Box", map.r#box), ("BoxEnd", map.box_end),
                ("Point", map.point), ("PointEnd", map.point_end),
                ("Polygon", map.polygon), ("PolygonEnd", map.polygon_end),
            ]
            .into_iter()
            .map(|(name, spelling)| {
                serde_json::json!({"name": name, "spelling": spelling})
            })
            .collect();
            Ok(serde_json::json!({"special_tokens": tokens, "count": tokens.len()}))
        }
        "model_versions" => {
            let versions: Vec<serde_json::Value> = [
                (ModelVersion::DeepseekV4Pro, "DeepseekV4Pro", "v4", false),
                (ModelVersion::DeepseekV41Flash, "DeepseekV41Flash", "v41", true),
            ]
            .iter()
            .map(|(mv, name, key, img)| {
                serde_json::json!({
                    "name": name, "render_key": key,
                    "supports_image": img,
                    "supports_mid_conversation_system": mv.supports_mid_conversation_system(),
                })
            })
            .collect();
            Ok(serde_json::json!({"model_versions": versions}))
        }
        "encode_text" => {
            let v = parse_payload()?;
            let text = v
                .get("text")
                .and_then(|t| t.as_str())
                .ok_or_else(|| serde_json::json!({"error": "missing text"}))?;
            let lock = state_v41().map_err(|e| e)?;
            let tok = lock
                .tokenizer
                .as_ref()
                .ok_or_else(|| serde_json::json!({"error": "tokenizer not loaded"}))?;
            let encoding = tok
                .encode(text, false)
                .map_err(|e| serde_json::json!({"error": e.to_string()}))?;
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            Ok(serde_json::json!({"token_ids": ids, "count": ids.len()}))
        }
        "image_token_spec" => image::op_image_token_spec(payload).map_err(|e| e),
        "image_resolve" => image::op_image_resolve(payload).map_err(|e| e),
        "version" => Ok(serde_json::json!({"version": env!("CARGO_PKG_VERSION")})),
        other => Err(serde_json::json!({"error": format!("unknown operation: {other}")})),
    }
}
