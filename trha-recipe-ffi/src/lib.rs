//! C ABI bridge for deepseek-recipe: JSON-in/JSON-out.
//!
//! Haskell (trha) calls these functions via `foreign import ccall`.
//! All complex types are serialized as JSON strings — serde_json on the
//! Rust side, Aeson on the Haskell side. Memory allocated by this library
//! must be freed with [`recipe_free_string`].
//!
//! # Thread safety
//!
//! The encoding state (V41 tokenizer + render logic) is held in a global
//! `Mutex` — all calls serialise on it. The tokenizer file (6.4 MB JSON) is
//! loaded once by [`recipe_init`] and reused.

use std::ffi::{c_char, CStr, CString};
use std::sync::{Mutex, OnceLock};

use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_encoding::PromptTokenizer;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::PromptEncoding;
use tokenizers::Tokenizer;

/// Global encoding state — initialised by `recipe_init`, reused by all calls.
static STATE: OnceLock<Mutex<FFIState>> = OnceLock::new();

/// V4.1 encoding with tokenizer.
struct FFIState {
    v41: DeepseekV41Encoding,
}

/// Convert a Rust string to a C string pointer for FFI return.
fn to_cstr(s: String) -> *mut c_char {
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
/// Returns JSON: {"ok": true} or {"ok": false, "error": "..."}
#[unsafe(no_mangle)]
pub extern "C" fn recipe_init(tokenizer_path: *const c_char) -> *mut c_char {
    let path = unsafe { CStr::from_ptr(tokenizer_path) };
    let path_str = path.to_string_lossy();

    let init_result = match Tokenizer::from_file(path_str.as_ref()) {
        Ok(tok) => {
            let prompt_tok = PromptTokenizer::new(&tok);
            let encoding = DeepseekV41Encoding::new().with_tokenizer(prompt_tok);
            let _ = STATE.set(Mutex::new(FFIState { v41: encoding }));
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

/// Render a Chat Completions request as a DeepSeek V4.1 prompt.
/// Input: ChatCompletionRequest JSON (the standard OpenAI format).
/// Output: {"prompt": "...", "special_ranges": [{"start": N, "end": N}, ...], "image_sources": [...]}
#[unsafe(no_mangle)]
pub extern "C" fn recipe_render_v41(request_json: *const c_char) -> *mut c_char {
    let lock = match STATE.get() {
        Some(m) => m.lock().unwrap(),
        None => {
            return to_cstr(
                serde_json::json!({"error": "not initialised, call recipe_init first"}).to_string(),
            )
        }
    };

    let input = unsafe { CStr::from_ptr(request_json) };
    let json_str = input.to_string_lossy();

    let request: ChatCompletionRequest = match serde_json::from_str(&json_str) {
        Ok(r) => r,
        Err(e) => {
            return to_cstr(
                serde_json::json!({"error": format!("deserialize: {}", e)}).to_string(),
            )
        }
    };

    let converted = match request.convert(ConversionOptions::default()) {
        Ok(c) => c,
        Err(e) => {
            return to_cstr(
                serde_json::json!({"error": format!("convert: {}", e)}).to_string(),
            )
        }
    };

    let rendered = lock.v41.render_conversation(&converted.conversation);

    let special_ranges: Vec<serde_json::Value> = rendered
        .special_ranges
        .iter()
        .map(|r| serde_json::json!({"start": r.start, "end": r.end}))
        .collect();

    let image_sources: Vec<serde_json::Value> = rendered
        .image_sources
        .iter()
        .map(|src| serde_json::json!({"detail": format!("{:?}", src.detail())}))
        .collect();

    to_cstr(
        serde_json::json!({
            "prompt": rendered.prompt,
            "special_ranges": special_ranges,
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
    let lock = match STATE.get() {
        Some(m) => m.lock().unwrap(),
        None => {
            return to_cstr(serde_json::json!({"error": "not initialised"}).to_string())
        }
    };

    let input = unsafe { CStr::from_ptr(request_json) };
    let json_str = input.to_string_lossy();

    let request: ChatCompletionRequest = match serde_json::from_str(&json_str) {
        Ok(r) => r,
        Err(e) => {
            return to_cstr(
                serde_json::json!({"error": format!("deserialize: {}", e)}).to_string(),
            )
        }
    };

    let converted = match request.convert(ConversionOptions::default()) {
        Ok(c) => c,
        Err(e) => {
            return to_cstr(
                serde_json::json!({"error": format!("convert: {}", e)}).to_string(),
            )
        }
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
    let lock = match STATE.get() {
        Some(m) => m.lock().unwrap(),
        None => {
            return to_cstr(serde_json::json!({"error": "not initialised"}).to_string())
        }
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
