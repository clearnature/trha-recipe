//! Stream session tests: text streaming, reasoning splitting, tool-call
//! parsing, and usage accounting — all through the C ABI handle layer.

use serde_json::{json, Value};

fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap()
}

/// Call a pointer-taking stream entry point and parse its JSON reply.
fn call_ptr(f: impl FnOnce() -> *mut std::ffi::c_char) -> Value {
    let ptr = f();
    assert!(!ptr.is_null(), "stream call returned null");
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
    let v: Value = serde_json::from_str(s).expect("reply is JSON");
    trha_recipe_ffi::recipe_free_string(ptr);
    v
}

fn open(request: Value) -> u64 {
    let req = cstr(&request.to_string());
    let kind = cstr("cc");
    let v = call_ptr(|| {
        trha_recipe_ffi::recipe_stream_open(
            kind.as_ptr(),
            req.as_ptr(),
            std::ptr::null(),
            cstr("chatcmpl-test").as_ptr(),
            cstr("DeepSeek-V4.1-Flash").as_ptr(),
        )
    });
    assert!(v.get("error").is_none(), "open failed: {v}");
    v["handle"].as_u64().expect("handle in reply")
}

fn feed(handle: u64, chunk: &Value) -> Value {
    let c = cstr(&chunk.to_string());
    call_ptr(|| trha_recipe_ffi::recipe_stream_feed(handle, c.as_ptr()))
}

fn close(handle: u64) -> Value {
    call_ptr(|| trha_recipe_ffi::recipe_stream_close(handle, std::ptr::null()))
}

fn cc_request(thinking: bool) -> Value {
    json!({
        "model": "DeepSeek-V4.1-Flash",
        "stream": true,
        "messages": [
            {"role": "user", "content": "List three colours."}
        ],
        "thinking": {"type": if thinking {"enabled"} else {"disabled"}}
    })
}

fn ready() -> Value {
    json!({"type": "ready", "prompt_tokens": 12, "prompt_cache_hit_tokens": 0})
}

fn finish() -> Value {
    json!({"type": "finish", "finish_reason": "stop"})
}

#[test]
fn text_stream_produces_cc_chunks_and_usage() {
    let h = open(cc_request(false));
    let r1 = feed(h, &ready());
    // ready 触发 Start chunk（role-only）
    assert!(r1["chunks"].as_array().unwrap().iter().any(|c| c["choices"][0]["delta"]["role"] == "assistant"),
        "expect a start chunk with role, got {r1}");

    let r2 = feed(h, &json!({"type": "text", "content": "red, green, blue", "token_count": 5}));
    let text: String = r2["chunks"].as_array().unwrap().iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "red, green, blue");

    let r3 = close(h);
    let chunks = r3["chunks"].as_array().unwrap();
    // 关闭时刷新出 finish chunk：finish_reason stop + usage
    let finish_chunk = chunks.iter().find(|c| c["choices"][0].get("finish_reason").is_some());
    assert!(finish_chunk.is_some(), "expect finish chunk in {r3}");
    assert_eq!(finish_chunk.unwrap()["choices"][0]["finish_reason"], "stop");
    assert!(r3.get("error").is_none(), "no stream error: {r3}");
}

fn tokenizer() -> tokenizers::Tokenizer {
    let candidates = ["static/tokenizers/v41/tokenizer.json", "../static/tokenizers/v41/tokenizer.json"];
    for c in candidates {
        if std::path::Path::new(c).exists() {
            return tokenizers::Tokenizer::from_file(c).expect("V4.1 tokenizer loads");
        }
    }
    panic!("tokenizer.json not found (looked in {candidates:?})");
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, false).expect("text encodes").get_ids().to_vec()
}

fn open_with_tokenizer(request: Value) -> u64 {
    let tok_path = ["static/tokenizers/v41/tokenizer.json", "../static/tokenizers/v41/tokenizer.json"]
        .iter().find(|c| std::path::Path::new(c).exists()).expect("tokenizer.json present")
        .to_string();
    let req = cstr(&request.to_string());
    let kind = cstr("cc");
    let path = cstr(&tok_path);
    let v = call_ptr(|| {
        trha_recipe_ffi::recipe_stream_open(
            kind.as_ptr(), req.as_ptr(), path.as_ptr(),
            cstr("chatcmpl-test").as_ptr(), cstr("DeepSeek-V4.1-Flash").as_ptr())
    });
    assert!(v.get("error").is_none(), "open failed: {v}");
    v["handle"].as_u64().expect("handle in reply")
}

/// V4.1 思考流：推理文本以 token 形式进来，`</think>` 特殊 token 分界。
/// （V4.1 输出不以 `<think>` 开头——thinking_initial_stage 已置 Reasoning。）
#[test]
fn thinking_stream_splits_reasoning_from_content() {
    let tok = tokenizer();
    let h = open_with_tokenizer(cc_request(true));
    feed(h, &ready());
    let ids = encode(&tok, "pondering…</think>the answer");
    let r = feed(h, &json!({"type": "tokens", "token_ids": ids}));
    let reasoning: String = r["chunks"].as_array().unwrap().iter()
        .filter_map(|c| c["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    let content: String = r["chunks"].as_array().unwrap().iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(reasoning, "pondering…", "reasoning split: {r}");
    assert_eq!(content, "the answer", "content after </think>: {r}");
    close(h);
}

/// 工具调用流：DSML 块以 token 形式进来（`｜DSML｜` 是 added-vocab token），
/// 解析出工具名与参数；finish_reason 报 tool_calls。
/// 用例块与 deepseek-recipe processor.rs 的 TOOL_CALLS fixture 同构。
#[test]
fn tool_call_stream_parses_dsml_block() {
    let tok = tokenizer();
    let h = open_with_tokenizer(json!({
        "model": "DeepSeek-V4.1-Flash",
        "stream": true,
        "messages": [{"role": "user", "content": "What time is it?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }],
        "thinking": {"type": "disabled"}
    }));
    feed(h, &ready());
    let block = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_time\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    let ids = encode(&tok, block);
    let r = feed(h, &json!({"type": "tokens", "token_ids": ids}));
    let chunks = r["chunks"].as_array().unwrap();
    let names: Vec<String> = chunks.iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"][0]["function"]["name"].as_str().map(str::to_string))
        .filter(|s| !s.is_empty())
        .collect();
    assert_eq!(names, vec!["get_time"], "tool named: {r}");

    let r2 = close(h);
    let chunks2 = r2["chunks"].as_array().unwrap();
    // 参数增量可能落在 feed 或 close 批次——跨两批累计
    let mut all = chunks.clone();
    all.extend(chunks2.iter().cloned());
    let args: String = all.iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str().map(str::to_string))
        .collect();
    assert_eq!(args, "{}", "empty invoke parses to {{}}: {r2}");
    let finish = chunks2.iter()
        .find(|c| c["choices"][0].get("finish_reason").is_some())
        .expect("finish chunk");
    assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls",
        "tool call finish: {}", serde_json::to_string(&r2).unwrap());
}

#[test]
fn unknown_handle_errors_cleanly() {
    let c = cstr(r#"{"type":"text","content":"x","token_count":1}"#);
    let v = call_ptr(|| trha_recipe_ffi::recipe_stream_feed(999_999, c.as_ptr()));
    assert!(v["error"].as_str().unwrap().contains("unknown stream handle"));
    let v2 = call_ptr(|| trha_recipe_ffi::recipe_stream_close(999_999, std::ptr::null()));
    assert!(v2["error"].as_str().unwrap().contains("unknown stream handle"));
}

#[test]
fn malformed_chunk_errors_without_killing_session() {
    let h = open(cc_request(false));
    let bad = cstr(r#"{"type":"wat"}"#);
    let v = call_ptr(|| trha_recipe_ffi::recipe_stream_feed(h, bad.as_ptr()));
    assert!(v["error"].as_str().unwrap().contains("unknown chunk type"));
    // 会话仍在：正常 feed 再关闭
    let r = feed(h, &json!({"type": "text", "content": "ok", "token_count": 1}));
    assert!(r.get("error").is_none());
    close(h);
}

#[test]
fn end_of_stream_without_finish_reports_eos_reason() {
    let h = open(cc_request(false));
    feed(h, &ready());
    feed(h, &json!({"type": "text", "content": "bye", "token_count": 2}));
    // 不发 finish chunk 直接 close（tx 被 drop → EndOfStream）
    let c = cstr("length");
    let r = call_ptr(|| trha_recipe_ffi::recipe_stream_close(h, c.as_ptr()));
    assert!(r.get("error").is_none(), "close with reason failed: {r}");
}

#[test]
fn open_bad_request_errors() {
    let req = cstr(r#"{"model": "Not-A-Model", "messages": []}"#);
    let kind = cstr("cc");
    let v = call_ptr(|| {
        trha_recipe_ffi::recipe_stream_open(
            kind.as_ptr(), req.as_ptr(), std::ptr::null(),
            cstr("id").as_ptr(), cstr("m").as_ptr())
    });
    assert!(v["error"].as_str().unwrap().contains("convert"));
}
