//! Dispatch smoke tests: every operation, plus a tokenizer-backed
//! render/encode round-trip against the vendored V4.1 tokenizer.

use serde_json::{json, Value};

fn call(op: &str, payload: &Value) -> Value {
    let op_c = std::ffi::CString::new(op).unwrap();
    let payload_c = std::ffi::CString::new(payload.to_string()).unwrap();
    let ptr = trha_recipe_ffi::recipe_dispatch(op_c.as_ptr(), payload_c.as_ptr());
    assert!(!ptr.is_null(), "dispatch returned null for {op}");
    let bytes = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let s = bytes.to_string_lossy().into_owned();
    trha_recipe_ffi::recipe_free_string(ptr);
    serde_json::from_str(&s).expect("dispatch result is JSON")
}

fn tokenizer_path() -> String {
    // cdylib tests run from the workspace root; the vendored tokenizer is
    // static/tokenizers/v41/tokenizer.json.
    let candidates = [
        "static/tokenizers/v41/tokenizer.json",
        "../static/tokenizers/v41/tokenizer.json",
    ];
    for c in candidates {
        if std::path::Path::new(c).exists() {
            return c.to_string();
        }
    }
    panic!("tokenizer.json not found (looked in {candidates:?})");
}

/// Minimal Chat Completions request (only required fields). The model name is
/// the exact API spelling the library validates ("DeepSeek-V4.1-Flash").
fn cc_request() -> Value {
    json!({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
    })
}

/// Idempotent init — every tokenizer-backed test calls this first (the global
/// OnceLock makes repeats free, and tests run in parallel threads).
fn ensure_init() {
    let init = call("init", &json!({"tokenizer_path": tokenizer_path()}));
    assert_eq!(init["ok"], true, "init failed: {init}");
}

#[test]
fn special_tokens_lists_all_18() {
    let v = call("special_tokens", &json!({}));
    assert_eq!(v["count"].as_u64(), Some(18));
    let tokens = v["special_tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), 18);
    for t in tokens {
        assert!(!t["spelling"].as_str().unwrap().is_empty());
    }
    // 抽查两个默认拼写
    let names: Vec<&str> = tokens.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"ThinkingStart"));
    assert!(names.contains(&"LatestReminder"));
}

#[test]
fn model_versions_lists_both() {
    let v = call("model_versions", &json!({}));
    let versions = v["model_versions"].as_array().unwrap();
    assert_eq!(versions.len(), 2);
    let keys: Vec<&str> = versions
        .iter()
        .map(|x| x["render_key"].as_str().unwrap())
        .collect();
    assert!(keys.contains(&"v4"));
    assert!(keys.contains(&"v41"));
}

#[test]
fn version_reports_cargo_version() {
    let v = call("version", &json!({}));
    assert_eq!(v["version"].as_str(), Some(env!("CARGO_PKG_VERSION")));
}

#[test]
fn unknown_operation_errors() {
    let v = call("no_such_op", &json!({}));
    assert!(v["error"].as_str().unwrap().contains("unknown operation"));
}

#[test]
fn convert_cc_produces_conversation() {
    let v = call(
        "convert",
        &json!({"kind": "cc", "request": cc_request()}),
    );
    assert!(v.get("error").is_none(), "unexpected: {v}");
    assert_eq!(v["model_version"].as_str(), Some("DeepseekV41Flash"));
    let conv = &v["conversation"];
    assert_eq!(conv["messages"].as_array().unwrap().len(), 2);
    assert_eq!(conv["messages"][0]["role"], "system");
    assert_eq!(conv["messages"][1]["role"], "user");
    assert_eq!(conv["thinking_mode"], true);
    // top_p 是解析后的 f64
    assert!(v["top_p"].is_number());
}

#[test]
fn convert_anthropic_maps_messages() {
    let v = call(
        "convert",
        &json!({"kind": "anthropic", "request": json!({
            "model": "DeepSeek-V4.1-Flash",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}]
        })}),
    );
    assert!(v.get("error").is_none(), "unexpected: {v}");
    let conv = &v["conversation"];
    assert!(conv["messages"].as_array().unwrap().len() >= 1);
}

#[test]
fn convert_responses_maps_input() {
    let v = call(
        "convert",
        &json!({"kind": "responses", "request": json!({
            "model": "DeepSeek-V4.1-Flash",
            "input": "Hi"
        })}),
    );
    assert!(v.get("error").is_none(), "unexpected: {v}");
    let conv = &v["conversation"];
    assert!(conv["messages"].as_array().unwrap().len() >= 1);
}

#[test]
fn convert_unknown_kind_errors() {
    let v = call(
        "convert",
        &json!({"kind": "wat", "request": cc_request()}),
    );
    assert!(v["error"].as_str().unwrap().contains("unknown protocol kind"));
}

#[test]
fn render_produces_prompt_with_special_ranges() {
    ensure_init();
    let v = call(
        "render",
        &json!({"kind": "cc", "request": cc_request()}),
    );
    assert!(v.get("error").is_none(), "render failed: {v}");
    let prompt = v["prompt"].as_str().unwrap();
    assert!(!prompt.is_empty());
    assert!(prompt.contains("Hello"), "prompt must contain user text");
    // special_ranges 指向 prompt 内的特殊 token 拼写
    let ranges = v["special_ranges"].as_array().unwrap();
    assert!(!ranges.is_empty());
    for r in ranges {
        let (s, e) = (r["start"].as_u64().unwrap() as usize, r["end"].as_u64().unwrap() as usize);
        assert!(e >= s && e <= prompt.len());
    }
}

#[test]
fn render_v4_explicit_version_override() {
    ensure_init();
    let v = call(
        "render",
        &json!({"kind": "cc", "request": cc_request(), "version": "v4"}),
    );
    assert!(v.get("error").is_none(), "v4 render failed: {v}");
    assert_eq!(v["model_version"].as_str(), Some("DeepseekV41Flash"));
    assert!(!v["prompt"].as_str().unwrap().is_empty());
}

#[test]
fn encode_counts_prompt_tokens() {
    ensure_init();
    let v = call(
        "encode",
        &json!({"kind": "cc", "request": cc_request()}),
    );
    assert!(v.get("error").is_none(), "encode failed: {v}");
    let count = v["count"].as_u64().unwrap();
    assert!(count > 0, "non-empty prompt must encode to >0 tokens");
    assert!(v.get("token_ids").is_none(), "ids are opt-in since the ids flag");
}

#[test]
fn count_text_counts_nonempty() {
    ensure_init();
    let v = call("count_text", &json!({"text": "你好，世界 hello"}));
    let count = v["count"].as_u64().expect("count_text returned: {v}");
    assert!(count > 0);
}

#[test]
fn resolve_limits_thinking_defaults() {
    ensure_init();
    // thinking 模式默认 65536；prompt 极短，context_limit 足够
    let v = call(
        "resolve_limits",
        &json!({
            "kind": "cc",
            "request": cc_request(),
            "prompt_tokens": 100,
            "context_limit": 1_000_000
        }),
    );
    assert!(v.get("error").is_none(), "resolve_limits failed: {v}");
    let max_tokens = v["max_tokens"].as_u64().unwrap();
    assert_eq!(max_tokens, 65_536, "thinking default is 65536");

    // context 只剩 10 token 时，默认 max_tokens 被压到剩余额度（clamp，不报错）
    let v2 = call(
        "resolve_limits",
        &json!({
            "kind": "cc",
            "request": cc_request(),
            "prompt_tokens": 999_990,
            "context_limit": 1_000_000
        }),
    );
    assert!(v2.get("error").is_none(), "clamp path: {v2}");
    assert_eq!(v2["max_tokens"].as_u64(), Some(10), "clamped to remaining context");
}

#[test]
fn dispatch_matches_legacy_entry_points() {
    ensure_init();
    // legacy render
    let req_c = std::ffi::CString::new(cc_request().to_string()).unwrap();
    let legacy_ptr = trha_recipe_ffi::recipe_render_v41(req_c.as_ptr());
    let legacy: Value = serde_json::from_str(
        unsafe { std::ffi::CStr::from_ptr(legacy_ptr) }.to_str().unwrap(),
    )
    .unwrap();
    trha_recipe_ffi::recipe_free_string(legacy_ptr);

    // dispatch render — 同输入同输出
    let modern = call("render", &json!({"kind": "cc", "request": cc_request()}));
    assert_eq!(legacy["prompt"], modern["prompt"]);
    assert_eq!(legacy["special_ranges"], modern["special_ranges"]);
}

#[test]
fn encode_omits_ids_by_default_and_includes_on_flag() {
    ensure_init();
    let v = call("encode", &json!({"kind": "cc", "request": cc_request()}));
    assert!(v.get("token_ids").is_none(), "ids must be opt-in: {v}");
    assert!(v["count"].as_u64().unwrap() > 0);

    let v2 = call("encode", &json!({"kind": "cc", "request": cc_request(), "ids": true}));
    assert!(v2["token_ids"].is_array(), "ids=true must return token_ids: {v2}");
}

#[test]
fn resolve_limits_returns_requested_top_p_from_inference_options() {
    ensure_init();
    // 请求带 top_p=0.7：thinking 模式上限 0.95，适配器解析成终值 0.95
    // （既不是请求原值 0.7，也不是无请求时的默认 1.0）——这就是要校准的语义
    let v = call(
        "resolve_limits",
        &json!({
            "kind": "cc",
            "request": {"model": "DeepSeek-V4.1-Flash",
                        "messages": [{"role": "user", "content": "hi"}],
                        "top_p": 0.7},
            "prompt_tokens": 100,
            "context_limit": 1_000_000
        }),
    );
    assert!(v.get("error").is_none(), "{v}");
    let top_p = v["top_p"].as_f64().unwrap();
    assert!((top_p - 0.95).abs() < 1e-6, "thinking top_p cap surfaces: {v}");
}

// ── image 子系统 ─────────────────────────────────────────────────────

/// 用 image crate 现场生成一张 PNG（红 64×48），避免 fixture 依赖。
fn red_png(width: u32, height: u32) -> Vec<u8> {
    use std::io::Cursor;
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |_x, _y| {
        image::Rgb([255u8, 0, 0])
    }));
    let mut out = Cursor::new(Vec::new());
    image::DynamicImage::write_to(&img, &mut out, image::ImageFormat::Png).expect("png encodes");
    out.into_inner()
}

#[test]
fn image_token_spec_matches_library_math() {
    // 大图（≥min_pixels）：token 数 ≤ 上限 1024
    let v = call("image_token_spec", &json!({"width": 1024, "height": 768}));
    assert!(v.get("error").is_none(), "{v}");
    let tokens = v["token_count"].as_u64().unwrap();
    assert!(tokens > 0 && tokens <= 1024, "token budget respected: {v}");
    assert_eq!(v["max_token_count"], 1024);

    // 小图被放大到 min 面积（544*544）后再对齐 patch
    let small = call("image_token_spec", &json!({"width": 64, "height": 48}));
    let st = small["token_count"].as_u64().unwrap();
    assert!(st > 0, "small image still costs tokens: {small}");

    // 两种尺寸的 token 数不同（数学不为常数）
    assert_ne!(tokens, st);
}

#[test]
fn image_resolve_data_url_preprocesses_and_counts_tokens() {
    let png = red_png(64, 48);
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    let v = call(
        "image_resolve",
        &json!({
            "sources": [{"kind": "data_url",
                         "data_url": format!("data:image/png;base64,{b64}"),
                         "path": "red.png"}]
        }),
    );
    assert!(v.get("error").is_none(), "resolve failed: {v}");
    let imgs = v["images"].as_array().unwrap();
    assert_eq!(imgs.len(), 1);
    assert!(imgs[0]["width"].as_u64().unwrap() >= 1);
    assert!(imgs[0]["token_count"].as_u64().unwrap() > 0);
    assert_eq!(v["image_count"], 1);
}

#[test]
fn image_resolve_rejects_private_url_hosts() {
    let v = call(
        "image_resolve",
        &json!({"sources": [{"kind": "url", "url": "http://127.0.0.1:9/a.png"}]}),
    );
    // 默认 SSRF 防护：环回地址在 DNS/连接前就被拒
    assert!(v.get("error").is_some(), "private host must be rejected: {v}");
}

#[test]
fn encode_text_maps_added_vocab_and_plain_text() {
    ensure_init();
    // added-vocab token（</think>、｜DSML｜）必须成为独立 token——流解析语义的基础
    let v = call("encode_text", &json!({"text": "思考</think>答案<｜DSML｜invoke"}));
    assert!(v.get("error").is_none(), "{v}");
    let ids = v["token_ids"].as_array().unwrap();
    assert!(ids.len() > 3, "multi-token: {v}");
    // </think> = 128822、｜DSML｜ = 128825（vendored 词表实测）
    let nums: Vec<u64> = ids.iter().map(|x| x.as_u64().unwrap()).collect();
    assert!(nums.contains(&128822), "</think> must be its own token: {v}");
    assert!(nums.contains(&128825), "｜DSML｜ must be its own token: {v}");
}
