//! Image FFI: `ImageTokenSpec` math and `ImageResolver` with a built-in
//! `image`-crate preprocessor.
//!
//! The trha build machines have no OpenCV, so `deepseek-recipe-image` is
//! built with `default-features = false` (no `opencv-preprocess`); this
//! module supplies the [`ImagePreprocessor`] instead, mirroring the OpenCV
//! pipeline's semantics with the pure-Rust `image` crate: decode →
//! low-detail long-side clamp → fit within the max dimension → re-encode
//! as PNG.
//!
//! URL sources fetch through [`ReqwestImageFetcher`]. The default
//! destination filter rejects loopback / private / link-local / ULA
//! addresses (SSRF guard); a payload flag lifts it for deployments that
//! resolve images from internal hosts.
//!
//! Payload dialect:
//!
//! ```json
//! {"sources": [
//!    {"kind": "data_url", "data_url": "data:image/png;base64,…"},
//!    {"kind": "url",      "url": "https://…"},
//!    {"kind": "bytes",    "base64": "…"}
//!  ],
//!  "detail": "high",              // low|high|original|auto（缺省 high）
//!  "allow_private_urls": false}   // 缺省 false：拒内网/环回地址
//! ```

use std::future::Future;
use std::io::Cursor;
use std::net::IpAddr;

use base64::Engine as _;
use serde_json::{json, Value};

use deepseek_recipe_core::multimodal::{ImageDetail, ImageInfo, ImageSource, ImageTokenSpec};
use deepseek_recipe_image::{
    ImageError, ImagePreprocessor, ImageQuota, ImageResolver, PreprocessOptions,
    ReqwestImageFetcher,
};

// ---------------------------------------------------------------------------
// 预处理器（无 OpenCV 的纯 Rust 实现）
// ---------------------------------------------------------------------------

/// Decode → low-detail clamp → fit within max dimension → PNG encode.
struct FfiImagePreprocessor;

impl ImagePreprocessor for FfiImagePreprocessor {
    fn preprocess(
        &self,
        data: Vec<u8>,
        options: PreprocessOptions,
    ) -> impl Future<Output = Result<ImageInfo, ImageError>> + Send {
        let fut = async move {
            // 解码/重采样是 CPU 密集：挪出异步执行线程
            tokio::task::spawn_blocking(move || preprocess_sync(&data, options))
                .await
                .map_err(|e| ImageError::decode(e.to_string()))?
        };
        fut
    }
}

fn preprocess_sync(data: &[u8], options: PreprocessOptions) -> Result<ImageInfo, ImageError> {
    if data.is_empty() {
        return Err(ImageError::empty_image());
    }
    let reader = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| ImageError::decode(e.to_string()))?;
    let mut img = reader
        .decode()
        .map_err(|e| ImageError::decode(e.to_string()))?;

    // 低细节：先按长边钳到 low_detail_max_dimension_px
    if options.detail.is_low() && options.low_detail_max_dimension_px > 0 {
        img = clamp_long_side(img, u32::try_from(options.low_detail_max_dimension_px).unwrap_or(u32::MAX));
    }
    // 高维度上限：等比缩到长边 ≤ max_dimension_px
    if options.max_dimension_px > 0 {
        img = clamp_long_side(img, options.max_dimension_px);
    }

    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| ImageError::encode(e.to_string()))?;
    Ok(ImageInfo {
        data: out.into_inner(),
        width: img.width(),
        height: img.height(),
    })
}

/// 等比缩到长边 ≤ max（只缩不放——放大交给 token-budget fitting）。
fn clamp_long_side(img: image::DynamicImage, max: u32) -> image::DynamicImage {
    let (w, h) = (img.width(), img.height());
    let long = w.max(h);
    if long == 0 || long <= max {
        return img;
    }
    let ratio = f64::from(max) / f64::from(long);
    let nw = ((f64::from(w) * ratio).round() as u32).max(1);
    let nh = ((f64::from(h) * ratio).round() as u32).max(1);
    img.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3)
}

// ---------------------------------------------------------------------------
// SSRF 防护：默认拒内网/环回/链路本地/ULA 地址
// ---------------------------------------------------------------------------

fn is_private_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(v4) => {
            v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
                || v4.is_broadcast() || v4.is_documentation()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.is_unspecified() || v6.is_unique_local()
                || (v6.is_unicast_link_local())
        }
    }
}

// ---------------------------------------------------------------------------
// 载荷解析
// ---------------------------------------------------------------------------

fn parse_detail(v: &Value) -> ImageDetail {
    match v.get("detail").and_then(|d| d.as_str()) {
        Some("low") => ImageDetail::Low,
        Some("original") => ImageDetail::Original,
        Some("auto") => ImageDetail::Auto,
        _ => ImageDetail::High,
    }
}

fn parse_sources(v: &Value) -> Result<Vec<ImageSource>, String> {
    let arr = v
        .get("sources")
        .and_then(|s| s.as_array())
        .ok_or("missing sources array")?;
    let default_detail = parse_detail(v);
    let mut out = Vec::with_capacity(arr.len());
    for s in arr {
        let kind = s.get("kind").and_then(|k| k.as_str()).ok_or("source missing kind")?;
        let path = s.get("path").and_then(|p| p.as_str()).unwrap_or("").to_string();
        // per-source detail 缺省回落到载荷顶层 detail（resolver 内部按
        // source.detail() + limits 推导 PreprocessOptions，调用方不传 options）
        let detail = if s.get("detail").is_some() { parse_detail(s) } else { default_detail };
        let source = match kind {
            "data_url" => {
                let data_url = s.get("data_url").and_then(|d| d.as_str()).ok_or("data_url source missing data_url")?;
                ImageSource::DataUrl {
                    data_url: data_url.to_string(),
                    detail,
                    path,
                }
            }
            "url" => {
                let url = s.get("url").and_then(|u| u.as_str()).ok_or("url source missing url")?;
                ImageSource::Url {
                    url: url.to_string(),
                    detail,
                    path,
                }
            }
            "bytes" => {
                let b64 = s.get("base64").and_then(|b| b.as_str()).ok_or("bytes source missing base64")?;
                let data = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| format!("bytes base64: {e}"))?;
                ImageSource::Bytes {
                    data,
                    detail,
                    path,
                }
            }
            other => return Err(format!("unknown source kind: {other}")),
        };
        out.push(source);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 操作实现
// ---------------------------------------------------------------------------

/// `image_token_spec`：V4.1 图像 token 数学（纯同步，不碰字节）。
pub fn op_image_token_spec(payload: &str) -> Result<Value, Value> {
    let v: Value = serde_json::from_str(payload)
        .map_err(|e| json!({"error": format!("parse payload: {e}")}))?;
    let width = v
        .get("width")
        .and_then(|w| w.as_u64())
        .ok_or_else(|| json!({"error": "missing width"}))?
        as usize;
    let height = v
        .get("height")
        .and_then(|h| h.as_u64())
        .ok_or_else(|| json!({"error": "missing height"}))?
        as usize;

    let spec = ImageTokenSpec::v41();
    let resize = spec
        .calc_resize(width, height)
        .map_err(|e| json!({"error": format!("{e}")}))?;
    Ok(json!({
        "token_count": resize.token_count,
        "best_width": resize.best_width,
        "best_height": resize.best_height,
        "n_llm_h": resize.n_llm_h,
        "n_llm_w": resize.n_llm_w,
        "max_token_count": spec.max_token_count(),
        "is_image_valid": spec.is_image_valid(width, height),
    }))
}

/// `image_resolve`：sources → 预处理后的图像 + 各图 token 数。
pub fn op_image_resolve(payload: &str) -> Result<Value, Value> {
    let v: Value = serde_json::from_str(payload)
        .map_err(|e| json!({"error": format!("parse payload: {e}")}))?;
    let sources = parse_sources(&v).map_err(|e| json!({"error": e}))?;
    let allow_private = v
        .get("allow_private_urls")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);

    let fetcher = ReqwestImageFetcher::new()
        .map_err(|e| json!({"error": e.to_string()}))?;
    let fetcher = if allow_private {
        fetcher
    } else {
        fetcher
            .with_destination_filter(|_host: &str, address: IpAddr| !is_private_address(address))
            .map_err(|e| json!({"error": e.to_string()}))?
    };
    let resolver = ImageResolver::new(fetcher, FfiImagePreprocessor);

    let mut quota = ImageQuota::new();
    let resolved = crate::stream::block_on(resolver.resolve(&sources, &mut quota))
        .map_err(|e| json!({"error": e.to_string()}))?;

    let spec = ImageTokenSpec::v41();
    let images: Vec<Value> = resolved
        .images
        .iter()
        .map(|img| {
            let tokens = spec.calc_token_count(img.width as usize, img.height as usize).unwrap_or(0);
            json!({
                "width": img.width,
                "height": img.height,
                "bytes_len": img.data.len(),
                "token_count": tokens,
            })
        })
        .collect();
    let total_tokens: u64 = images
        .iter()
        .filter_map(|i| i["token_count"].as_u64())
        .sum();
    Ok(json!({
        "images": images,
        "total_tokens": total_tokens,
        "image_count": images.len(),
    }))
}
