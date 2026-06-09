/// Downloads Kokoro-82M ONNX model files from HuggingFace into `./models/`.
///
/// Run with:
///   cargo run --bin download-model
///
/// Files downloaded:
///   - `models/model_quantized.onnx` (~82 MB — INT8 dynamic quant; default)
///   - `models/model.onnx` (326 MB — FP32; set KOKORO_MODEL_FILE=model.onnx to use)
///   - `models/vocab.json` (extracted from tokenizer.json; tokenizer.json itself is not kept)
///   - `models/voices/<name>.bin` (all 54 voice style embeddings)
///
/// Override the model variant:
///   KOKORO_MODEL_FILE=model.onnx cargo run --bin download-model
///   KOKORO_MODEL_FILE=model_q4.onnx cargo run --bin download-model
///
/// Skips any file that already exists on disk (idempotent).
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{Url, blocking::Client};
use std::{collections::HashMap, fs, io::Write, path::Path};

/// Raw files in the onnx/ subdirectory (model weights).
const ONNX_BASE: &str =
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/";

/// Raw files at the repo root (tokenizer, voices).
const ROOT_BASE: &str = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/";

/// All voices from the HuggingFace repo `voices/` directory.
/// Source: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices
const VOICES: &[&str] = &[
    // American English — Female
    "af",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    // American English — Male
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    // British English — Female
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    // British English — Male
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    // European Spanish
    "ef_dora",
    "em_alex",
    "em_santa",
    // French
    "ff_siwis",
    // Hindi
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    // Italian
    "if_sara",
    "im_nicola",
    // Japanese
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    // Portuguese
    "pf_dora",
    "pm_alex",
    "pm_santa",
    // Chinese (Mandarin)
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
];

#[derive(Debug, thiserror::Error)]
enum DownloadError {
    #[error("filesystem: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("URL parse: {0}")]
    Url(#[from] url::ParseError),
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  {spinner:.cyan} {msg:<30} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
    )
    .unwrap()
    .progress_chars("=>-")
}

/// Download `url` to `dest`, streaming with a per-file progress bar.
/// Skips if `dest` already exists (idempotent).
fn download(client: &Client, url: &str, dest: &Path) -> Result<(), DownloadError> {
    if dest.exists() {
        println!("  skip (exists): {}", dest.display());
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut response = client.get(url).send()?.error_for_status()?;

    let total = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(bar_style());
    pb.set_message(
        dest.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("…")
            .to_string(),
    );

    let mut file = fs::File::create(dest)?;

    let mut buf = [0u8; 65_536];
    loop {
        use std::io::Read;
        let n = response.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }

    pb.finish_with_message(format!(
        "✓ {}",
        dest.file_name().and_then(|n| n.to_str()).unwrap_or("done")
    ));
    Ok(())
}

/// Download `tokenizer.json` from the repo root and extract the flat `model.vocab`
/// map, saving it as `vocab.json` that `Phonemizer::load` expects.
///
/// HuggingFace tokenizer.json nests the vocab under `model.vocab`; our phonemizer
/// reads a top-level `{ "<char>": id }` flat object.
fn download_vocab(client: &Client, tokenizer_url: &str, dest: &Path) -> Result<(), DownloadError> {
    if dest.exists() {
        println!("  skip (exists): {}", dest.display());
        return Ok(());
    }

    println!("  downloading vocab from tokenizer.json…");

    let body = client
        .get(tokenizer_url)
        .send()?
        .error_for_status()?
        .text()?;

    // tokenizer.json → { "model": { "vocab": { "<char>": id, … } } }
    #[derive(serde::Deserialize)]
    struct TokenizerJson {
        model: TokenizerModel,
    }
    #[derive(serde::Deserialize)]
    struct TokenizerModel {
        vocab: HashMap<String, i64>,
    }

    let tokenizer: TokenizerJson = serde_json::from_str(&body)?;

    let vocab_json = serde_json::to_string_pretty(&tokenizer.model.vocab)?;

    fs::write(dest, vocab_json.as_bytes())?;

    println!("  ✓ vocab.json ({} entries)", tokenizer.model.vocab.len());
    Ok(())
}

fn main() -> Result<(), DownloadError> {
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let dest = Path::new("models");
    fs::create_dir_all(dest)?;
    fs::create_dir_all(dest.join("voices"))?;

    let model_file =
        std::env::var("KOKORO_MODEL_FILE").unwrap_or_else(|_| "model_quantized.onnx".into());

    println!(
        "=== Kokoro-82M model download ===\nDestination: {}\nModel: {model_file}\n",
        dest.canonicalize().unwrap_or(dest.to_path_buf()).display()
    );

    let onnx_base = Url::parse(ONNX_BASE)?;
    let root_base = Url::parse(ROOT_BASE)?;

    println!("[1/3] ONNX model ({model_file})");
    download(
        &client,
        onnx_base.join(&model_file)?.as_str(),
        &dest.join(&model_file),
    )?;

    println!("\n[2/3] Vocab (extracted from tokenizer.json)");
    download_vocab(
        &client,
        root_base.join("tokenizer.json")?.as_str(),
        &dest.join("vocab.json"),
    )?;

    println!("\n[3/3] Voice embeddings ({} voices)", VOICES.len());
    for voice in VOICES {
        let voice_url = root_base.join(&format!("voices/{voice}.bin"))?;
        download(
            &client,
            voice_url.as_str(),
            &dest.join("voices").join(format!("{voice}.bin")),
        )?;
    }

    println!("\nDone. Run `cargo run --bin tts-rs` to start the server.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn download_skips_existing_file() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("existing.bin");
        std::fs::File::create(&dest)
            .unwrap()
            .write_all(b"data")
            .unwrap();

        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(1))
            .build()
            .unwrap();

        let result = download(&client, "http://0.0.0.0:1/nonexistent", &dest);
        assert!(result.is_ok(), "expected skip; got {result:?}");
    }

    #[test]
    fn download_vocab_skips_existing_file() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("vocab.json");
        std::fs::write(&dest, b"{}").unwrap();

        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(1))
            .build()
            .unwrap();

        let result = download_vocab(&client, "http://0.0.0.0:1/nonexistent", &dest);
        assert!(result.is_ok(), "expected skip; got {result:?}");
    }

    #[test]
    fn voices_list_is_nonempty() {
        assert!(!VOICES.is_empty());
    }

    #[test]
    fn voices_list_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for v in VOICES {
            assert!(seen.insert(*v), "duplicate voice: {v}");
        }
    }

    #[test]
    fn onnx_base_uses_resolve() {
        assert!(
            ONNX_BASE.contains("/resolve/main/"),
            "must use resolve, not blob"
        );
    }

    #[test]
    fn root_base_uses_resolve() {
        assert!(
            ROOT_BASE.contains("/resolve/main/"),
            "must use resolve, not blob"
        );
    }

    #[test]
    fn default_model_is_quantized() {
        let model_filename =
            std::env::var("KOKORO_MODEL_FILE").unwrap_or_else(|_| "model_quantized.onnx".into());
        // Default must not be FP32 — the 2 GB Fly machine can't hold a full-precision session.
        assert!(
            model_filename.contains("quantized")
                || model_filename.contains("q4")
                || model_filename.contains("q8")
                || model_filename.contains("uint8"),
            "expected a quantized model by default, got: {model_filename}",
        );
    }

    #[test]
    fn url_join_builds_correct_onnx_url() {
        let base = Url::parse(ONNX_BASE).unwrap();
        let url = base.join("model.onnx").unwrap();
        assert!(url.as_str().ends_with("/onnx/model.onnx"));
        assert!(url.as_str().contains("/resolve/main/"));
    }

    #[test]
    fn url_join_builds_correct_voice_url() {
        let base = Url::parse(ROOT_BASE).unwrap();
        let url = base.join("voices/af_heart.bin").unwrap();
        assert!(url.as_str().ends_with("/voices/af_heart.bin"));
    }
}
