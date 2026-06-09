use async_channel::{Receiver, Sender};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{PrimitiveTensorElementType, Tensor},
};
use std::{fmt::Debug, path::Path};

use crate::error::InferenceError;

fn build_tensor<T>(
    name: &'static str,
    shape: Vec<usize>,
    data: Vec<T>,
) -> Result<Tensor<T>, InferenceError>
where
    T: PrimitiveTensorElementType + Debug + Clone + 'static,
{
    Tensor::<T>::from_array((shape, data.into_boxed_slice())).map_err(|e| {
        InferenceError::TensorBuild {
            name,
            message: e.to_string(),
        }
    })
}

/// Kept as a plain synchronous function so it can run inside
/// `spawn_blocking` without an inner closure, and can be called from tests
/// without a Tokio runtime.
fn run_inference(
    session: &mut Session,
    token_ids: Vec<i64>,
    style: Vec<f32>,
    speed: f32,
) -> Result<Vec<f32>, InferenceError> {
    let seq_len = token_ids.len();

    let ids_tensor = build_tensor("input_ids", vec![1, seq_len], token_ids)?;
    let style_tensor = build_tensor("style", vec![1, 256], style)?;
    let speed_tensor = build_tensor("speed", vec![1], vec![speed])?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids_tensor,
            "style"     => style_tensor,
            "speed"     => speed_tensor,
        ])
        .map_err(|e| InferenceError::InferenceRun(e.to_string()))?;

    let (_, waveform) = outputs["waveform"]
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::TensorExtract {
            name: "waveform",
            message: e.to_string(),
        })?;

    let pcm = waveform.to_vec(); // owned copy — releases borrow on outputs
    drop(outputs); // explicit: releases ort's internal borrow on session
    Ok(pcm)
}

/// Returns the session to the pool on drop, even if `run_inference` panics.
///
/// With `pool_size = 1` a panic that bypasses the normal `pool_tx.try_send`
/// call would permanently drain the pool, deadlocking all future synthesis
/// requests. The RAII guard ensures the session is always returned.
struct SessionGuard {
    session: Option<Session>,
    pool_tx: Sender<Session>,
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if let Some(s) = self.session.take() {
            let _ = self.pool_tx.try_send(s);
        }
    }
}

/// Loaded Kokoro ONNX session pool. Thread-safe — share via `Arc<TtsEngine>`.
///
/// N sessions are pre-loaded and stored in a bounded async channel. Each
/// `synthesize` call checks out a session via `recv().await` (yields the
/// tokio task instead of blocking), runs inference in `spawn_blocking`, then
/// returns the session unconditionally. Allows up to N concurrent GPU inferences.
pub struct TtsEngine {
    pool_rx: Receiver<Session>,
    pool_tx: Sender<Session>,
}

impl std::fmt::Debug for TtsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsEngine")
            .field("pool_capacity", &self.pool_rx.capacity())
            .finish_non_exhaustive()
    }
}

impl TtsEngine {
    /// Load N ONNX sessions from `model_path` into a bounded channel pool.
    ///
    /// Pool size defaults to 4; override with `SESSION_POOL_SIZE` env var.
    /// Applies Level 3 graph optimisations (constant folding, node fusion).
    pub fn load(model_path: &Path) -> Result<Self, InferenceError> {
        let pool_size: usize = std::env::var("SESSION_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);

        let (pool_tx, pool_rx) = async_channel::bounded(pool_size);

        for i in 0..pool_size {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_inter_threads(1)?
                .with_intra_threads(1)?
                // Disable ORT's growing memory arena. Without this, ORT caches
                // one arena allocation per unique tensor shape (different text
                // lengths = different seq_len) and never releases those pages to
                // the OS, causing RSS to climb indefinitely over time.
                .with_memory_pattern(false)?
                .commit_from_file(model_path)
                .map_err(|e| InferenceError::ModelLoad {
                    path: model_path.to_path_buf(),
                    message: e.to_string(),
                })?;
            tracing::info!(
                "ONNX session {}/{} loaded from {}",
                i + 1,
                pool_size,
                model_path.display()
            );
            // Channel is bounded to pool_size and starts empty, so this never fails.
            pool_tx
                .try_send(session)
                .expect("pool channel full at init — logic error");
        }

        tracing::info!("session pool ready ({pool_size} sessions)");
        Ok(Self { pool_rx, pool_tx })
    }

    /// Run TTS inference.
    ///
    /// # Arguments
    /// - `token_ids` — phoneme tokens `[seq_len]`; must be non-empty
    /// - `style`     — voice embedding `[256]` f32
    /// - `speed`     — playback speed multiplier (1.0 = normal)
    ///
    /// # Returns
    /// Raw PCM samples at 24 kHz (f32, range ~[-1, 1]).
    pub async fn synthesize(
        &self,
        token_ids: &[i64],
        style: &[f32],
        speed: f32,
    ) -> Result<Vec<f32>, InferenceError> {
        if token_ids.is_empty() {
            return Err(InferenceError::EmptyTokenIds);
        }
        if style.len() != 256 {
            return Err(InferenceError::StyleLength(style.len()));
        }
        if speed <= 0.0 {
            return Err(InferenceError::NonPositiveSpeed(speed));
        }

        // Async checkout — yields the tokio task if all sessions are busy.
        let session = self
            .pool_rx
            .recv()
            .await
            .map_err(|_| InferenceError::SessionPoisoned)?;

        // Clone inputs so they can be moved into the blocking closure
        // (borrows cannot cross spawn_blocking).
        let token_ids = token_ids.to_vec();
        let style = style.to_vec();
        let pool_tx = self.pool_tx.clone();

        // GPU inference runs on a blocking thread — session.run() is synchronous.
        // SessionGuard returns the session to the pool even if run_inference panics.
        tokio::task::spawn_blocking(move || {
            let mut guard = SessionGuard {
                session: Some(session),
                pool_tx,
            };
            run_inference(
                guard
                    .session
                    .as_mut()
                    .expect("guard holds session until drop"),
                token_ids,
                style,
                speed,
            )
        })
        .await?
    }
}

/// Encode raw f32 PCM samples (range [-1, 1]) to MP3 bytes (128 kbps CBR).
///
/// Uses LAME via `mp3lame-encoder` — linked directly, no subprocess.
pub fn encode_mp3(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, InferenceError> {
    use mp3lame_encoder::{Builder, FlushNoGap, MonoPcm, Quality};

    let mut builder = Builder::new().ok_or(InferenceError::Mp3Init)?;
    builder.set_sample_rate(sample_rate)?;
    builder.set_num_channels(1)?;
    builder.set_quality(Quality::Good)?;

    let mut encoder = builder.build()?;

    let pcm: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();

    let mut mp3 = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));

    let n = encoder.encode(MonoPcm(&pcm), mp3.spare_capacity_mut())?;
    // SAFETY: encode() wrote `n` initialised bytes into spare capacity.
    unsafe { mp3.set_len(mp3.len() + n) };

    let n = encoder.flush::<FlushNoGap>(mp3.spare_capacity_mut())?;
    // SAFETY: flush() wrote `n` initialised bytes into spare capacity.
    unsafe { mp3.set_len(mp3.len() + n) };

    Ok(mp3)
}

/// Encode raw f32 PCM samples (range [-1, 1]) to 16-bit PCM WAV bytes.
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, InferenceError> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::io::Cursor;

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut buf = Cursor::new(Vec::new());
    {
        // hound::Error is #[from] in InferenceError::WavEncode, so ? converts directly.
        let mut writer = WavWriter::new(&mut buf, spec)?;
        for &s in samples {
            let pcm = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(pcm)?;
        }
        writer.finalize()?;
    }

    Ok(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_wav_produces_riff_header() {
        let silence = vec![0.0f32; 24_000];
        let bytes = encode_wav(&silence, 24_000).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
    }

    #[test]
    fn encode_wav_correct_sample_count() {
        let n_samples = 480;
        let signal = vec![0.5f32; n_samples];
        let bytes = encode_wav(&signal, 24_000).unwrap();
        // Standard PCM WAV: 44-byte header + 2 bytes per i16 sample
        assert_eq!(bytes.len(), 44 + n_samples * 2);
    }

    #[test]
    fn wav_clamps_range() {
        let signal = vec![2.0f32, -2.0, 0.0];
        assert!(encode_wav(&signal, 24_000).is_ok());
    }

    #[test]
    fn encode_wav_empty_samples_ok() {
        let bytes = encode_wav(&[], 24_000).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
    }

    #[test]
    fn nonexistent_model_error() {
        let err = TtsEngine::load(Path::new("/nonexistent/model.onnx")).unwrap_err();
        assert!(
            matches!(err, InferenceError::ModelLoad { .. }),
            "expected ModelLoad, got {err:?}",
        );
    }

    #[test]
    fn empty_tokens_error_message() {
        let err = InferenceError::EmptyTokenIds;
        assert_eq!(err.to_string(), "token_ids must not be empty");
    }

    #[test]
    fn style_length_error_message() {
        let err = InferenceError::StyleLength(42);
        assert_eq!(
            err.to_string(),
            "style vector must have exactly 256 elements, got 42"
        );
    }

    #[test]
    fn non_positive_speed_error_message() {
        let err = InferenceError::NonPositiveSpeed(0.0);
        assert_eq!(err.to_string(), "speed must be positive, got 0");
    }

    #[test]
    fn tensor_build_has_name() {
        let err = InferenceError::TensorBuild {
            name: "input_ids",
            message: "bad shape".into(),
        };
        assert!(err.to_string().contains("input_ids"));
    }

    #[test]
    fn model_error_has_path() {
        let err = InferenceError::ModelLoad {
            path: "/tmp/model.onnx".into(),
            message: "file not found".into(),
        };
        assert!(err.to_string().contains("/tmp/model.onnx"));
    }

    #[test]
    fn mp3_header_format() {
        let silence = vec![0.0f32; 24_000];
        let bytes = encode_mp3(&silence, 24_000).unwrap();
        assert!(!bytes.is_empty());
        // MP3 frames start with 0xFF 0xFB/0xFA/0xF3 sync word,
        // or ID3 tag (0x49 0x44 0x33 = "ID3") prepended by LAME.
        let starts_sync = bytes[0] == 0xFF && (bytes[1] & 0xE0 == 0xE0);
        let starts_id3 = bytes.starts_with(b"ID3");
        assert!(
            starts_sync || starts_id3,
            "unexpected MP3 header: {:02x?}",
            &bytes[..4]
        );
    }

    #[test]
    fn mp3_clamps_range() {
        let signal = vec![2.0f32, -2.0, 0.0];
        assert!(encode_mp3(&signal, 24_000).is_ok());
    }

    #[test]
    fn encode_mp3_empty_samples_ok() {
        let bytes = encode_mp3(&[], 24_000).unwrap();
        // LAME may produce a small flush frame even for empty input.
        assert!(
            bytes.len() < 1024,
            "unexpected large output for empty input"
        );
    }
}
