use std::path::PathBuf;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    // ORT session lifecycle
    #[error("ONNX runtime setup failed: {0}")]
    OrtSetup(String),
    #[error("load model from `{path}`: {message}")]
    ModelLoad { path: PathBuf, message: String },

    // Input validation
    #[error("token_ids must not be empty")]
    EmptyTokenIds,
    #[error("style vector must have exactly 256 elements, got {0}")]
    StyleLength(usize),
    #[error("speed must be positive, got {0}")]
    NonPositiveSpeed(f32),

    // ORT tensor & inference errors (Display-only because ort::Error is not Send+Sync)
    #[error("build `{name}` tensor: {message}")]
    TensorBuild { name: &'static str, message: String },
    #[error("inference run: {0}")]
    InferenceRun(String),
    #[error("extract `{name}` tensor: {message}")]
    TensorExtract { name: &'static str, message: String },
    #[error("ONNX session mutex poisoned (a previous inference panicked)")]
    SessionPoisoned,
    #[error("inference task panicked: {0}")]
    TaskPanic(#[from] tokio::task::JoinError),

    // WAV encoding (`hound::Error` is Send+Sync, so #[from] works)
    #[error("WAV encode: {0}")]
    WavEncode(#[from] hound::Error),

    // MP3 encoding via mp3lame-encoder (linked, no subprocess)
    #[error("MP3 init failed (lame returned null)")]
    Mp3Init,
    #[error("MP3 build: {0}")]
    Mp3Build(#[from] mp3lame_encoder::BuildError),
    #[error("MP3 encode: {0}")]
    Mp3Encode(#[from] mp3lame_encoder::EncodeError),
}

impl<T> From<ort::Error<T>> for InferenceError {
    fn from(value: ort::Error<T>) -> Self {
        InferenceError::OrtSetup(value.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PhonemizerError {
    #[error("read vocab file `{path}`: {source}")]
    VocabRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse vocab JSON: {0}")]
    VocabParse(#[from] serde_json::Error),
    #[error("BOS token '$' missing from vocab")]
    MissingBos,
    #[error("EOS token ';' missing from vocab")]
    MissingEos,
    #[error("espeak-ng init: {0}")]
    EspeakInit(String),
    #[error("espeak-ng: {0}")]
    EspeakFailed(String),
    #[error("misaki-rs: {0}")]
    MisakiError(#[from] misaki_rs::g2p::G2PError),
    #[error("unknown PHONEMIZER backend: '{0}' (valid options: \"espeak\", \"misaki\")")]
    UnknownBackend(String),
}

#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("open voices directory `{path}`: {source}")]
    DirOpen {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("read directory entry: {0}")]
    DirEntry(#[source] std::io::Error),
    #[error("invalid voice filename: `{0}`")]
    InvalidFilename(PathBuf),
    #[error("read voice file `{path}`: {source}")]
    FileRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("voice file `{path}` has {len} bytes, which is not a multiple of 256 × 4")]
    InvalidBinSize { path: PathBuf, len: usize },
    #[error("no .bin voice files found in `{0}`")]
    EmptyVoicesDir(PathBuf),
    #[error("unknown voice: '{name}'")]
    UnknownVoice { name: String },
    #[error("voice '{name}' has zero style embeddings")]
    ZeroStyles { name: String },
}

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("bad request: {0}")]
    Validation(String),
    #[error("not found")]
    NotFound,
    #[error("conflict: {0}")]
    Conflict(String),
    #[error(transparent)]
    Voice(#[from] VoiceError),
    #[error(transparent)]
    Phonemizer(#[from] PhonemizerError),
    #[error(transparent)]
    Inference(#[from] InferenceError),
    #[error("task panicked: {0}")]
    TaskPanic(#[from] tokio::task::JoinError),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::Validation(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound => StatusCode::NOT_FOUND,
            AppError::Conflict(_) => StatusCode::CONFLICT,
            AppError::Voice(VoiceError::UnknownVoice { .. }) => StatusCode::NOT_FOUND,
            AppError::Voice(_)
            | AppError::Phonemizer(_)
            | AppError::Inference(_)
            | AppError::TaskPanic(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        tracing::error!("request error: {:#}", self);
        (status, self.to_string()).into_response()
    }
}
