pub mod error;
pub mod handlers;
pub mod inference;
pub mod phonemizer;
pub mod rpc;
pub mod voices;

/// Shared application state injected into every handler via `State<Arc<AppState>>`.
pub struct AppState {
    pub engine: inference::TtsEngine,
    pub phonemizer: phonemizer::Phonemizer,
    pub voices: voices::VoiceBank,
    /// Async Redis connection (auto-reconnects on drop).
    pub redis: redis::aio::ConnectionManager,
    pub s3: aws_sdk_s3::Client,
    pub bucket: String,
}
