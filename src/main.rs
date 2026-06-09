use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use tts_rs::{
    AppState,
    error::{InferenceError, PhonemizerError, VoiceError},
    handlers,
    inference::TtsEngine,
    phonemizer::Phonemizer,
    rpc,
    voices::VoiceBank,
};

#[derive(Debug, thiserror::Error)]
enum StartupError {
    #[error(transparent)]
    Inference(#[from] InferenceError),
    #[error(transparent)]
    Phonemizer(#[from] PhonemizerError),
    #[error(transparent)]
    Voice(#[from] VoiceError),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("Redis: {0}")]
    Redis(#[from] redis::RedisError),
}

use axum::{
    Router,
    extract::Request,
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use std::{net::SocketAddr, path::Path, sync::Arc};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

fn api_key_gate(expected: &str, provided: &str) -> bool {
    expected.is_empty() || provided == expected
}

async fn require_api_key(req: Request, next: Next) -> Result<Response, StatusCode> {
    let expected = std::env::var("TTS_RS_API_KEY").unwrap_or_default();
    let provided = req
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if !api_key_gate(&expected, provided) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(next.run(req).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_key_allows_any() {
        assert!(api_key_gate("", "anything"));
        assert!(api_key_gate("", ""));
    }

    #[test]
    fn correct_key_is_allowed() {
        assert!(api_key_gate("secret-key", "secret-key"));
    }

    #[test]
    fn wrong_key_is_denied() {
        assert!(!api_key_gate("secret-key", "wrong-key"));
    }

    #[test]
    fn missing_key_denied() {
        assert!(!api_key_gate("secret-key", ""));
    }

    #[test]
    fn key_exact_match() {
        assert!(!api_key_gate("secret", "secret-extra"));
        assert!(!api_key_gate("secret-extra", "secret"));
    }
}

#[tokio::main]
async fn main() -> Result<(), StartupError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tts_rs=info,tower_http=info".into()),
        )
        .init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "./models".into());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".into())
        .parse()
        .expect("PORT must be a valid u16");

    let model_dir = Path::new(&model_dir);

    tracing::info!("ORT build: {}", ort::info());

    let model_file = std::env::var("MODEL_FILE").unwrap_or_else(|_| "model_quantized.onnx".into());
    tracing::info!("loading ONNX model ({model_file})…");
    let engine = TtsEngine::load(&model_dir.join(&model_file))?;
    tracing::info!("ONNX sessions loaded (CPU)");

    tracing::info!("loading phonemizer vocab…");
    let phonemizer = Phonemizer::load_from_env(&model_dir.join("vocab.json"))?;

    tracing::info!("loading voice bank…");
    let voices = VoiceBank::load(&model_dir.join("voices"))?;

    tracing::info!("connecting to Redis…");
    let redis_url = std::env::var("REDIS_URL").expect("REDIS_URL is required");
    let redis = redis::aio::ConnectionManager::new(redis::Client::open(redis_url)?).await?;
    tracing::info!("Redis connected");

    tracing::info!("initialising S3 client…");
    let s3_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .load()
        .await;
    let s3 = aws_sdk_s3::Client::new(&s3_config);
    let bucket = std::env::var("BUCKET_NAME").expect("BUCKET_NAME is required");

    let state = Arc::new(AppState {
        engine,
        phonemizer,
        voices,
        redis,
        s3,
        bucket,
    });

    let protected = Router::new()
        .route("/rpc", post(rpc::rpc_handler))
        .route("/voices", get(handlers::voices))
        .layer(middleware::from_fn(require_api_key))
        .with_state(Arc::clone(&state));

    let app = Router::new()
        .merge(protected)
        .route("/health", get(handlers::health).post(handlers::health))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("tts-rs listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
