use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use redis::AsyncCommands;
use serde::Serialize;
use std::sync::Arc;

use crate::{AppState, error::AppError, inference::encode_mp3};

#[derive(Serialize)]
pub struct VoicesResponse {
    pub voices: Vec<String>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub redis: &'static str,
    pub voices_loaded: usize,
}

/// `GET /health` — readiness probe: checks Redis connectivity and model load.
/// Returns 200 when healthy, 503 when Redis is unreachable or no voices loaded.
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let redis_ok = {
        let mut conn = state.redis.clone();
        conn.set_ex::<_, _, ()>("health:ping", "1", 10)
            .await
            .is_ok()
    };
    let voices_loaded = state.voices.voice_names().len();
    let healthy = redis_ok && voices_loaded > 0;

    let body = Json(HealthResponse {
        status: if healthy { "ok" } else { "degraded" },
        redis: if redis_ok { "ok" } else { "error" },
        voices_loaded,
    });

    if healthy {
        (StatusCode::OK, body).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
    }
}

/// `GET /voices` — return all loaded voice names.
pub async fn voices(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let voices = state
        .voices
        .voice_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    Json(VoicesResponse { voices })
}



pub fn validate_synth(text: &str, speed: f32) -> Result<(), AppError> {
    if text.trim().is_empty() {
        return Err(AppError::Validation("text must not be empty".into()));
    }
    if speed <= 0.0 || speed > 4.0 {
        return Err(AppError::Validation("speed must be in (0, 4]".into()));
    }
    Ok(())
}

pub async fn run_synthesis(
    state: Arc<AppState>,
    text: &str,
    voice: &str,
    speed: f32,
) -> Result<Vec<u8>, AppError> {
    let text = text.to_owned();
    let cloned_state = Arc::clone(&state);
    let token_ids =
        tokio::task::spawn_blocking(move || cloned_state.phonemizer.phonemize(&text)).await??;

    let style = state.voices.get_style(voice, token_ids.len())?;
    let samples = state.engine.synthesize(&token_ids, &style, speed).await?;
    Ok(encode_mp3(&samples, 24_000)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_empty_text_err() {
        assert!(validate_synth("", 1.0).is_err());
    }

    #[test]
    fn synth_whitespace_err() {
        assert!(validate_synth("   ", 1.0).is_err());
    }

    #[test]
    fn synth_zero_speed_err() {
        assert!(validate_synth("hello", 0.0).is_err());
    }

    #[test]
    fn synth_negative_speed_err() {
        assert!(validate_synth("hello", -1.0).is_err());
    }

    #[test]
    fn synth_speed_over_max() {
        assert!(validate_synth("hello", 4.001).is_err());
    }

    #[test]
    fn synth_speed_at_max() {
        assert!(validate_synth("hello", 4.0).is_ok());
    }

    #[test]
    fn synth_valid_request() {
        assert!(validate_synth("hello world", 1.0).is_ok());
    }

    #[test]
    fn synth_empty_text_msg() {
        let err = validate_synth("", 1.0).unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }

    #[test]
    fn synth_bad_speed_msg() {
        let err = validate_synth("hi", 0.0).unwrap_err();
        assert!(err.to_string().contains("speed"), "got: {err}");
    }
}
