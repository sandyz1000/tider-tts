use aws_sdk_s3::primitives::ByteStream;
use axum::{Json, extract::State};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{AppState, handlers};

#[derive(Deserialize)]
pub enum RpcMethod {
    #[serde(rename = "tts.submit")]
    Submit,
    #[serde(rename = "tts.status")]
    Status,
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: RpcMethod,
    pub params: serde_json::Value,
}

#[derive(Serialize)]
pub struct RpcResponse {
    pub jsonrpc: &'static str,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

#[derive(Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
}

const ERR_INVALID_REQUEST: i32 = -32600;
const ERR_METHOD_NOT_FOUND: i32 = -32601;
const ERR_INVALID_PARAMS: i32 = -32602;
const ERR_SYNTHESIS_FAILED: i32 = -32000;
const ERR_JOB_NOT_FOUND: i32 = -32001;

fn ok(id: serde_json::Value, result: serde_json::Value) -> Json<RpcResponse> {
    Json(RpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    })
}

fn err(id: serde_json::Value, code: i32, message: impl Into<String>) -> Json<RpcResponse> {
    Json(RpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(RpcError {
            code,
            message: message.into(),
        }),
    })
}

/// `POST /rpc` — JSON-RPC 2.0 dispatch over a single HTTP endpoint.
/// Auth (`x-api-key`) is enforced at the Axum layer — not inside the envelope.
pub async fn rpc_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RpcRequest>,
) -> Json<RpcResponse> {
    if req.jsonrpc != "2.0" {
        return err(req.id, ERR_INVALID_REQUEST, "jsonrpc must be \"2.0\"");
    }
    match req.method {
        RpcMethod::Submit => submit(state, req.id, req.params).await,
        RpcMethod::Status => status(state, req.id, req.params).await,
        RpcMethod::Unknown => err(req.id, ERR_METHOD_NOT_FOUND, "method not found"),
    }
}

#[derive(Deserialize)]
struct SubmitParams {
    text: String,
    voice: String,
    #[serde(default = "default_speed")]
    speed: f32,
    s3_key: String,
}

fn default_speed() -> f32 {
    1.0
}

#[derive(Deserialize)]
struct StatusParams {
    job_id: String,
}

async fn submit(
    state: Arc<AppState>,
    id: serde_json::Value,
    params: serde_json::Value,
) -> Json<RpcResponse> {
    let p: SubmitParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return err(id, ERR_INVALID_PARAMS, format!("invalid params: {e}")),
    };

    if let Err(e) = handlers::validate_synth(&p.text, p.speed) {
        return err(id, ERR_INVALID_PARAMS, e.to_string());
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    let instance_id = std::env::var("FLY_MACHINE_ID").unwrap_or_else(|_| "local".into());

    let redis_key = format!("tts:job:{job_id}");
    if let Err(e) = set_job_status(&state, &redis_key, "processing", None).await {
        return err(id, ERR_SYNTHESIS_FAILED, format!("redis error: {e}"));
    }

    // Spawn async synthesis — response returns immediately
    let state2 = Arc::clone(&state);
    let jid = job_id.clone();
    tokio::spawn(async move {
        let key = format!("tts:job:{jid}");

        let mp3 =
            match handlers::run_synthesis(Arc::clone(&state2), &p.text, &p.voice, p.speed).await {
                Ok(mp3) => mp3,
                Err(e) => {
                    let _ = set_job_status(&state2, &key, "failed", Some(&e.to_string())).await;
                    return;
                }
            };

        match upload_to_s3(&state2, &p.s3_key, mp3).await {
            Ok(_) => {
                let _ = set_job_status(&state2, &key, "done", None).await;
            }
            Err(e) => {
                let _ = set_job_status(&state2, &key, "failed", Some(&e)).await;
            }
        }
    });

    ok(
        id,
        serde_json::json!({ "job_id": job_id, "instance_id": instance_id }),
    )
}

async fn status(
    state: Arc<AppState>,
    id: serde_json::Value,
    params: serde_json::Value,
) -> Json<RpcResponse> {
    let p: StatusParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return err(id, ERR_INVALID_PARAMS, format!("invalid params: {e}")),
    };

    let redis_key = format!("tts:job:{}", p.job_id);
    let mut conn = state.redis.clone();
    let raw: Option<String> = match conn.get(&redis_key).await {
        Ok(v) => v,
        Err(e) => return err(id, ERR_SYNTHESIS_FAILED, format!("redis error: {e}")),
    };

    let raw = match raw {
        Some(v) => v,
        None => return err(id, ERR_JOB_NOT_FOUND, "job not found or expired"),
    };

    let stored: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            return err(
                id,
                ERR_SYNTHESIS_FAILED,
                format!("invalid stored state: {e}"),
            );
        }
    };

    ok(id, stored)
}

async fn set_job_status(
    state: &AppState,
    key: &str,
    status: &str,
    error: Option<&str>,
) -> Result<(), String> {
    let value = if let Some(msg) = error {
        serde_json::json!({ "status": status, "error": msg })
    } else {
        serde_json::json!({ "status": status })
    };
    let serialized = serde_json::to_string(&value).map_err(|e| e.to_string())?;
    let mut conn = state.redis.clone();
    conn.set_ex::<_, _, ()>(key, serialized, 7200)
        .await
        .map_err(|e| e.to_string())
}

async fn upload_to_s3(state: &AppState, s3_key: &str, mp3: Vec<u8>) -> Result<(), String> {
    state
        .s3
        .put_object()
        .bucket(&state.bucket)
        .key(s3_key)
        .body(ByteStream::from(mp3))
        .content_type("audio/mpeg")
        .send()
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}
