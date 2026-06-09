# tider-tts

Kokoro ONNX TTS server for [Tider](https://www.tider.studio). Accepts JSON-RPC 2.0 requests, synthesizes speech via [Kokoro-82M](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX), and uploads the resulting MP3 directly to S3.

## Requirements

- Rust 1.87+
- `libespeak-ng` (default phonemizer backend)
  - macOS: `brew install espeak-ng`
  - Debian/Ubuntu: `apt install libespeak-ng-dev`
- `libmp3lame` (MP3 encoding via LAME)
  - macOS: `brew install lame`
  - Debian/Ubuntu: `apt install libmp3lame-dev`
- Redis instance (for job state)
- S3-compatible storage (Tigris on Fly.io, or MinIO for local dev)

## Quick start

```bash
# 1. Download model files (~600 MB total)
cargo run --bin download-model

# 2. Copy and fill environment variables
cp .env.example .env

# 3. Run the server
MODEL_DIR=./models REDIS_URL=redis://localhost:6379 cargo run --bin tts-rs
```

The server listens on `PORT` (default `8080`).

## Environment variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `REDIS_URL` | yes | — | Redis connection URL |
| `BUCKET_NAME` | yes | — | S3 bucket for MP3 uploads |
| `AWS_ACCESS_KEY_ID` | yes | — | S3 credentials |
| `AWS_SECRET_ACCESS_KEY` | yes | — | S3 credentials |
| `AWS_ENDPOINT_URL_S3` | no | AWS default | Override for Tigris / MinIO |
| `TTS_RS_API_KEY` | no | *(open)* | Bearer key enforced on `/rpc` and `/voices`; unset = no auth |
| `MODEL_DIR` | no | `./models` | Directory with `model_quantized.onnx`, `vocab.json`, `voices/` |
| `MODEL_FILE` | no | `model_quantized.onnx` | ONNX model filename inside `MODEL_DIR` |
| `SESSION_POOL_SIZE` | no | `2` | Number of concurrent ONNX inference sessions |
| `PHONEMIZER` | no | `espeak` | `espeak` or `misaki` |
| `PORT` | no | `8080` | HTTP listen port |

## API

### `GET /health` · `POST /health`

Returns `{"status":"ok","redis":"ok","voices_loaded":<n>}` (HTTP 200) when healthy, or HTTP 503 when Redis is unreachable or no voices are loaded.

### `GET /voices`  *(requires `x-api-key`)*

Returns `{"voices":["af_heart","am_adam",…]}` — all loaded voice names sorted.

### `POST /rpc`  *(requires `x-api-key`)*

JSON-RPC 2.0 endpoint. Two methods:

#### `tts.submit`

Enqueues a synthesis job. Returns immediately.

```json
{
  "jsonrpc": "2.0", "id": 1,
  "method": "tts.submit",
  "params": {
    "text": "Hello world",
    "voice": "af_heart",
    "speed": 1.0,
    "s3_key": "audio/user123/chunk-0.mp3"
  }
}
```

Response:

```json
{"jsonrpc":"2.0","id":1,"result":{"job_id":"<uuid>","instance_id":"<fly-machine-id>"}}
```

#### `tts.status`

Polls job state. Pin to the submitting machine with `fly-force-instance-id: <instance_id>`.

```json
{"jsonrpc":"2.0","id":2,"method":"tts.status","params":{"job_id":"<uuid>"}}
```

Response (when done):

```json
{"jsonrpc":"2.0","id":2,"result":{"status":"done"}}
```

Possible `status` values: `processing` · `done` · `failed`. Job state expires after 2 hours.

## Utility binaries

```bash
# Download all model files from HuggingFace
cargo run --bin download-model

# Local synthesis (play or save, no server needed)
cargo run --bin synth -- "Hello world" --play
cargo run --bin synth -- "Hello world" -v am_adam -o out.mp3
cargo run --bin synth -- --list-voices
```

## Deployment (Fly.io)

```bash
fly deploy --config fly.toml --app tider-tts
```

## Development

```bash
cargo test          # unit tests (requires libespeak-ng)
cargo clippy -- -D warnings
cargo fmt --check
```
