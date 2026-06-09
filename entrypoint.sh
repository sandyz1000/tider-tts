#!/usr/bin/env bash


# flyctl volumes create models_data --size 2 --region lhr --app tider-tts

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILE="${MODEL_FILE:-model_quantized.onnx}"

# Download Kokoro-82M model assets if not already present.
# download-model writes to ./models relative to CWD, so run from /.
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "[entrypoint] Downloading Kokoro-82M model assets to $MODEL_DIR …"
    cd / && /usr/local/bin/download-model
    echo "[entrypoint] Model download complete."
else
    echo "[entrypoint] Model already present — skipping download."
fi

exec /usr/local/bin/tts-rs
