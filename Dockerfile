# syntax=docker/dockerfile:1
# Build context: backends/tts-rs  (fly deploy --config backends/tts-rs/fly.toml)
#
# Both stages use Ubuntu 24.04 (glibc 2.39). ORT's prebuilt download-binaries are
# compiled against glibc 2.38+ (__isoc23_strtol), so Debian bookworm (glibc 2.36)
# fails to link and fails to run the binary.

# ── Builder ───────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    pkg-config libssl-dev libespeak-ng-dev build-essential clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.88 via rustup (no official rust:slim image for Ubuntu 24.04)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .

# Ubuntu 24.04 ships clang-18; LIBCLANG_PATH points bindgen at libclang.so
RUN LIBCLANG_PATH=/usr/lib/llvm-18/lib \
    cargo build --release --bin tts-rs --bin download-model

# ── Runner ────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libespeak-ng1 libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/tts-rs /usr/local/bin/tts-rs
COPY --from=builder /app/target/release/download-model /usr/local/bin/download-model
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODEL_DIR=/models
ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
