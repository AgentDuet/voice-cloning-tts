# TTS Streaming Server — Deployment Guide

A real-time text-to-speech server. You upload a short audio sample once to register a "voice," then stream synthesized speech for any text in that voice over a WebSocket — chunk by chunk, as it's generated, so playback can start before the full clip is rendered.

This guide takes you from nothing to a running server, step by step, using Docker.

---

## How it works

- FastAPI accepts WebSocket connections and hands synthesis to a pool of worker processes, streaming the resulting audio chunks back as they're produced.
- The voice model is [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts), downloaded from the Hugging Face Hub on first run.
- Each registered voice is a folder on disk (`origin.wav`, a precomputed `prompt_audio.pt`, and `meta.json`).

---

## Prerequisites

### 1. Hugging Face access (required)

The model `kyutai/pocket-tts` is a **gated** repository, so you must have a Hugging Face account, accept its terms, and use an access token. Without this, the server cannot download the weights and synthesis will fail (or produce garbled audio). Do this first:

1. Create a free account at <https://huggingface.co/join>.
2. Create an access token with **Read** access at <https://huggingface.co/settings/tokens>.
3. Open <https://huggingface.co/kyutai/pocket-tts> and click **"Agree and access repository"**. A valid token is still rejected for this model until you accept its terms here.
4. Keep the token handy — you'll pass it as `HF_TOKEN` when you run the server.

### 2. Docker

Install Docker: <https://docs.docker.com/get-docker/>. The container needs outbound internet access on first run to download the model weights.

---

## Deploy — step by step

### Step 1 — Build the image

From the repository root (a `Dockerfile` is included):

```bash
docker build -t tts-server:local .
```

### Step 2 — Run the container

Pass your Hugging Face token and mount two host folders so the model cache and registered voices survive container restarts:

```bash
docker run --rm -it \
  -p 8000:8000 \
  -e HF_TOKEN=<your-hugging-face-token> \
  -v "$(pwd)/voices:/app/voices" \
  -v "$(pwd)/hf-cache:/root/.cache/huggingface" \
  --name tts-server \
  tts-server:local
```

- `-e HF_TOKEN=…` — authenticates the gated-model download (Prerequisite 1).
- `-p 8000:8000` — exposes the API on `http://localhost:8000`.
- `-v …/voices:/app/voices` — keeps registered voices on the host so `docker rm` doesn't wipe them.
- `-v …/hf-cache:/root/.cache/huggingface` — caches the downloaded weights so they aren't re-fetched every time you recreate the container.
- Any `TTS_*` variable from [Configuration](#configuration) can be added with `-e VAR=value`.

The first run downloads the model (hundreds of MB) and may take a few minutes.

### Step 3 — Verify it's up

```bash
curl http://localhost:8000/health
# {"status":"ok","workers":N}   # N = TTS_NUM_WORKERS (defaults to your CPU count − 1)
```

### Step 4 — Register a voice and synthesize

Use the ready-made client in [`examples/`](examples/):

```bash
uv sync --group examples
uv run python examples/simple_client.py   # registers examples/sample.wav, writes out.wav
```

Play `out.wav` — that's your cloned voice. To stop the server: `docker stop tts-server`.

---

## Examples

Client code lives in [`examples/`](examples/) (see [`examples/README.md`](examples/README.md)):

| File | What it shows |
|------|---------------|
| `simple_client.py` | Register a voice, stream synthesized speech to a file. |
| `phone_agent_tts.py` | A phone agent (via the `agentduet` SDK + Gemini Live) that answers calls and speaks in a cloned voice from this server. |

Install their extra dependencies with `uv sync --group examples`.

---

## Configuration

Set any of these with `-e VAR=value` on `docker run`. Most deployments only touch `TTS_NUM_WORKERS`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_TOKEN` | — | Hugging Face token for the gated model download (see Prerequisites). |
| `TTS_PORT` | `8000` | Bind port. |
| `TTS_NUM_WORKERS` | CPUs − 1 (min 1) | Model worker processes. **Main throughput knob** — one worker serves one generation at a time. Each worker loads its own full copy of the model, so raising this increases both throughput and memory use. |
| `TTS_VOICE_DIR` | `voices` | Where voice data is stored. |
| `TTS_MAX_TEXT_LENGTH` | `2000` | Max characters per synthesis request. |
| `TTS_CORS_ORIGINS` | `*` | Comma-separated allowed browser origins — set to your front-end origin in production. |
| `TTS_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`, written to stdout. |

The full set of tuning knobs (sample-rate limits, upload/duration caps, cache sizes, queue timeouts) lives in the `Config` class in `server.py`.

---

## Wire protocol

Two WebSocket endpoints. All audio is raw **16-bit signed PCM, mono, little-endian**.

### Register a voice — `WS /tts/upload`

1. Connect, send the sample audio as one **binary** frame (any format `soundfile` reads — WAV, FLAC, OGG…).
2. Server replies with a **text** JSON frame: `{"event":"prompt_loaded","voice_id":"a1b2c3d4","sample_rate":24000}` (or `{"event":"error","message":"…"}`).

Keep samples short — a few seconds is enough; uploads over `TTS_MAX_AUDIO_DURATION_SECONDS` (30s) are rejected.

### Synthesize — `WS /tts/{voice_id}/{sr}`

- `voice_id`: from the upload step. `sr`: output sample rate (server resamples).

1. Connect. Unknown `voice_id` / out-of-range `sr` → `error` event, then close.
2. Send a **text** frame: `{"text":"Hello, world."}`.
3. Server streams back: `{"event":"generating"}` → binary PCM frames → `{"event":"done"}`.
4. Send another `{"text":"…"}` on the same connection to continue. One generation at a time per connection.

See `examples/simple_client.py` for a complete Python implementation.

---

## Operational endpoints

- `GET /health` — liveness, returns `{"status":"ok","workers":N}`.
- `GET /api/voices` — lists registered voices and their metadata.
- `GET /` — serves `index.html` if present next to `server.py` (optional demo UI).

---

## Security notes

- **No authentication is built in.** If reachable outside a trusted network, put it behind an API gateway / reverse proxy that handles auth.
- `voice_id` values are validated and resolved paths are checked against the voice storage root (closes path traversal).
- Upload size and text length are capped to bound per-request memory/CPU.
- Set `TTS_CORS_ORIGINS` to your real front-end origin(s) in production.
- The provided `Dockerfile` runs as `root`; for production add a non-root `USER`. GPU inference needs an `nvidia/cuda` base image, a matching CUDA build of `torch`/`torchaudio`, and `--gpus all`.

---

## Logging

Plain Python `logging` to stdout — no external shipper required. Every line carries a correlation id (`cid=…`) so you can trace one connection:

```
2026-08-04 10:15:22,001 | INFO | tts_stream | cid=3f1a9c... | finished, total audio=4.32s
```

Pipe stdout into whatever your infra uses; adjust verbosity with `TTS_LOG_LEVEL`.

---

## Troubleshooting

**Audio is noisy, garbled, or clearly wrong.** Almost always the model weights didn't download — usually a Hugging Face auth issue. Confirm the Prerequisites (token created *and* "Agree and access repository" clicked), then:

```bash
docker logs tts-server | grep -iE "huggingface|401|403|gated|token"
```

`401`, `Repository Not Found`, or a *gated repo* message confirms it. Check the cache actually holds real weight files (tens/hundreds of MB), not just tiny `.json`/`.lock` files:

```bash
ls -la hf-cache/hub
```

If an earlier run without a token left a broken cache entry, delete that model's subfolder under `hf-cache/hub` and restart the container to force a clean re-download.
