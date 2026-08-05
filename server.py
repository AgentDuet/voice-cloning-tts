"""
TTS streaming server.

Architecture
------------
A small pool of worker processes each keep one instance of the TTS model
loaded (model load is expensive, so we never reload it per-request). The
FastAPI process itself never touches the model directly - it only talks to
workers through the multiprocessing pool and hands audio chunks back to the
browser over a WebSocket as soon as the worker produces them.

Two WebSocket endpoints are exposed:
  - /tts/upload            register a new voice from a short audio sample
  - /tts/{voice_id}/{sr}   stream synthesized speech for arbitrary text

See README.md for the wire protocol and a usage example.
"""

import os
import re
import sys
import json
import uuid
import shutil
import logging
import contextvars
import asyncio
import multiprocessing as mp
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pocket_tts import TTSModel


# ==================================================
# CONFIGURATION
# ==================================================
class Config:
    """All tunables live here and can be overridden via environment
    variables, so the same image can be deployed with different limits
    per environment without code changes."""

    SR = int(os.getenv("TTS_DEFAULT_SR", "24000"))
    NUM_WORKERS = int(os.getenv("TTS_NUM_WORKERS", max(os.cpu_count() - 1, 1)))
    VOICE_DIR = os.getenv("TTS_VOICE_DIR", "voices")

    HOST = os.getenv("TTS_HOST", "0.0.0.0")
    PORT = int(os.getenv("TTS_PORT", "8000"))

    # Abuse / resource protections
    MAX_UPLOAD_BYTES = int(os.getenv("TTS_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))   # 20 MB
    MAX_AUDIO_DURATION_SECONDS = float(os.getenv("TTS_MAX_AUDIO_DURATION_SECONDS", "30"))
    MAX_TEXT_LENGTH = int(os.getenv("TTS_MAX_TEXT_LENGTH", "2000"))
    MIN_SAMPLE_RATE = int(os.getenv("TTS_MIN_SAMPLE_RATE", "8000"))
    MAX_SAMPLE_RATE = int(os.getenv("TTS_MAX_SAMPLE_RATE", "48000"))

    # Per-worker in-memory cache of loaded voice states (LRU eviction)
    STATE_CACHE_SIZE = int(os.getenv("TTS_STATE_CACHE_SIZE", "8"))

    QUEUE_TIMEOUT_SECONDS = float(os.getenv("TTS_QUEUE_TIMEOUT", "30.0"))

    CORS_ORIGINS = [o.strip() for o in os.getenv("TTS_CORS_ORIGINS", "*").split(",")]

    # voice_id is always generated server-side as uuid4()[:8], but the
    # /tts/{voice_id}/{sr} route accepts it from the client, so it must be
    # validated before ever touching the filesystem.
    VOICE_ID_PATTERN = re.compile(r"^[a-f0-9]{8}$")

    WARMUP_TEXT = "warmup"


os.makedirs(Config.VOICE_DIR, exist_ok=True)

pool: Optional["mp.pool.Pool"] = None
manager: Optional["mp.managers.SyncManager"] = None
_index_html_cache: Optional[str] = None


# ==================================================
# Logging (stdlib only — no external/internal log service required)
# ==================================================
# Holds the current request's correlation id so every log line emitted
# while handling that request/connection can be tied together, without
# having to thread the id through every function call manually.
correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


class _CorrelationIdFilter(logging.Filter):
    """Injects the current correlation id into every log record so it
    shows up in the formatted output automatically."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_ctx.get()
        return True


def configure_logging() -> None:
    """Call once per process (main process and each worker process)."""
    level = os.getenv("TTS_LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-entered in a worker) — just make
        # sure the filter is attached and bail out.
        for h in root.handlers:
            if not any(isinstance(f, _CorrelationIdFilter) for f in h.filters):
                h.addFilter(_CorrelationIdFilter())
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_CorrelationIdFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | cid=%(correlation_id)s | %(message)s",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)


logger = logging.getLogger("tts_server")


class OpLog:
    """Small helper mirroring the shape of a "structured operation log"
    (name + key/value metrics + a message) but built entirely on top of
    the standard `logging` module — no external log shipper required.
    Each call logs immediately; there's no separate flush step."""

    def __init__(self, name: str, correlation_id: Optional[str] = None):
        self.name = name
        self.correlation_id = correlation_id or correlation_id_ctx.get()
        self.metrics: dict = {}
        self._logger = logging.getLogger(name)

    def put_metric(self, key: str, value) -> None:
        self.metrics[key] = value

    def _format(self, msg: str) -> str:
        if self.metrics:
            extra = " ".join(f"{k}={v}" for k, v in self.metrics.items())
            return f"{msg} | {extra}"
        return msg

    def debug(self, msg: str) -> None:
        self._logger.debug(self._format(msg))

    def info(self, msg: str) -> None:
        self._logger.info(self._format(msg))

    def warn(self, msg: str) -> None:
        self._logger.warning(self._format(msg))

    def error(self, exc: Exception) -> None:
        self._logger.error(self._format(str(exc)), exc_info=exc)


# ==================================================
# Small shared helpers
# ==================================================
def is_valid_voice_id(voice_id: str) -> bool:
    return bool(Config.VOICE_ID_PATTERN.match(voice_id))


def safe_voice_folder(voice_id: str) -> str:
    """Resolve the on-disk folder for a voice_id, refusing anything that
    would escape VOICE_DIR. Defense in depth on top of is_valid_voice_id."""
    base = os.path.realpath(Config.VOICE_DIR)
    folder = os.path.realpath(os.path.join(base, voice_id))
    if not (folder == base or folder.startswith(base + os.sep)):
        raise ValueError("invalid voice_id")
    return folder


def _load_state_dict(state_path: str):
    """torch.load with weights_only=True where available (torch >= 2.4).
    Falls back gracefully on older torch builds."""
    try:
        return torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


class _StreamResampler:
    """Resamples a stream of audio chunks continuously.

    torchaudio's resample is a sinc filter that needs context from
    neighboring samples to behave correctly. Calling it independently on
    each streamed chunk (as if each chunk were a standalone clip)
    introduces a small discontinuity at every chunk boundary, which is
    audible as clicking/warbling on the output stream.

    This keeps a short tail of raw (pre-resample) audio from the previous
    chunk, resamples [tail + chunk] together so the filter has proper
    context, then drops the portion of the output that corresponds to
    that borrowed tail — leaving only the newly resampled audio for the
    current chunk, with a clean, continuous result across chunk boundaries.
    """

    # ~20ms of context is comfortably more than torchaudio's default
    # low-pass filter kernel needs at typical speech sample rates.
    _OVERLAP_MS = 20

    def __init__(self, orig_freq: int, new_freq: int):
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self._overlap_samples = max(1, int(orig_freq * self._OVERLAP_MS / 1000))
        self._tail = np.zeros(0, dtype=np.float32)

    def push(self, wav: np.ndarray) -> np.ndarray:
        if self.orig_freq == self.new_freq or wav.size == 0:
            return wav

        combined = np.concatenate([self._tail, wav])
        drop_samples = int(round(len(self._tail) * self.new_freq / self.orig_freq))

        resampled = (
            torchaudio.functional.resample(
                torch.from_numpy(combined).unsqueeze(0),
                orig_freq=self.orig_freq,
                new_freq=self.new_freq,
            )
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )

        # Save raw (untouched) trailing samples as context for next call
        self._tail = (
            wav[-self._overlap_samples:] if len(wav) >= self._overlap_samples else wav.copy()
        )

        return resampled[drop_samples:]


# ==================================================
# Worker initialization
# ==================================================
def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Configure stdlib logging inside worker process (each worker is a separate process)
    configure_logging()

    global model
    model = TTSModel.load_model().eval()

    log = OpLog("init_worker")
    log.debug(f"Worker ready PID={os.getpid()}")


# ==================================================
# Worker: register a new voice (runs inside pool)
# ==================================================
def worker_register_voice(voice_id: str, audio_bytes: bytes) -> dict:
    """Runs inside the worker pool so the already-loaded model is reused.
    Saves origin.wav and prompt_audio.pt, returns sample_rate."""
    log_obj = OpLog("register_voice")
    log_obj.put_metric("voice_id", voice_id)
    log_obj.put_metric("pid", os.getpid())

    try:
        voice_folder = safe_voice_folder(voice_id)
        os.makedirs(voice_folder, exist_ok=True)

        wav_path = os.path.join(voice_folder, "origin.wav")
        state_path = os.path.join(voice_folder, "prompt_audio.pt")

        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        try:
            audio, sr = sf.read(wav_path)
        except Exception:
            raise Exception("uploaded file is not a valid audio file")

        duration_seconds = len(audio) / float(sr)
        if duration_seconds > Config.MAX_AUDIO_DURATION_SECONDS:
            raise Exception(
                f"audio is {duration_seconds:.1f}s long, which exceeds the "
                f"{Config.MAX_AUDIO_DURATION_SECONDS:.0f}s limit for a voice "
                f"sample — trim it and try again"
            )

        audio_tensor = torch.from_numpy(audio).float()

        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.mean(dim=1)

        audio_tensor = audio_tensor.unsqueeze(0)

        if sr != Config.SR:
            resampler = torchaudio.transforms.Resample(sr, Config.SR)
            audio_tensor = resampler(audio_tensor)

        voice_state = model.get_state_for_audio_prompt(audio_tensor)

        with torch.no_grad():
            for _ in model.generate_audio_stream(voice_state, Config.WARMUP_TEXT, copy_state=True):
                break

        torch.save(voice_state, state_path)

        if not os.path.exists(state_path):
            raise Exception("prompt_audio.pt missing after save")

        meta = {"voice_id": voice_id, "sample_rate": sr}
        with open(os.path.join(voice_folder, "meta.json"), "w") as f:
            json.dump(meta, f)

        log_obj.info(f"Voice registered voice_id={voice_id} sr={sr}")

        return {"voice_id": voice_id, "sample_rate": int(sr)}

    except Exception as e:
        log_obj.error(e)
        # Clean up any partial state (e.g. origin.wav saved but validation
        # failed afterward) so a later /tts/{voice_id}/... call can't pick
        # up a half-registered, oversized, or otherwise invalid voice.
        try:
            if "voice_folder" in locals() and os.path.isdir(voice_folder):
                shutil.rmtree(voice_folder, ignore_errors=True)
        except Exception:
            pass
        raise


# ==================================================
# Worker: TTS streaming
# ==================================================
def worker_tts_stream(voice_id: str, text: str, sr: int, queue) -> None:
    """Runs inside the worker pool. Streams int16 PCM chunks onto `queue`.
    A sentinel `None` is ALWAYS sent last, even on early errors, so the
    consumer side (stream_audio) never hangs. Errors are also sent as a
    structured message so the client can display something meaningful
    instead of silently getting an empty stream."""
    log_obj = OpLog("tts_stream")
    log_obj.put_metric("voice_id", voice_id)
    log_obj.put_metric("pid", os.getpid())

    try:
        log_obj.debug(f"Generating... PID={os.getpid()}")

        voice_folder = safe_voice_folder(voice_id)
        if not os.path.exists(voice_folder):
            raise Exception(f"voice folder not found: {voice_folder}")

        wav_path = os.path.join(voice_folder, "origin.wav")
        state_path = os.path.join(voice_folder, "prompt_audio.pt")

        model.eval()

        # ----------------------------------
        # Per-process LRU state cache
        # ----------------------------------
        if not hasattr(worker_tts_stream, "state_cache"):
            worker_tts_stream.state_cache = OrderedDict()

        cache: "OrderedDict" = worker_tts_stream.state_cache
        voice_state = None

        # 1. Try in-process cache
        if voice_id in cache:
            cache.move_to_end(voice_id)
            voice_state = cache[voice_id]
            log_obj.debug(f"cache hit PID={os.getpid()}")
        else:
            log_obj.debug(f"cache miss PID={os.getpid()}")

            # 2. Try loading prompt_audio.pt from disk
            if os.path.exists(state_path):
                try:
                    loaded = _load_state_dict(state_path)

                    if not isinstance(loaded, dict) or len(loaded) == 0:
                        raise Exception("invalid state dict")

                    warmup_ok = False
                    with torch.no_grad():
                        for _ in model.generate_audio_stream(loaded, Config.WARMUP_TEXT, copy_state=True):
                            warmup_ok = True
                            break

                    if not warmup_ok:
                        raise Exception("state warmup produced no output")

                    voice_state = loaded
                    log_obj.debug(f"prompt_audio.pt loaded + warmup ok PID={os.getpid()}")

                except Exception as e:
                    log_obj.debug(f"prompt_audio.pt invalid, will rebuild. reason={e}")
                    voice_state = None

            # 3. Rebuild from origin.wav
            if voice_state is None:
                if not os.path.exists(wav_path):
                    raise Exception("origin.wav missing and prompt_audio.pt invalid — cannot synthesise")

                log_obj.debug(f"rebuilding state from origin.wav PID={os.getpid()}")

                audio, wav_sr = sf.read(wav_path)

                duration_seconds = len(audio) / float(wav_sr)
                if duration_seconds > Config.MAX_AUDIO_DURATION_SECONDS:
                    raise Exception(
                        f"origin.wav for voice_id={voice_id} is "
                        f"{duration_seconds:.1f}s, which exceeds the "
                        f"{Config.MAX_AUDIO_DURATION_SECONDS:.0f}s limit — "
                        f"re-register this voice with a shorter sample"
                    )

                audio_tensor = torch.from_numpy(audio).float()

                if audio_tensor.ndim == 2:
                    audio_tensor = audio_tensor.mean(dim=1)

                audio_tensor = audio_tensor.unsqueeze(0)

                if wav_sr != Config.SR:
                    resampler = torchaudio.transforms.Resample(wav_sr, Config.SR)
                    audio_tensor = resampler(audio_tensor)

                voice_state = model.get_state_for_audio_prompt(audio_tensor)

                warmup_ok = False
                with torch.no_grad():
                    for _ in model.generate_audio_stream(voice_state, Config.WARMUP_TEXT, copy_state=True):
                        warmup_ok = True
                        break

                if not warmup_ok:
                    raise Exception("rebuilt state warmup produced no output")

                torch.save(voice_state, state_path)
                log_obj.debug(f"state rebuilt and saved PID={os.getpid()}")

            cache[voice_id] = voice_state
            cache.move_to_end(voice_id)
            if len(cache) > Config.STATE_CACHE_SIZE:
                evicted_id, _ = cache.popitem(last=False)
                log_obj.debug(f"evicted voice_id={evicted_id} from state cache PID={os.getpid()}")

        if voice_state is None:
            raise Exception("failed to load or rebuild voice state")

        # ----------------------------------
        # Stream generation
        # ----------------------------------
        log_obj.debug(f"start streaming text_len={len(text)} PID={os.getpid()}")

        total_samples = 0
        clipped_chunks = 0
        chunks = model.generate_audio_stream(voice_state, text, copy_state=True)
        resampler = _StreamResampler(Config.SR, sr) if sr != Config.SR else None

        with torch.no_grad():
            for chunk in chunks:
                wav = chunk.squeeze().cpu().numpy().astype(np.float32)

                # Diagnostic: the int16 conversion below assumes the model
                # emits floats in [-1, 1]. If it doesn't (e.g. degraded or
                # incorrectly-loaded model weights), values get hard-clipped
                # into harsh, distorted audio with no visible error — so
                # surface it in the logs instead of failing silently.
                if wav.size:
                    peak = float(np.max(np.abs(wav)))
                    if peak > 1.05:
                        clipped_chunks += 1

                if resampler is not None:
                    wav = resampler.push(wav)

                wav_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
                total_samples += len(wav_int16)
                queue.put(wav_int16.tobytes())

        if clipped_chunks:
            log_obj.warn(
                f"model output exceeded expected [-1, 1] range on "
                f"{clipped_chunks} chunk(s) — audio was hard-clipped; this "
                f"usually means the model weights are degraded/incorrectly "
                f"loaded (check Hugging Face auth) rather than a code issue"
            )

        if total_samples == 0:
            log_obj.warn("model produced zero audio chunks")
        else:
            log_obj.info(f"finished, total audio={total_samples / sr:.2f}s PID={os.getpid()}")

    except Exception as e:
        log_obj.error(e)
        # Structured error so the client gets a real message instead of a
        # silently-empty stream.
        queue.put({"event": "error", "message": str(e)})

    finally:
        queue.put(None)


# ==================================================
# FastAPI lifespan
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, manager, _index_html_cache

    # Configure stdlib logging in main process (must be after uvicorn initializes
    # its own logging, so ours doesn't get clobbered)
    configure_logging()

    manager = mp.Manager()
    pool = mp.Pool(
        processes=Config.NUM_WORKERS,
        initializer=init_worker,
    )

    # Read once at startup rather than on every GET /
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            _index_html_cache = f.read()

    logger.info(f"Worker pool ready ({Config.NUM_WORKERS} workers)")

    yield

    pool.close()
    pool.join()
    manager.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# Stream audio helper
# ==================================================
async def stream_audio(ws: WebSocket, queue, log_name: str, correlation_id: str) -> None:
    loop = asyncio.get_running_loop()

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    loop.run_in_executor(None, queue.get),
                    timeout=Config.QUEUE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                err_log = OpLog(log_name, correlation_id=correlation_id)
                err_log.warn("timeout waiting for audio chunk, assuming worker died")
                break

            if item is None:
                break

            if ws.client_state.name != "CONNECTED":
                break

            try:
                if isinstance(item, dict):
                    # Structured message from the worker (e.g. an error)
                    await ws.send_text(json.dumps(item))
                else:
                    await ws.send_bytes(item)
            except Exception:
                break

        if ws.client_state.name == "CONNECTED":
            await ws.send_text(json.dumps({"event": "done"}))

    finally:
        pass


# ==================================================
# List voices endpoint
# ==================================================
@app.get("/api/voices")
async def list_voices():
    voices = []
    if os.path.exists(Config.VOICE_DIR):
        for voice_id in os.listdir(Config.VOICE_DIR):
            voice_folder = os.path.join(Config.VOICE_DIR, voice_id)
            meta_path = os.path.join(voice_folder, "meta.json")
            if os.path.isdir(voice_folder) and os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    voices.append(meta)
    return {"voices": voices}


# ==================================================
# Upload voice endpoint
# ==================================================
@app.websocket("/tts/upload")
async def upload_voice(ws: WebSocket):
    await ws.accept()

    correlation_id = str(uuid.uuid4())
    log_name = "tts_upload"

    token = correlation_id_ctx.set(correlation_id)
    try:
        conn_log = OpLog(log_name, correlation_id=correlation_id)
        conn_log.debug("WebSocket connected")

        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message["type"] == "websocket.receive" and message.get("bytes") is not None:
                audio_bytes = message["bytes"]

                if not audio_bytes:
                    continue

                if len(audio_bytes) > Config.MAX_UPLOAD_BYTES:
                    await ws.send_text(json.dumps({
                        "event": "error",
                        "message": f"audio exceeds max allowed size of {Config.MAX_UPLOAD_BYTES} bytes",
                    }))
                    continue

                voice_id = str(uuid.uuid4())[:8]

                loop = asyncio.get_running_loop()
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda: pool.apply(worker_register_voice, (voice_id, audio_bytes)),
                    )
                except Exception as e:
                    await ws.send_text(json.dumps({"event": "error", "message": str(e)}))
                    continue

                upload_log = OpLog(log_name, correlation_id=correlation_id)
                upload_log.put_metric("voice_id", result["voice_id"])
                upload_log.put_metric("sample_rate", result["sample_rate"])
                upload_log.info(
                    f"prompt loaded voice_id={result['voice_id']} sample_rate={result['sample_rate']}"
                )

                await ws.send_text(
                    json.dumps(
                        {
                            "event": "prompt_loaded",
                            "voice_id": result["voice_id"],
                            "sample_rate": result["sample_rate"],
                        }
                    )
                )

    except WebSocketDisconnect:
        pass

    except Exception as e:
        err_log = OpLog(log_name, correlation_id=correlation_id)
        err_log.error(e)
        try:
            await ws.send_text(json.dumps({"event": "error", "message": str(e)}))
        except Exception:
            pass
        await ws.close()

    finally:
        correlation_id_ctx.reset(token)
        done_log = OpLog(log_name, correlation_id=correlation_id)
        done_log.debug("WebSocket session ended")


# ==================================================
# TTS endpoint
# ==================================================
@app.websocket("/tts/{voice_id}/{sr}")
async def tts_socket(ws: WebSocket, voice_id: str, sr: int):
    await ws.accept()

    correlation_id = str(uuid.uuid4())
    log_name = "tts_stream"

    # Serializes generation *per connection* so two texts sent back-to-back
    # can't stream onto the same socket concurrently and interleave bytes.
    generation_lock = asyncio.Lock()

    token = correlation_id_ctx.set(correlation_id)
    try:
        conn_log = OpLog(log_name, correlation_id=correlation_id)
        conn_log.put_metric("voice_id", voice_id)
        conn_log.put_metric("sr", sr)
        conn_log.debug(f"WebSocket connected voice_id={voice_id} sr={sr}")

        if not is_valid_voice_id(voice_id):
            await ws.send_text(json.dumps({"event": "error", "message": "invalid voice_id"}))
            await ws.close()
            return

        if not (Config.MIN_SAMPLE_RATE <= sr <= Config.MAX_SAMPLE_RATE):
            await ws.send_text(json.dumps({
                "event": "error",
                "message": f"sr must be between {Config.MIN_SAMPLE_RATE} and {Config.MAX_SAMPLE_RATE}",
            }))
            await ws.close()
            return

        voice_folder = safe_voice_folder(voice_id)

        if not os.path.exists(voice_folder):
            err_log = OpLog(log_name, correlation_id=correlation_id)
            err_log.error(Exception(f"voice not found: {voice_id}"))
            await ws.send_text(json.dumps({"event": "error", "message": "voice not found"}))
            await ws.close()
            return

        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message["type"] == "websocket.receive" and message.get("text"):
                data = json.loads(message["text"])
                text = data.get("text", "").strip()

                if not text:
                    continue

                if len(text) > Config.MAX_TEXT_LENGTH:
                    await ws.send_text(json.dumps({
                        "event": "error",
                        "message": f"text exceeds max length of {Config.MAX_TEXT_LENGTH} characters",
                    }))
                    continue

                if generation_lock.locked():
                    await ws.send_text(json.dumps({
                        "event": "error",
                        "message": "a generation is already in progress on this connection",
                    }))
                    continue

                async def _run_generation(text=text):
                    async with generation_lock:
                        queue = manager.Queue()
                        await ws.send_text(json.dumps({"event": "generating"}))

                        def _on_worker_error(exc):
                            err_log = OpLog(log_name, correlation_id=correlation_id)
                            err_log.error(exc)

                        pool.apply_async(
                            worker_tts_stream,
                            (voice_id, text, sr, queue),
                            error_callback=_on_worker_error,
                        )
                        await stream_audio(ws, queue, log_name, correlation_id)

                asyncio.create_task(_run_generation())

    except WebSocketDisconnect:
        pass

    except Exception as e:
        err_log = OpLog(log_name, correlation_id=correlation_id)
        err_log.error(e)

    finally:
        correlation_id_ctx.reset(token)
        done_log = OpLog(log_name, correlation_id=correlation_id)
        done_log.debug("WebSocket session ended")


# ==================================================
# UI route
# ==================================================
@app.get("/")
def index():
    if _index_html_cache is None:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(_index_html_cache)


# ==================================================
# Healthcheck endpoint
# ==================================================
@app.get("/health")
async def healthcheck():
    pool_alive = pool is not None
    return {"status": "ok" if pool_alive else "degraded", "workers": Config.NUM_WORKERS}


# ==================================================
# Run server
# ==================================================
if __name__ == "__main__":
    import uvicorn

    mp.set_start_method("spawn", force=True)

    uvicorn.run(
        "server:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
    )