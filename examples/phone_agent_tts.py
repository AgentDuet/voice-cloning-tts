"""
Phone agent that speaks in a CLONED voice, powered by this repo's TTS server.

This example wires three pieces together:

  1. agentduet    — phone I/O (caller audio in, agent audio out)
  2. Gemini Live  — the agent's "ears + brain" (speech-to-text + reasoning)
  3. this TTS     — the agent's "mouth": streaming voice-cloning TTS server
                    (server.py, exposed at ws://localhost:8000)

Gemini's generated audio is DISCARDED — we keep only Gemini's output
*transcript*, stream it sentence-by-sentence into this repo's TTS server, and
play the resulting cloned-voice audio back to the caller. That is what actually
demonstrates the TTS capability over the phone.

Why the transcript and not text modality?
    Gemini Live's text-only response modality is no longer supported, so we
    request the AUDIO modality (required), throw away the audio bytes, and use
    `output_audio_transcription` as the authoritative text to synthesize.

Flow:
    caller audio ─▶ Gemini Live ─▶ transcript deltas ─▶ SentenceBuffer
                                                            │ (per sentence)
                                                            ▼
                                                        TtsSpeaker
                                              ws://…/tts/{voice_id}/24000
                                                            │ PCM chunks
                                                            ▼
                                                     call.send_audio() ─▶ caller

Barge-in: when the caller talks over the agent, Gemini emits an `interrupted`
signal. We clear the pending sentence queue, drop the remaining audio of the
sentence currently being synthesized (while still draining the TTS socket so it
stays reusable — no reconnect), and flush the phone-side buffer via
`call._interrupt()`, so the agent falls silent immediately.

Setup:
    1. Start the TTS server:      uv run python server.py
    2. Register a voice once (e.g. via simple_client.py's register_voice or
       the /tts/upload endpoint) and note the returned voice_id.
    3. In .env set:
           GEMINI_API_KEY=...
           AGENTDUET_API_KEY=...
           AGENTDUET_CONNECTOR_UUID=...
           TTS_VOICE_ID=<the voice_id from step 2>
           TTS_SERVER_URL=ws://localhost:8000      # optional, this is the default
    4. Run:                       uv run python phone_agent_tts_example.py
    5. Call the number bound to your agentduet connector.

Requirements:
    - dotenv, websockets
    - google-genai>=1.57.0  (GEMINI_API_KEY)
    - agentduet
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

import websockets
from websockets import ConnectionClosed
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.genai.live import AsyncSession

from agentduet import (
    SessionManager,
    SessionManagerConfig,
    Call,
    CallAudioConfig,
    IncomingCallNotification,
    BufferFullError,
    CallClosedError,
    new_session_id,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("agentduet").setLevel(logging.INFO)

# The call runs at 24 kHz; we synthesize TTS at the same rate so the PCM we feed
# back to agentduet needs no resampling.
SAMPLE_RATE = 24000

genai_client = genai.Client(vertexai=False, api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "models/gemini-3.1-flash-live-preview"

# AUDIO modality is required, but we discard Gemini's audio and consume the
# transcript instead. The specific Gemini voice is therefore irrelevant.
CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO],
    output_audio_transcription=types.AudioTranscriptionConfig(),
    system_instruction=(
        "You are a helpful, friendly assistant answering a phone call. "
        "Keep your replies short and conversational, as if speaking aloud."
    ),
)


class SentenceBuffer:
    """Accumulates streamed transcript deltas and emits complete sentences.

    Pure and I/O-free. `add()` returns any sentences completed by the new
    delta; `flush()` returns the trailing partial (used at turn boundaries).
    """

    # Match a run of text up to and including a sentence terminator.
    _BOUNDARY = re.compile(r".*?[.!?\n]+", re.S)

    def __init__(self) -> None:
        self._buf = ""

    def add(self, text: str) -> list[str]:
        self._buf += text
        sentences: list[str] = []
        while True:
            match = self._BOUNDARY.match(self._buf)
            if not match:
                break
            sentence = match.group().strip()
            self._buf = self._buf[match.end():]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> Optional[str]:
        trailing = self._buf.strip()
        self._buf = ""
        return trailing or None


class TtsSpeaker:
    """Streams sentences to this repo's TTS server and plays the audio to the call.

    Owns a single persistent WebSocket. Sentences are queued and synthesized one
    at a time; the resulting int16 PCM chunks are forwarded to `call.send_audio()`.
    """

    def __init__(self, call: Call, voice_id: str, server_url: str) -> None:
        self._call = call
        self._uri = f"{server_url.rstrip('/')}/tts/{voice_id}/{SAMPLE_RATE}"
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._drop = False  # when set, discard the in-flight sentence's audio
        self._consumer_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "TtsSpeaker":
        try:
            self._ws = await websockets.connect(self._uri)
            logger.info("Connected to TTS server at %s", self._uri)
        except Exception:
            # Keep the call alive even if TTS is unreachable: the caller can
            # still be heard by Gemini; the agent simply won't speak.
            logger.exception("Could not connect to TTS server at %s", self._uri)
            self._ws = None
        self._consumer_task = asyncio.create_task(self._consume())
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    def enqueue(self, sentence: str) -> None:
        self._queue.put_nowait(sentence)

    def interrupt(self) -> None:
        """Barge-in: drop queued sentences and the audio of the current one."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._drop = True

    async def _consume(self) -> None:
        try:
            while True:
                sentence = await self._queue.get()
                try:
                    await self._speak(sentence)
                except ConnectionClosed:
                    logger.warning("TTS socket closed; agent will stay silent")
                    self._ws = None
                except Exception:
                    logger.exception("Error synthesizing sentence")
        except asyncio.CancelledError:
            raise

    async def _speak(self, sentence: str) -> None:
        if self._ws is None:
            return  # TTS unavailable; drop silently
        # A fresh sentence always plays; only barge-in mid-sentence sets _drop.
        self._drop = False
        await self._ws.send(json.dumps({"text": sentence}))
        async for message in self._ws:
            if isinstance(message, bytes):
                if self._drop:
                    continue  # discard interrupted audio, but keep draining
                try:
                    await self._call.send_audio(message)
                except BufferFullError:
                    logger.warning("Call buffer full; dropping TTS chunk")
            else:
                event = json.loads(message)
                kind = event.get("event")
                if kind == "done":
                    break
                if kind == "error":
                    logger.error("TTS server error: %s", event.get("message"))
                    break
                # "generating" and other status events are ignored

    async def close(self) -> None:
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None


class PhoneAgent:
    """Bridges an agentduet Call and Gemini Live through the cloned-voice TTS."""

    def __init__(
        self,
        call: Call,
        gemini_session: AsyncSession,
        voice_id: str,
        tts_server_url: str,
    ) -> None:
        self._call = call
        self._gemini_session = gemini_session
        self._voice_id = voice_id
        self._tts_server_url = tts_server_url
        self._sentences = SentenceBuffer()
        self._speaker: Optional[TtsSpeaker] = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None

    async def _on_hangup(self, _evt) -> None:
        logger.info("Received call.terminated event")
        try:
            await self._gemini_session.close()
        except Exception:
            logger.exception("Error closing Gemini session")

        for task in (self._send_task, self._recv_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, genai_errors.APIError):
                    pass

        if self._speaker:
            await self._speaker.close()

    async def run(self) -> None:
        self._call.on_hangup(self._on_hangup)
        async with TtsSpeaker(self._call, self._voice_id, self._tts_server_url) as speaker:
            self._speaker = speaker
            self._send_task = asyncio.create_task(self._stream_to_gemini())
            self._recv_task = asyncio.create_task(self._receive_from_gemini())
            await asyncio.gather(
                self._send_task, self._recv_task, return_exceptions=True
            )

    async def _stream_to_gemini(self) -> None:
        """Forward caller audio into Gemini Live."""
        try:
            async for audio_chunk in self._call.caller.audio_stream():
                await self._gemini_session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}"
                    )
                )
        except (ConnectionClosed, CallClosedError):
            pass
        except Exception:
            logger.exception("Error streaming audio to Gemini")
            raise

    async def _receive_from_gemini(self) -> None:
        """Consume Gemini's transcript and speak it via the cloned voice."""
        try:
            while True:
                async for response in self._gemini_session.receive():
                    server_content = response.server_content
                    if not server_content:
                        continue

                    if server_content.interrupted:
                        logger.debug("Gemini interrupted -> barge-in")
                        if self._speaker:
                            self._speaker.interrupt()
                        await self._call._interrupt()
                        self._sentences.flush()  # discard the interrupted turn's text
                        continue

                    transcription = server_content.output_transcription
                    if transcription and transcription.text:
                        for sentence in self._sentences.add(transcription.text):
                            if self._speaker:
                                self._speaker.enqueue(sentence)

                    # server_content.model_turn holds Gemini's own audio — discarded.

                    if server_content.turn_complete:
                        trailing = self._sentences.flush()
                        if trailing and self._speaker:
                            self._speaker.enqueue(trailing)
        except (ConnectionClosed, genai_errors.APIError):
            logger.debug("Gemini session closed")
        except asyncio.CancelledError:
            raise
        except CallClosedError:
            logger.debug("Call closed; stopping receive from Gemini")
        except Exception:
            logger.exception("Error receiving from Gemini")
            raise


async def main() -> None:
    voice_id = os.getenv("TTS_VOICE_ID")
    if not voice_id:
        raise SystemExit(
            "TTS_VOICE_ID is required. Register a voice via /tts/upload first, "
            "then set TTS_VOICE_ID in your environment."
        )
    tts_server_url = os.getenv("TTS_SERVER_URL", "ws://localhost:8000")

    config = SessionManagerConfig.create(
        api_key=os.getenv("AGENTDUET_API_KEY"),
        connector_uuid=os.getenv("AGENTDUET_CONNECTOR_UUID"),
        call_audio=CallAudioConfig(
            sample_rate=SAMPLE_RATE,
            buffer_size=1024 * 1024,  # 1 MB
        ),
    )

    async with SessionManager(config) as sm:
        logger.info("SessionManager started %s", sm.id)

        @sm.on_incoming_call
        async def on_call(noti: IncomingCallNotification):
            session = await sm.open_session(new_session_id(), noti.subscriber)
            call = await session.process_call(noti)
            logger.info("New incoming call=%s, caller=%s", call.id, call.caller)
            try:
                async with genai_client.aio.live.connect(
                    model=MODEL, config=CONFIG
                ) as gemini_session:
                    result = await call.answer()
                    if not result:
                        logger.error("Failed to answer call %s", call.id)
                        return
                    agent = PhoneAgent(call, gemini_session, voice_id, tts_server_url)
                    await agent.run()
            except Exception:
                logger.exception("Error in phone agent")
                await call.close()
                raise

        await sm.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
