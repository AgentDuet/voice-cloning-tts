# Examples

Runnable clients for the TTS streaming server. Each is self-contained and documented at the top of its file.

| File | What it shows |
|------|---------------|
| `simple_client.py` | Bare-minimum client: register a voice from a `.wav`, then stream synthesized speech to an output file. |
| `phone_agent_tts.py` | A phone agent that speaks in a **cloned voice** from this repo's TTS server: Gemini is the ears+brain, its transcript is streamed sentence-by-sentence into `/tts/{voice_id}/{sr}`, and the cloned-voice audio is played back to the caller. |

## Dependencies

The examples need packages the server itself does not (`agentduet`, `google-genai`, `python-dotenv`). They live in the `examples` [dependency group](https://peps.python.org/pep-0735/) in `pyproject.toml`:

```bash
uv sync --group examples
```

The server image built by the `Dockerfile` uses `uv sync --no-dev` and does **not** pull this group, so it stays lean.

## Running

First, start the TTS server — see the main [README](../readme.md) (Docker), or for local dev run `uv run python server.py`. Then run an example **from the repository root** (so relative paths like `examples/sample.wav` resolve):

```bash
uv run python examples/simple_client.py
```

`simple_client.py` registers `examples/sample.wav`, prints the resulting `voice_id`, and writes the synthesized speech to `out.wav`.

### Phone agent (`phone_agent_tts.py`)

This needs a configured [agentduet](https://pypi.org/project/agentduet/) connector and API keys. Put them in `examples/.env` (loaded automatically — `load_dotenv()` searches from the script's own directory):

```dotenv
GEMINI_API_KEY=...
AGENTDUET_API_KEY=...
AGENTDUET_CONNECTOR_UUID=...

# phone_agent_tts.py only:
TTS_VOICE_ID=<a voice_id registered with this server>
TTS_SERVER_URL=ws://localhost:8000   # optional; this is the default
```

Register a voice first — run `simple_client.py` and copy the `voice_id` it prints (or send a sample to the `/tts/upload` WebSocket yourself) — and set it as `TTS_VOICE_ID`. Then call the phone number bound to your agentduet connector and the agent answers in the cloned voice.
