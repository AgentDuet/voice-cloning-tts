import asyncio
import json
import websockets

async def register_voice(path_to_wav: str) -> str:
    async with websockets.connect("ws://localhost:8000/tts/upload") as ws:
        with open(path_to_wav, "rb") as f:
            await ws.send(f.read())
        reply = json.loads(await ws.recv())
        if reply["event"] == "error":
            raise RuntimeError(reply["message"])
        return reply["voice_id"]

async def synthesize(voice_id: str, text: str, out_path: str, sr: int = 24000):
    uri = f"ws://localhost:8000/tts/{voice_id}/{sr}"
    pcm_chunks = []
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"text": text}))
        async for message in ws:
            if isinstance(message, bytes):
                pcm_chunks.append(message)
            else:
                event = json.loads(message)
                if event["event"] == "done":
                    break
                if event["event"] == "error":
                    raise RuntimeError(event["message"])

    import numpy as np
    import soundfile as sf
    audio = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)
    sf.write(out_path, audio, sr)

async def main():
    voice_id = await register_voice("examples/sample.wav")
    print(f"Registered voice_id: {voice_id}")
    await synthesize(voice_id, "Hello, this is a streaming test.", "out.wav")

asyncio.run(main())