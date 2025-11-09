import os
from websockets.asyncio.client import connect as ws_connect  # pip install websockets

OPENAI_KEY = os.environ.get("OPENAI_KEY")

async def open_openai_ws():
    """
    Not used anymore.
    Use HFWhisperBackend instead.
    """
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_KEY not set")

    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    return await ws_connect(url, additional_headers=headers)