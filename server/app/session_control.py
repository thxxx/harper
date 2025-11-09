import contextlib
from app.session import Session
import asyncio
from fastapi import WebSocket
import json

def jdumps(o): return json.dumps(o).decode()

async def teardown_session(sess: Session):
    sess.running = False

    tasks = [sess.tts_task, sess.sender_task, sess.conversation_task, sess.silence_nudge_task]
    # 1) 모두 취소
    for t in tasks:
        if t and not t.done():
            t.cancel()
    for t in tasks:
        if t:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

async def outbound_sender(sess: Session, client_ws: WebSocket):
    try:
        while sess.running:
            msg = await sess.out_q.get()
            if isinstance(msg, (bytes, bytearray)):
                await client_ws.send_text(msg)
            else:
                await client_ws.send_text(msg)
            # await client_ws.send_text(msg)
    except Exception:
        pass