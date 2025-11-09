import time
from app.session import Session
from utils.utils import dprint, lprint
import re 
import asyncio
from utils.constants import LLM_MODEL
if LLM_MODEL == "local":
    from llm.ollama import chat_reply, chat_greeting
else:
    from llm.openai import chat_reply, chat_greeting
import orjson as json

def jdumps(o): return json.dumps(o).decode()

SILENCE_PATTERN = re.compile(r"<\s*silence\s+(\d+(?:\.\d+)?)\s*>", re.IGNORECASE)

def split_by_silence_markers(text: str):
    """
    '<silence N>' 기준으로 텍스트/무음 명령을 순서대로 반환.
    반환 예) ["Hello.", ("__silence__", 3.0), "How are you?"]
    """
    parts = []
    pos = 0
    for m in SILENCE_PATTERN.finditer(text):
        if m.start() > pos:
            seg = text[pos:m.start()].strip()
            if seg:
                parts.append(seg + "<cont>")
        dur = float(m.group(1))
        parts.append(("__silence__", dur))
        pos = m.end()
    # 꼬리 텍스트
    tail_seg = text[pos:].strip()
    if tail_seg:
        parts.append(tail_seg)
    return parts

def reset_conversation(sess: Session):
    sess.transcripts.append(sess.current_transcript)
    sess.current_transcript = ""
    sess.outputs.append(sess.answer)

async def conversation_worker(sess: Session):
    while sess.running:
        text = await sess.answer_q.get()
        await answer_one(sess, text)

async def answer_one(sess: Session, transcript: str):
    st = time.time()
    sess.current_transcript += " " + transcript
    answer_text = await run_answer_async(sess)  # 내부에서 run_in_executor 사용
    dprint(f"[Answer {time.time() - st:.2f}s] - {answer_text!r}")
    sess.answer = answer_text.strip()

    sess.transcripts.append(sess.current_transcript)
    sess.current_transcript = ""
    sess.outputs.append(sess.answer)

async def answer_greeting(sess: Session):
    loop = asyncio.get_running_loop()

    def run_blocking():
        return chat_greeting(
            language=sess.language,
            name=sess.name,
            current_time=sess.current_time
        )

    output = await loop.run_in_executor(None, run_blocking)
    answer_text = (output.get("text", "") or "").strip()
    sess.transcripts.append('[Call is started. User says nothing yet]')
    sess.outputs.append(answer_text)

    if answer_text:
        loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, answer_text)
        loop.call_soon_threadsafe(
            sess.out_q.put_nowait,
            jdumps({
                "type": "speaking",
                "script": sess.current_transcript,
                "text": answer_text,
                "is_final": True
            })
        )

async def run_answer_async(sess: Session) -> str:
    loop = asyncio.get_running_loop()
    sentence = ''
    sent_chars = 0  # answer_text 중 이미 보낸 문자 수

    def safe_push_tts(obj):
        loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, obj)

    def safe_push_out(msg: dict):
        loop.call_soon_threadsafe(sess.out_q.put_nowait, jdumps(msg))

    def push_pieces(pieces):
        # pieces: ["text", ("__silence__", 3.0), ...]
        for p in pieces:
            safe_push_tts(p)

    def flush_buffer_if_has_silence():
        nonlocal sentence, sent_chars
        if SILENCE_PATTERN.search(sentence):
            pieces = split_by_silence_markers(sentence)
            push_pieces(pieces)
            sent_chars += len(sentence)
            sentence = ""

    def on_token(tok: str):
        nonlocal sentence, sent_chars
        # 토큰 누적
        sentence += tok
        safe_push_out({"type": "speaking", "text": tok, "is_final": False})
        # 버퍼 기준으로 <silence N> 완성 여부 확인 후 즉시 플러시
        flush_buffer_if_has_silence()
        return

    def run_blocking():
        return chat_reply(
            prev_scripts=sess.transcripts[-10:],
            prev_answers=sess.outputs[-10:],
            input_sentence=sess.current_transcript,
            language=sess.language,
            onToken=on_token,           # executor 스레드에서 호출될 가능성 높음
            name=sess.name,
            current_time=sess.current_time
        )

    # st = time.time()
    output = await loop.run_in_executor(None, run_blocking)
    answer_text = output.get("text", "") or ""
    # print(f"[{time.time() - st}s] - {answer_text}")
    
    tail = answer_text[sent_chars:]
    if tail:
        pieces = split_by_silence_markers(tail)
        push_pieces(pieces)
    
    await asyncio.sleep(0)
    cts = sess.current_transcript
    
    def _g():
        sess.out_q.put_nowait(jdumps({"type": "speaking", "script": cts, "text": answer_text, "is_final": True}))
    loop.call_soon_threadsafe(_g)
    
    return answer_text

