import asyncio
import json
from datetime import datetime
from typing import Dict, Optional
from llm.openai import translate
from stt.openai import open_openai_ws
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
import orjson as json  # loads/dumps 호환 아님
import os
import websockets
import base64
import re
import time
from websockets.asyncio.client import connect as ws_connect

print("=== server.py loaded ===")

def jdumps(o): return json.dumps(o).decode()

VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "wj5ree7FcgKDPFphpPWQ")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

app = FastAPI()

DEBUG = False
def dprint(*a, **k): 
    if DEBUG: print(*a, **k)

LOGG = False
def lprint(*a, **k): 
    if LOGG: print(*a, **k)

# 클라이언트 1명당 OpenAI Realtime WS를 하나씩 유지하기 위한 세션 상태
class Session:
    def __init__(self):
        self.oai_ws = None  # OpenAI WS 연결
        self.oai_task: Optional[asyncio.Task] = None  # OAI -> Client listener task
        self.running = True

        self.current_transcript: str = ''
        self.transcripts: list[str] = []
        self.current_translated: str = ''
        self.translateds: list[str] = []

        # 송신을 Queue로 관리
        self.out_q: asyncio.Queue[str] = asyncio.Queue()
        self.sender_task: Optional[asyncio.Task] = None

        # --- TTS용 필드, text buffer 단위 speech 생성 ---
        self.tts_ws = None
        self.tts_task: Optional[asyncio.Task] = None
        self.tts_in_q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)

        # onToken에서 단어/프레이즈 coalescing용
        self.tts_buf: list[str] = []
        self.tts_debounce_task: Optional[asyncio.Task] = None

        # variables for logging
        self.start_scripting_time = 0
        self.end_scripting_time = 0
        self.end_translation_time = 0
        self.first_translated_token_output_time = 0
        self.end_tts_time = 0
        self.end_audio_input_time = 0

        # time logging
        self.connection_start_time = 0
        self.llm_cached_token_count = 0
        self.llm_input_token_count = 0
        self.llm_output_token_count = 0
        self.tts_output_token_count = 0

        self.is_network_logging = False
        

sessions: Dict[int, Session] = {}  # id(ws)로 매핑

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session()
    sessions[id(ws)] = sess

    sess.sender_task = asyncio.create_task(outbound_sender(sess, ws))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(jdumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                t = data.get("type")

                # (A) 핑-퐁: 시계 오프셋 추정용
                if t == "latency.ping":
                    t1 = int(time.time() * 1000)   # server recv
                    t2 = int(time.time() * 1000)   # server send (즉시)
                    await ws.send_text(jdumps({
                        "type": "latency.pong",
                        "t0": data["t0"], "t1": t1, "t2": t2
                    }))
                    if not sess.is_network_logging:
                        sess.is_network_logging = True
                        lprint("network latency logging started")
                    continue

                # 1) 세션 시작: OpenAI Realtime WS 연결
                if t == "scriptsession.start":
                    if sess.oai_ws is None:
                        sess.oai_ws = await open_openai_ws()
                        sess.connection_start_time = time.time()
                        model_name = data.get("model") or 'gpt-4o-mini-transcribe'
                        
                        try:
                            initMsg = {
                                "type": 'transcription_session.update',
                                "session": {
                                    "input_audio_format": 'pcm16',
                                    "input_audio_transcription": {
                                        "model": model_name,
                                        "prompt": '',
                                        "language": (data.get("language") or "en")[:2],
                                    },
                                    # "turn_detection": None,
                                    "turn_detection": {
                                        "type": 'server_vad',
                                        "threshold": 0.4,
                                        "prefix_padding_ms": 200,
                                        "silence_duration_ms": 80,
                                    },
                                    "input_audio_noise_reduction": { "type": 'far_field' },
                                },
                            };
                            # 연결 직후 OpenAI 세션에 초기 메시지 전송 for setting up session
                            await sess.oai_ws.send(jdumps(initMsg))

                            # --- ElevenLabs TTS 스트리머 시작 ---
                            sess.tts_task = asyncio.create_task(
                                elevenlabs_streamer(
                                    sess,
                                    voice_id=VOICE_ID,
                                    api_key=ELEVENLABS_API_KEY,
                                    output_format="mp3_22050_32"
                                )
                            )
    
                            # OpenAI 이벤트를 client로 릴레이하는 백그라운드 태스크
                            sess.oai_task = asyncio.create_task(relay_openai_to_client(sess, ws))
                            await ws.send_text(jdumps({"type": "scriptsession.started"}))
                        except Exception as e:
                            dprint("TTS connection error ", e)
                    else:
                        await ws.send_text(jdumps({"type": "warn", "message": "already started"}))

                # 2) 오디오 append → Open AI로 그대로 전달
                elif t == "input_audio_buffer.append":
                    if not sess.oai_ws:
                        await ws.send_text(jdumps({"type": "error", "message": "session not started"}))
                        continue
                    
                    if sess.is_network_logging:
                        t1 = int(time.time() * 1000)   # server recv
                        t0 = data.get("t0")  # client send(ms)
                        await ws.send_text(jdumps({
                            "type": "audio.recv.ack",
                            "t0": t0,
                            "t1": t1
                        }))
                    
                    # 현재는 base64가 아닌 PCM이 온다.
                    if data.get("audio") and 'data' in data.get("audio"):
                        b64 = base64.b64encode(bytes(data.get("audio")['data'])).decode('ascii')
                        await sess.oai_ws.send(jdumps({
                            "type": "input_audio_buffer.append",
                            "audio": b64
                        }))

                # 3) 커밋 신호 전달 (chunk 경계)
                elif t == "input_audio_buffer.commit":
                    lprint("input_audio_buffer.commit")
                    sess.end_audio_input_time = time.time()
                    sess.end_tts_time = 0
                    if not sess.oai_ws:
                        await ws.send_text(jdumps({"type": "error", "message": "session not started"}))
                        continue
                    await sess.oai_ws.send(jdumps({"type": "input_audio_buffer.commit"}))

                elif t == "test":
                    ct = data.get("current_time")
                    if ct:
                        lprint("network latency : ", time.time()*1000 - ct)
                elif t == "session.close":
                    await ws.send_text(jdumps({
                        "type": "session.close",
                        "payload": {"status": "closed successfully"},
                        "connected_time": time.time() - sess.connection_start_time,
                        "llm_cached_token_count": sess.llm_cached_token_count,
                        "llm_input_token_count": sess.llm_input_token_count,
                        "llm_output_token_count": sess.llm_output_token_count,
                        "tts_output_token_count": sess.tts_output_token_count,
                    }))
                    break

                else:
                    # 필요시 기타 타입 처리
                    pass

            elif msg.get("bytes") is not None:
                # 바이너리로도 보낼 수 있다면 여기서 OAI로 전달하는 변형 가능
                buf: bytes = msg["bytes"]
                await ws.send_text(jdumps({
                    "type": "binary_ack",
                    "payload": {"received_bytes": len(buf)}
                }))

    except WebSocketDisconnect:
        pass
    finally:
        await teardown_session(sess)
        sessions.pop(id(ws), None)

# 3) relay_openai_to_client 수정본
async def relay_openai_to_client(sess: Session, client_ws: WebSocket):
    try:
        async for raw in sess.oai_ws:
            try:
                evt = json.loads(raw)
            except Exception:
                await sess.out_q.put(raw)
                continue

            etype = evt.get("type", "")

            if etype.endswith(".delta"):
                text = evt.get("delta") or evt.get("text") or evt.get("content") or ""
                await sess.out_q.put(jdumps({"type": "delta", "text": text})) # 거의 걸리지 않음.
                # pass
            elif etype.endswith(".completed"):
                # lprint("latency to transcribe end : ", time.time() - sess.end_audio_input_time)

                sess.first_translated_token_output_time = 0
                # 3-1) 최종 전사 수신
                final_text = (evt.get("transcript") or evt.get("content") or "").strip()
                sess.end_scripting_time = time.time()
                
                await sess.out_q.put(jdumps({
                    "type": "transcript", "text": final_text, "final": True
                }))

                # 3-2) 연결별 누적 전사 업데이트
                if sess.current_transcript:
                    sess.current_transcript += " " + final_text
                else:
                    sess.current_transcript = final_text
                
                if len(sess.current_transcript) < 3:
                    continue
                
                dprint("prevScripts", sess.transcripts[-5:])
                dprint("current_scripted_sentence", sess.current_transcript)
                dprint("current_translated", sess.current_translated)

                # --- 여기부터 교체 ---
                # 동기 translate를 스레드에서 비동기처럼 실행(스트리밍 콜백 유지)
                translated_text = await run_translate_async(sess)
                # --- 교체 끝 ---
                sess.end_translation_time = time.time()
                lprint("[Latency logging] script end to translate end time : ", sess.end_translation_time - sess.end_scripting_time)
                lprint("[Latency logging] first to last translate token time : ", sess.end_translation_time - sess.first_translated_token_output_time)

                translated_text = translated_text.replace("<SKIP>", "")
                if translated_text == "" or translated_text is None:
                    continue

                # 3-5) 누적 번역 업데이트 (공백 관리)
                if sess.current_translated:
                    sess.current_translated += " " + translated_text
                else:
                    sess.current_translated = translated_text

                # 3-6) 문장 종료(<END>) 처리
                if "<END>" in translated_text:
                    # 저장 시에는 <END> 제거해서 넣는 걸 권장
                    translated_text = translated_text.replace("<END>", "").strip()

                    sess.transcripts.append(sess.current_transcript)
                    sess.translateds.append(translated_text)

                    # 다음 문장 누적용 버퍼 비우기
                    sess.current_transcript = ""
                    sess.current_translated = ""

            elif etype == "error":
                await sess.out_q.put(jdumps({
                    "type": "error", "message": evt.get("error", evt)
                }))
            else:
                await sess.out_q.put(jdumps({
                    "type": "oai_event", "event": evt
                }))

    except ConnectionClosed:
        pass
    except Exception as e:
        await sess.out_q.put(jdumps({
            "type": "error", "message": f"OAI relay error: {e}"
        }))


async def teardown_session(sess: Session):
    sess.running = False

    tasks = [sess.tts_task, sess.oai_task, sess.sender_task]
    # 1) 모두 취소
    for t in tasks:
        if t and not t.done():
            t.cancel()
    for t in tasks:
        if t:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
    if sess.oai_ws:
        with contextlib.suppress(Exception):
            await sess.oai_ws.close()
    if sess.tts_ws:
        with contextlib.suppress(Exception):
            if sess.tts_ws.open:
                await sess.tts_ws.send(jdumps({"text": ""}))  # EOS
            await sess.tts_ws.wait_closed()
        sess.tts_ws = None

async def outbound_sender(sess: Session, client_ws: WebSocket):
    try:
        while sess.running:
            msg = await sess.out_q.get()
            await client_ws.send_text(msg)
    except Exception:
        pass

# --- 추가: 동기 translate를 스레드에서 돌리고, 콜백은 루프-세이프로 브리지 ---
async def run_translate_async(sess: Session) -> str:
    """
    sess.current_transcript, sess.current_translated, sess.transcripts[-5:]
    를 사용해서 동기 translate를 스레드에서 실행.
    onToken 콜백은 루프-세이프로 sess.out_q에 push.
    최종 완성 문자열을 반환.
    """
    loop = asyncio.get_running_loop()

    def flush_tts_chunk():
        if not sess.tts_buf or len(sess.tts_buf) == 0:
            return
        
        chunk = "".join(sess.tts_buf).strip()
        chunk = re.sub(r"<[^>]*>", "", chunk)

        dprint("[flush_tts_chunk] ", repr(chunk))
        sess.tts_buf.clear()
        try:
            # sess.tts_in_q.put_nowait(chunk) # 마지막 청크
            chunks = chunk.split(" ")
            for i in range(0, len(chunks), 3):
                if i >= len(chunks) - 4:
                    sess.tts_in_q.put_nowait(" ".join(chunks[i:])) # 마지막 청크
                    break
                else:
                    sess.tts_in_q.put_nowait(" ".join(chunks[i:i+3]) + " ") # 기본적으로는 단어를 3개씩 끊어서 보내기
        except asyncio.QueueFull:
            dprint("[flush_tts_chunk] WARN: tts_in_q full, dropping chunk")

    async def debounce_flush(delay_ms: int = 110):
        try:
            await asyncio.sleep(delay_ms / 1000)
            flush_tts_chunk()
        except Exception as e:
            dprint("[debounce_flush] WARN: debounce_flush cancelled", e)

    def on_token(tok: str):
        # 다른 스레드에서 불릴 수 있으므로 루프 스레드로 래핑
        if sess.first_translated_token_output_time == 0:
            sess.first_translated_token_output_time = time.time()

        def _append_and_schedule():
            sess.tts_buf.append(tok)
            if sess.tts_debounce_task and not sess.tts_debounce_task.done():
                sess.tts_debounce_task.cancel()
            sess.tts_debounce_task = asyncio.create_task(debounce_flush(110))
        loop.call_soon_threadsafe(_append_and_schedule)

    def run_blocking():
        # 동기 translate 호출
        return translate(
            prevScripts=sess.transcripts[-5:],
            current_scripted_sentence=sess.current_transcript,
            current_translated=sess.current_translated,
            onToken=on_token,
        )

    # 동기 작업을 thread로
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, run_blocking)
    dprint("llm output : ", output)
    final_text = output.get("text", "")
    if final_text == "":
        dprint("No translated text")
        return ''
    dprint("final_text : ", final_text)
    sess.llm_cached_token_count += output["prompt_tokens_cached"]
    sess.llm_input_token_count += output["prompt_tokens"]
    sess.llm_output_token_count += output["completion_tokens"]

    if sess.tts_debounce_task and not sess.tts_debounce_task.done():
        sess.tts_debounce_task.cancel()

    def _final_flush():
        if sess.tts_debounce_task and not sess.tts_debounce_task.done():
            sess.tts_debounce_task.cancel()
        flush_tts_chunk()

    loop.call_soon_threadsafe(_final_flush)

    # 최종 결과 알림
    await sess.out_q.put(jdumps({"type": "translated", "text": final_text}))
    return final_text

async def elevenlabs_streamer(
        sess: Session, 
        voice_id: str, 
        api_key: str,
        output_format: str = "mp3_22050_32",
        keepalive_interval: int = 18
    ):
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_flash_v2_5&output_format={output_format}&auto_mode=true"
    headers = [("xi-api-key", api_key)]
    try:
        async with websockets.connect(url, additional_headers=headers, max_size=None) as elws:
            sess.tts_ws = elws
            # settings 보낸 직후, 가벼운 텍스트 한 번(트리거는 False)
            try:
                await elws.send(jdumps({
                    "text": " ",  # 워밍업 (공백)
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "speed": 1.0},
                    "generation_config": { "chunk_length_schedule": [120,160,250,290] },
                    "xi_api_key": api_key  # 헤더 + 바디 둘 다 주면 가장 호환성 좋음
                }))
            except Exception as e:
                dprint("[elevenlabs_streamer] initial warmup send error:", e)

            async def recv_loop():
                dprint("[elevenlabs_streamer] recv_loop START")
                try:
                    async for msg in elws:
                        # ElevenLabs는 text 프레임(JSON)로 응답
                        data = json.loads(msg)
                        if sess.end_tts_time == 0:
                            lprint("Audio end to first audio output time : ", time.time() - sess.end_audio_input_time)
                        sess.end_tts_time = time.time()

                        # 오디오 청크
                        if "audio" in data and data['audio'] is not None:
                            dprint("[elevenlabs_streamer] Is Final? : ", len(data['audio']), data["isFinal"])
                            lprint("tts time : ", time.time() - sess.first_translated_token_output_time)
                            await sess.out_q.put(jdumps({
                                "type": "tts_audio",
                                "format": output_format,
                                "audio": data["audio"],
                                "isFinal": data.get("isFinal", False),
                            }))
                        else:
                            dprint("[elevenlabs_streamer] recv_loop msg", msg)
                        # 경고/오류
                        if "warning" in data or "error" in data:
                            await sess.out_q.put(jdumps({"type":"tts_info","payload":data}))
                except Exception as e:
                    dprint("[elevenlabs_streamer] recv_loop error:", e)
                    raise
                finally:
                    dprint("[elevenlabs_streamer] recv_loop END")

            async def send_loop():
                dprint("[elevenlabs_streamer] send_loop START")
                try:
                    while sess.running:
                        text_chunk = await sess.tts_in_q.get()
                        if not (text_chunk and text_chunk.strip()):
                            continue
                    
                        dprint("[elevenlabs_streamer] send_loop →", repr(text_chunk))
                        await elws.send(jdumps({
                            "text": text_chunk,
                            "try_trigger_generation": True
                        }))
                        # await elws.send(jdumps({"text": ""}))
                except asyncio.CancelledError:
                    dprint("[elevenlabs_streamer] send_loop CANCELLED")
                    raise
                except Exception as e:
                    dprint("[elevenlabs_streamer] send_loop error:", e)
                    raise
                finally:
                    dprint("[elevenlabs_streamer] send_loop END")

            async def keepalive_loop():
                dprint("[elevenlabs_streamer] keepalive_loop START")
                try:
                    while sess.running:
                        await asyncio.sleep(keepalive_interval)
                        # 공백 하나는 inactivity 연장에 안전
                        try:
                            await elws.send(jdumps({
                                "text": " ",
                                "try_trigger_generation": False
                            }))
                            dprint("[elevenlabs_streamer] keepalive sent")
                        except Exception as e:
                            dprint("[elevenlabs_streamer] keepalive error:", e)
                            raise
                except asyncio.CancelledError:
                    dprint("[elevenlabs_streamer] keepalive_loop CANCELLED")
                    raise
                finally:
                    dprint("[elevenlabs_streamer] keepalive_loop END")

            recv_task = asyncio.create_task(recv_loop())
            send_task = asyncio.create_task(send_loop())
            ka_task   = asyncio.create_task(keepalive_loop())

            # 셋 중 하나라도 예외로 끝나면 나머지도 정리
            done, pending = await asyncio.wait(
                {recv_task, send_task, ka_task},
                return_when=asyncio.FIRST_EXCEPTION
            )

            # 남은 태스크 취소
            for t in pending:
                t.cancel()
                with contextlib.suppress(Exception):
                    await t
            dprint(f"\n\nPrint for check End : {done} \n\n")
    except asyncio.CancelledError:
        dprint("[elevenlabs_streamer] Main task cancelled.")
    except Exception as e:
        dprint(f"[elevenlabs_streamer] An unexpected error occurred: {e}")
        await sess.out_q.put(jdumps({"type": "tts_error", "message": str(e)}))
    finally:
        # ✅ CRITICAL FIX: Send the End-of-Stream (EOS) message before closing.
        if sess.tts_ws and sess.tts_ws.open:
            dprint("[elevenlabs_streamer] Sending End-of-Stream message.")
            try:
                await sess.tts_ws.send(jdumps({"text": ""}))
            except Exception as e:
                dprint(f"[elevenlabs_streamer] Failed to send EOS message: {e}")
        
        sess.tts_ws = None
        dprint("[elevenlabs_streamer] Connection closed and cleaned up.")

# 필요 import
import contextlib
