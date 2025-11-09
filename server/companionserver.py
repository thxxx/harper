import os
import socket
import multiprocessing as mp
import asyncio
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import orjson as json
import time
from websockets.asyncio.client import connect as ws_connect
import numpy as np
import librosa
import torch
import threading
import re

from stt.asr import load_asr_backend
from stt.vad import check_audio_state
from utils.process import process_data_to_audio
from app.session import Session
from app.session_control import teardown_session, outbound_sender
from utils.text_process import text_pr
from utils.process import get_volume, pcm16_b64
from utils.utils import dprint, lprint
from llm.conversation import conversation_worker, answer_greeting

from tts.chatter_infer import chatter_streamer, cancel_silence_nudge
from utils.opus import ensure_opus_decoder, decode_opus_float, parse_frame
from third.smart_turn.inference import predict_endpoint

INPUT_SAMPLE_RATE = 24000
WHISPER_SR = 16000

def jdumps(o):
    return json.dumps(o).decode()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI()

ASR = None
LLM = None

sessions: Dict[int, Session] = {}  # map by id(ws)


@app.on_event("startup")
def init_models():
    """Initialize global runtime switches and preload optional assets."""
    # Multiprocessing / env knobs must be set ASAP
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"

    # Python multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Already set by another import path
        pass

    # If you need to warm up any LLM backends, do it here (lazy by default)
    # global LLM


async def transcribe_pcm_generic(audios, sample_rate: int, channels: int, language: str) -> str:
    if not audios:
        return ""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: ASR.transcribe_pcm(audios, sample_rate, channels, language=language)
    )


async def stt_worker(sess: Session, in_q: asyncio.Queue, out_q: asyncio.Queue):
    try:
        while True:
            pcm_bytes = await in_q.get()  # wait independently of the main loop
            try:
                sttstart = time.time()
                text = await transcribe_pcm_generic(
                    audios=pcm_bytes,
                    sample_rate=WHISPER_SR,
                    channels=sess.input_channels,
                    language=sess.language,
                )
                await out_q.put({"type": "delta", "text": text})
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[stt_worker] error: {e}")
            finally:
                in_q.task_done()
    finally:
        # drain to avoid pending tasks on shutdown
        while not in_q.empty():
            try:
                in_q.get_nowait()
                in_q.task_done()
            except Exception:
                break


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session(input_sr=INPUT_SAMPLE_RATE, input_channels=1)
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

                if t == "ping":
                    await ws.send_text(
                        jdumps(
                            {
                                "type": "pong",
                                "t0": data.get("t0"),
                                "server_now": int(time.time() * 1000),
                            }
                        )
                    )
                    continue

                if t == "scriptsession.setvoice":
                    inputprompt = data.get("prompt")
                    print("[scriptsession.setvoice] : ", inputprompt)
                    if inputprompt:
                        sess.prompt = inputprompt

                if t == "scriptsession.clonevoice":
                    if data.get("voice") is not None:
                        voice = data.get("voice")
                        audio = process_data_to_audio(voice, input_sample_rate=24000, whisper_sr=WHISPER_SR)
                        sess.ref_audios.put(audio)

                # 1) Session start
                if t == "scriptsession.start":
                    global ASR
                    lprint("Start ", data)

                    # current time (ms) if provided by client
                    if data.get("time") is not None:
                        sess.current_time = data.get("time")

                    requested_lang = data.get("language", "en").strip()
                    if ASR is None: # currently only one ASR backend is supported
                        ASR = load_asr_backend()
                    sess.language = requested_lang

                    sess.name = data.get("name", "hojin")

                    if sess.stt_task is None:
                        sess.stt_task = asyncio.create_task(stt_worker(sess, sess.stt_in_q, sess.stt_out_q))

                    if sess.stt_out_consumer_task is None:
                        sess.stt_out_consumer_task = asyncio.create_task(stt_out_consumer(sess))

                    if sess.tts_task is None:
                        try:
                            sess.tts_task = asyncio.create_task(chatter_streamer(sess))
                            sess.conversation_task = asyncio.create_task(conversation_worker(sess))
                            sess.is_use_filler = data.get("use_filler", False)

                            await ws.send_text(jdumps({"type": "scriptsession.started"}))
                        except Exception as e:
                            dprint("TTS connection error ", e)
                    else:
                        await ws.send_text(jdumps({"type": "warn", "message": "already started"}))

                    await answer_greeting(sess)

                elif t == "input_audio_buffer.append":
                    try:
                        aud = data.get("audio")
                        if aud:
                            audio = process_data_to_audio(
                                aud, input_sample_rate=INPUT_SAMPLE_RATE, whisper_sr=WHISPER_SR
                            )
                            if audio is None:
                                dprint("[NO AUDIO]")
                                continue

                            vad_event = check_audio_state(audio)

                            if sess.current_audio_state != "start":
                                sess.pre_roll.append(audio)
                                if vad_event == "start":
                                    energy_ok = get_volume(
                                        np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False)
                                    )[1] > 0.02
                                    if not energy_ok:
                                        continue

                                    sess.current_audio_state = "start"
                                    cancel_silence_nudge(sess)
                                    await interrupt_output(sess, reason="start speaking")
                                    if len(sess.pre_roll) > 0:
                                        sess.audios = (np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False))
                                    else:
                                        sess.audios = audio.astype(np.float32, copy=False)

                                    print("[Voice Start]")
                                    sess.pre_roll.clear()
                                    sess.buf_count = 0
                                continue

                            sess.audios = np.concatenate([sess.audios, audio])
                            sess.buf_count += 1

                            if vad_event == "end" and sess.transcript != "": # immediately stop
                                # check by using smart-turn-detection-v3 from pipecat
                                print("[Voice End] - ", sess.transcript)
                                audio = sess.audios[:WHISPER_SR * 12]
                                turn_result = predict_endpoint(audio)
                                
                                print("Turn result : ", turn_result, "\n")
                                if turn_result['probability'] < 0.01:
                                    score = turn_result['probability']
                                    delay = 4.0 * (1 - score)
                                    # We can ignore when vad evenet is end but smart turn detection is not.
                                    # But what if it is wrong?
                                    # Set timer and trigger turn end after 1 second.
                                    if getattr(sess, "pending_turn_task", None):
                                        sess.pending_turn_task.cancel()
                                        sess.pending_turn_task = None
                            
                                    # 2) 새로 1초짜리 타이머 걸기
                                    sess.pending_turn_task = asyncio.create_task(
                                        delayed_force_turn_end(sess, delay=delay, script=sess.transcript.strip())
                                    )
                                    continue
                                
                                sess.current_audio_state = "none"
                                await sess.out_q.put(
                                    jdumps({"type": "transcript", "text": sess.transcript.strip(), "is_final": True})
                                )
                                
                                sess.audios = np.empty(0, dtype=np.float32)
                                sess.end_scripting_time = time.time() % 1000

                                sess.answer_q.put_nowait(sess.transcript)
                                sess.transcript = ""
                                continue

                            if sess.buf_count % 8 == 7 and sess.current_audio_state == "start":
                                sess.audios = sess.audios[-WHISPER_SR * 20 :]
                                pcm_bytes = (
                                    np.clip(sess.audios, -1.0, 1.0) * 32767.0
                                ).astype(np.int16).tobytes()

                                # Non-blocking queue push (drop oldest on overflow)
                                try:
                                    sess.stt_in_q.put_nowait(pcm_bytes)
                                except asyncio.QueueFull:
                                    try:
                                        _ = sess.stt_in_q.get_nowait()
                                        sess.stt_in_q.task_done()
                                    except asyncio.QueueEmpty:
                                        pass
                                    try:
                                        sess.stt_in_q.put_nowait(pcm_bytes)
                                    except asyncio.QueueFull:
                                        pass
                                sess.buf_count = 0

                    except Exception as e:
                        dprint("Error : ", e)

                elif t == "input_audio_buffer.commit":
                    lprint("input_audio_buffer.commit - ", sess.transcript)
                    if sess.transcript is not None and sess.transcript != "":
                        await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript, "is_final": True}))

                    if sess.transcript is not None and sess.transcript != "":
                        sess.answer_q.put_nowait(sess.transcript)
                        sess.transcript = ""

                    sess.current_audio_state = "none"
                    sess.audios = np.empty(0, dtype=np.float32)

                elif t == "session.close":
                    await ws.send_text(
                        jdumps(
                            {
                                "type": "session.close",
                                "payload": {"status": "closed successfully"},
                                "connected_time": time.time() - sess.connection_start_time,
                                "llm_cached_token_count": sess.llm_cached_token_count,
                                "llm_input_token_count": sess.llm_input_token_count,
                                "llm_output_token_count": sess.llm_output_token_count,
                            }
                        )
                    )
                    break

                else:
                    # other message types
                    pass

            elif msg.get("bytes") is not None:
                buf: bytes = msg["bytes"]

                frm = parse_frame(buf)
                if not frm:
                    await ws.send_text(jdumps({"type": "binary_ack", "payload": {"received_bytes": len(buf)}}))
                    continue
            
                if frm["is_config"]:
                    continue

                dec = ensure_opus_decoder(sess, sr=INPUT_SAMPLE_RATE, ch=1)
                try:
                    audio = decode_opus_float(frm["payload"], dec, sr=INPUT_SAMPLE_RATE)  # np.float32, mono
                except Exception as e:
                    dprint("[Opus decode error]", e)
                    continue

                if len(sess.bufs) == 0:
                    sess.bufs.append(audio)
                    continue
                else:
                    sess.bufs.append(audio)
                    audio = np.concatenate(sess.bufs, axis=0).astype(np.float32)
                    sess.bufs = []

                vad_event = check_audio_state(audio)

                if sess.current_audio_state != "start":
                    sess.pre_roll.append(audio)
                    if vad_event == "start":
                        energy_enough = get_volume(np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False))[1] > 0.02
                        if not energy_enough:
                            continue

                        sess.current_audio_state = "start"
                        cancel_silence_nudge(sess)
                        await interrupt_output(sess, reason="start speaking")

                        if len(sess.pre_roll) > 0:
                            sess.audios = np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False)
                        else:
                            sess.audios = audio.astype(np.float32, copy=False)
                        print("[Voice Start]")
                        sess.pre_roll.clear()
                        sess.buf_count = 0
                    continue

                sess.audios = np.concatenate([sess.audios, audio]).astype(np.float32, copy=False)
                sess.buf_count += 1

                if vad_event == "end" and sess.transcript != "":
                    print("[Voice End] - ", sess.transcript)
                    await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript.strip(), "is_final": True}))
                    sess.current_audio_state = "none"
                    sess.audios = np.empty(0, dtype=np.float32)
                    sess.end_scripting_time = time.time() % 1000
                    sess.answer_q.put_nowait(sess.transcript) # answer_q will be used as input for llm response
                    sess.transcript = ""
                    continue

                if sess.buf_count % 5 == 4 and sess.current_audio_state == "start":
                    sess.audios = sess.audios[-WHISPER_SR * 20:]
                    pcm_bytes = (np.clip(sess.audios, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    try:
                        sess.stt_in_q.put_nowait(pcm_bytes)
                    except asyncio.QueueFull:
                        _ = sess.stt_in_q.get_nowait()
                        sess.stt_in_q.task_done()
                        sess.stt_in_q.put_nowait(pcm_bytes)
                    sess.buf_count = 0
                
    except WebSocketDisconnect:
        pass
    finally:
        await teardown_session(sess)
        sessions.pop(id(ws), None)


async def stt_out_consumer(sess: Session):
    while sess.running:
        msg = await sess.stt_out_q.get()
        try:
            new_text = (msg or {}).get("text", "") or ""
            if sess.current_audio_state != "none":
                sess.transcript = text_pr(sess.transcript, new_text)
                await sess.out_q.put(jdumps({"type": "delta", "text": sess.transcript, "is_final": False}))
        finally:
            sess.stt_out_q.task_done()


def _trim_last_one_words(s: str) -> str:
    words = re.findall(r"\S+", s)
    if len(words) <= 1:
        return ""
    return " ".join(words[:-1])

async def _transcribe_tts_buffer(sess: Session) -> str:
    buf = getattr(sess, "tts_pcm_buffer", np.empty(0, dtype=np.float32))
    sr  = getattr(sess, "tts_buffer_sr", 24000)

    print("buf size : ", buf.size)
    if buf.size < 4800:
        return ""
    # float32 [-1,1] → int16 bytes
    pcm_i16 = np.clip(buf, -1.0, 1.0)
    pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16, copy=False).tobytes()
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            lambda: ASR.transcribe_pcm(pcm_i16, sr, 1, language=sess.language)
        )
        return text or ""
    except Exception as e:
        print("[interrupt_output] ASR error:", e)
        return ""
    finally:
        # 사용 후 즉시 비움
        sess.tts_pcm_buffer = np.empty(0, dtype=np.float32)


async def interrupt_output(sess: Session, reason: str = "start speaking"):
    print("[interrupt_output] ", reason)
    st = time.time()

    now = time.time()
    if now - getattr(sess, "last_interrupt_ts", 0) < 1.0:
        print("If under 1s")
        return
    sess.last_interrupt_ts = now

    try:
        sess.out_q.put_nowait(jdumps({"type": "tts_stop", "reason": reason}))
    except Exception:
        pass

    try:
        sess.tts_stop_event.set()
    except Exception:
        pass

    for task_name in ("tts_task", "conversation_task", "silence_nudge_task"):
        task = getattr(sess, task_name, None)
        if task and not task.done():
            task.cancel()

            try:
                await asyncio.wait_for(task, timeout=0.2)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            finally:
                setattr(sess, task_name, None)

    # Drain pending queues
    def drain_aio_queue(q: asyncio.Queue):
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except asyncio.QueueEmpty:
            pass

    for q in (sess.tts_in_q,):
        drain_aio_queue(q)

    sess.tts_stop_event = threading.Event()

    if sess.running:
        if sess.tts_task is None:
            sess.tts_task = asyncio.create_task(chatter_streamer(sess))
        if sess.conversation_task is None:
            sess.conversation_task = asyncio.create_task(conversation_worker(sess))

    try:
        partial_text = await _transcribe_tts_buffer(sess)
        print("[interrupt_output] partial_text: ", partial_text)
        if partial_text != "":
            # trimmed = _trim_last_one_words(partial_text.strip())
            trimmed = partial_text.strip()
            print("[interrupt_output] trimmed: ", trimmed)
            try:
                sess.outputs[-1] = trimmed
                sess.out_q.put_nowait(jdumps({"type": "interrupt_output", "text": trimmed}))
            except Exception:
                pass
    except Exception as e:
        print("Error ", e)
    
    print("[interrupt_output] took ", time.time() - st)

async def delayed_force_turn_end(sess: Session, delay: float = 1.0, script: str = ''):
    try:
        await asyncio.sleep(delay)
        print("delayed_force_turn_end : ", sess.current_audio_state, script, delay, "\n\n")
        if sess.current_audio_state == "start":
            return

        if sess.transcript.strip():
            await sess.out_q.put(
                jdumps({"type": "transcript", "text": script, "is_final": True})
            )
            sess.answer_q.put_nowait(script)
            sess.transcript = ""

        sess.current_audio_state = "none"
        sess.audios = np.empty(0, dtype=np.float32)
        sess.end_scripting_time = time.time() % 1000
    finally:
        sess.pending_turn_task = None
