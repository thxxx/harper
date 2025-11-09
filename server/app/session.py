# app/ws/session.py
import threading
import asyncio, time, numpy as np
from typing import Optional, List
from collections import deque
from asyncio import Queue
import queue
import orjson as json

def jdumps(o): return json.dumps(o).decode()

class Session:
    answer_q: Queue[str]
    conversation_task: Optional[asyncio.Task]
    running: bool = True
    
    def __init__(self, input_sr: int, input_channels: int):
        self.answer_q = asyncio.Queue(maxsize=5)
        self.conversation_task = None
        self.prompt = ''
        self.name = 'hojin'
        self.user_memory = ''
        self.my_memory = ''
        self.current_time = ""
        self.bufs = []
        self.pending_turn_task = None

        self.tts_buffer_sr = 24000
        self.tts_pcm_buffer = np.empty(0, dtype=np.float32)
        
        self.running = True

        self.audio_buf = bytearray()
        self.audios = np.empty(0, dtype=np.float32)
        self.end_task = None

        self.input_sample_rate = input_sr
        self.input_channels = input_channels
        self.last_interrupt_ts: float = 0.0
        self.tts_stop_event = threading.Event()

        self.current_transcript: str = ''
        self.transcripts: List[str] = []
        self.answer: str = ''
        self.outputs: List[str] = []

        self.out_q: asyncio.Queue[str] = asyncio.Queue()
        self.sender_task: Optional[asyncio.Task] = None

        self.language = "ko"
        self.tts_task: Optional[asyncio.Task] = None
        self.stt_task: Optional[asyncio.Task] = None
        self.silence_nudge_task: Optional[asyncio.Task] = None
        self.stt_out_consumer_task: Optional[asyncio.Task] = None
        self.tts_in_q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)

        # ---- 전역/세션 초기화 시 ----
        self.stt_in_q: asyncio.Queue[bytes]  = asyncio.Queue(maxsize=2)
        self.stt_out_q: asyncio.Queue[dict]  = asyncio.Queue(maxsize=32)  # {"type": "delta"/"final", "text": ...}

        self.tts_buf: list[str] = []
        self.tts_debounce_task: Optional[asyncio.Task] = None
        self.buf_count = 0
        self.buf_length = 0

        # 로깅/통계
        self.start_scripting_time = 0
        self.end_scripting_time = 0
        self.end_translation_time = 0
        self.first_translated_token_output_time = 0
        self.end_tts_time = 0
        self.end_audio_input_time = 0

        self.connection_start_time = 0
        self.llm_cached_token_count = 0
        self.llm_input_token_count = 0
        self.llm_output_token_count = 0

        self.filler_audios = []

        self.is_network_logging = False
        self.current_audio_state = "none"
        self.new_speech_start = 0

        self.transcript = ""
        self.end_count = -1
        self.count_after_last_translation = 0
        self.pre_roll = deque(maxlen=3)

        self.is_use_filler = False

        self.ref_audios = queue.Queue()

def cancel_end_timer(sess):
    if sess.end_task and not sess.end_task.done():
        sess.end_task.cancel()
    sess.end_task = None

def arm_end_timer(sess, delay=3):
    cancel_end_timer(sess)
    async def _timer():
        await asyncio.sleep(delay)
        await sess.out_q.put(jdumps({"type": "translated", "script": None, "text": "<END>", "is_final": True}))
    
    sess.end_task = asyncio.create_task(_timer())
