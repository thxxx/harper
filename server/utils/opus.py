# top-level import
import numpy as np
from opuslib import Decoder as OpusDecoder

MAGIC_A = 0xA1
MAGIC_B = 0x51

def parse_frame(buf: bytes):
    """[A1 51][flags1][reserved1][seq u32][ts_lo u32][ts_hi u32][len u32][payload...]"""
    if len(buf) < 20:
        return None
    if buf[0] != MAGIC_A or buf[1] != MAGIC_B:
        return None
    flags = buf[2]
    is_config = (flags & 1) != 0
    # reserved = buf[3]
    seq = int.from_bytes(buf[4:8], "little")
    ts_lo = int.from_bytes(buf[8:12], "little")
    ts_hi = int.from_bytes(buf[12:16], "little")
    plen = int.from_bytes(buf[16:20], "little")
    if 20 + plen > len(buf):
        return None
    payload = buf[20:20+plen]
    ts_usec = (ts_hi << 32) | ts_lo
    return {
        "is_config": is_config,
        "seq": seq,
        "ts_usec": ts_usec,
        "payload": payload,
    }

def ensure_opus_decoder(sess, sr=24000, ch=1):
    if getattr(sess, "opus_decoder", None) is None:
        sess.opus_decoder = OpusDecoder(sr, ch)
    return sess.opus_decoder

def decode_opus_float(payload: bytes, decoder: OpusDecoder, sr=24000) -> np.ndarray:
    frame_size = 1440 # 60ms, 24kHz
    pcm = decoder.decode(payload, frame_size, decode_fec=False)
    pcm_i16 = np.frombuffer(pcm, dtype="<i2")  # little-endian int16
    pcm_f32 = (pcm_i16.astype(np.float32) / 32767.0)
    return pcm_f32
