import base64
import torch
import torchaudio
import numpy as np

def process_data_to_audio(aud, input_sample_rate: int, whisper_sr: int) -> np.ndarray:
    """
    process input audio data to audio
    """
    if isinstance(aud, dict) and "data" in aud:
        pcm_bytes = bytes(aud["data"])  # int 배열
    elif isinstance(aud, str):
        pcm_bytes = base64.b64decode(aud)
    else:
        pcm_bytes = None
    
    if pcm_bytes:
        x = np.frombuffer(pcm_bytes, dtype=np.int16)
        # audio = x.astype(np.float32)
        audio = x.astype(np.float32) / 32768.0
        
        if input_sample_rate != whisper_sr:
            audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, n)
            audio_resampled = torchaudio.functional.resample(audio_t, orig_freq=input_sample_rate, new_freq=whisper_sr)
            audio = audio_resampled.squeeze(0).contiguous().numpy()
        return audio
    else:
        return None

def get_volume(audio: np.ndarray):
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    dbfs = 20.0 * np.log10(rms)
    peak = float(np.max(np.abs(audio)))
    return (dbfs, peak)

def pcm16_b64(part: torch.Tensor) -> str:
    arr = (torch.clamp(part, -1.0, 1.0).to(torch.float32) * 32767.0) \
            .to(torch.int16).cpu().numpy().tobytes()
    return base64.b64encode(arr).decode()