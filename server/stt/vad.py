import torch

SAMPLE_RATE_FOR_VAD = 16000
INPUT_FRAME_MS = 32
SAMPLES_PER_FRAME = SAMPLE_RATE_FOR_VAD * INPUT_FRAME_MS // 1000

silero, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
(get_speech_timestamps, _, read_audio, VADIterator, collect_chunks) = utils
vad_iter = VADIterator(silero, threshold=0.4, sampling_rate=SAMPLE_RATE_FOR_VAD, min_silence_duration_ms=250)

def check_audio_state(x):
    t = torch.from_numpy(x)[:512]
    
    label = vad_iter(t, return_seconds=False)
    # {'end': 212448} | {'start': 212448} | None

    if label is not None:
        if "start" in label:
            return 'start'
        if "end" in label:
            return 'end'
    return None
