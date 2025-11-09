from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
import time
from tqdm import tqdm
from IPython.display import Audio
import torchaudio
import re
from utils.constants import DEFAULT_VOICE_PATH, COMMON_STARTERS

def prepare_voice():
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    total_seconds = 0
    total_latency = 0
    for text in COMMON_STARTERS:
        st = time.time()

        wav = model.generate(
            text, 
            audio_prompt_path=DEFAULT_VOICE_PATH, 
            language_id='en',
            exaggeration=0.6,
            cfg_weight=0.7,
            temperature=0.7,
            repetition_penalty=1.3,
            min_p=0.02,
            top_p=0.9
        )
        total_seconds += wav.shape[-1]/24000
        total_latency += time.time() - st
        torchaudio.save(f"{re.sub('\.\.\.|!', '', text)}.wav", wav, sample_rate=24000)
        print(f"Generated {text} in {time.time() - st:.3f} seconds")

if __name__ == "__main__":
    prepare_voice()