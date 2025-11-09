from dataclasses import dataclass
from pathlib import Path
import os
from warnings import catch_warnings

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download
import os
import numpy as np
import time
import asyncio

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()
        self._cond_cache = {}  # key -> {"ve": tensor, "t3_prompt": tensor|None, "s3ref": dict}

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "mtl_tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main", 
                allow_patterns=["ve.pt", "t3_23lang.safetensors", "s3gen.pt", "mtl_tokenizer.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    

    def _cache_key(self, wav_fpath):
        if isinstance(wav_fpath, str):
            st = os.stat(wav_fpath)
            return ("path", wav_fpath, st.st_size, int(st.st_mtime))
        else:
            import numpy as np, hashlib
            x = np.asarray(wav_fpath, dtype=np.float32)
            # 가벼운 해시 키 (길이/평균/표준편차 + 해시 앞부분)
            h = hashlib.sha1(x[: min(len(x), 48000)].tobytes()).hexdigest()[:8]
            return ("arr", len(x), float(x.mean()), float(x.std()), h)
            
    def prepare_conditionals1(self, wav_fpath, exaggeration=0.5):
        if isinstance(wav_fpath, str):
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        else:
            ref_16k_wav = wav_fpath
            s3gen_ref_wav = librosa.resample(ref_16k_wav, orig_sr=16000, target_sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    @torch.inference_mode()
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        key = self._cache_key(wav_fpath)
        cached = self._cond_cache.get(key)

        if cached is None:
            if isinstance(wav_fpath, str):
                s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR, mono=True, dtype=np.float32)
                ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR, res_type="soxr_qq")
            else:
                ref_16k_wav = np.asarray(wav_fpath, dtype=np.float32)
                s3gen_ref_wav = librosa.resample(ref_16k_wav, orig_sr=16000, target_sr=S3GEN_SR, res_type="soxr_qq")

            s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

            # Speech cond prompt tokens
            t3_cond_prompt_tokens = None
            if (plen := self.t3.hp.speech_cond_prompt_len):
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)).to(self.device)
            ve_embed = ve_embed.mean(dim=0, keepdim=True)

            cached = {"ve": ve_embed, "t3_prompt": t3_cond_prompt_tokens, "s3ref": s3gen_ref_dict}
            self._cond_cache[key] = cached

        # 여기서는 emotion만 갱신 (재계산 없음)
        t3_cond = T3Cond(
            speaker_emb=cached["ve"],
            cond_prompt_speech_tokens=cached["t3_prompt"],
            emotion_adv=torch.full((1,1,1), float(exaggeration), device=self.device),
        ).to(device=self.device)

        self.conds = Conditionals(t3_cond, cached["s3ref"])
    
    async def generate_stream(
        self,
        text,
        language_id,
        repetition_penalty=2.4,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        chunk_size=20,
    ):
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(f"Unsupported language_id '{language_id}'. Supported: {supported_langs}")
    
        if audio_prompt_path is not None:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"
    
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower()).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        sot, eot = self.t3.hp.start_text_token, self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    
        alpha, min_new_tokens, max_cap = 4.0, 100, 600
        text_len = text_tokens.shape[-1]
        max_new_tokens = min(max_cap, max(min_new_tokens, int(alpha * text_len)))
    
        audio_tokens = []
        queue = asyncio.Queue()

        lm_times = 0
        async def run_decoder(speech_tokens, ref_dict):
            wav, _ = await asyncio.to_thread(
                self.s3gen.inference,
                speech_tokens=speech_tokens,
                ref_dict=ref_dict
            )
            print("Decode! ")
            await queue.put({"type": "chunk", "audio": wav})
    
        # async def producer():
        #     st = time.time()
        #     for buf in self.t3.inference_streaming(
        #         t3_cond=self.conds.t3,
        #         text_tokens=text_tokens,
        #         max_new_tokens=max_new_tokens,
        #         temperature=temperature,
        #         cfg_weight=cfg_weight,
        #         repetition_penalty=repetition_penalty,
        #         min_p=min_p,
        #         top_p=top_p,
        #     ):
        #         if buf["type"] == "token":
        #             audio_tokens.append(buf["token_id"])
    
        #         if (buf["type"] == "token" and len(audio_tokens) % chunk_size == (chunk_size - 1)) \
        #         or (buf["type"] == "eos" and len(audio_tokens) > 0):
        #             speech_tokens = drop_invalid_tokens(torch.cat(audio_tokens, -1)[0]).to(self.device)
        #             asyncio.create_task(run_decoder(speech_tokens, self.conds.gen))
        #             await asyncio.sleep(0)
    
        #         if buf["type"] == "eos":
        #             print(f"[LM {time.time() - st}]")
        #             break
    
            # await queue.put({"type": "eos"})
        async def producer():
            st = time.time()
        
            # 동기 제너레이터를 가져오고
            it = self.t3.inference_streaming(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            SENTINEL = object()
        
            while True:
                # next(it) 한 번만 스레드에서 실행 → 이벤트 루프에 양보가 생김
                buf = await asyncio.to_thread(lambda: next(it, SENTINEL))
                if buf is SENTINEL:
                    break
        
                if buf["type"] == "token":
                    audio_tokens.append(buf["token_id"])
        
                if (buf["type"] == "token" and len(audio_tokens) % chunk_size == (chunk_size - 1)) \
                or (buf["type"] == "eos" and len(audio_tokens) > 0):
                    # >>> 전체 재디코딩 유지 <<<
                    speech_tokens = drop_invalid_tokens(torch.cat(audio_tokens, -1)[0]).to(self.device)
                    asyncio.create_task(run_decoder(speech_tokens, self.conds.gen))
                
                if buf["type"] == "eos":
                    print(f"[LM {time.time() - st}]")
                    # eos는 producer가 넣어주는 게 흐름이 깔끔
                    await queue.put({"type": "eos"})
                    return
        
                # 아주 짧게라도 양보 (안정성)
                await asyncio.sleep(0)
    
        asyncio.create_task(producer())

        ct = 0
        while True:
            evt = await queue.get()
            if evt['type'] == 'chunk':
                ct += 1
                if ct>2:
                    await queue.put({"type": "eos"})
                yield evt
            if evt["type"] == "eos":
                yield evt
                break
            yield evt