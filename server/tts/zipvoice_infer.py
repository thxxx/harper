# zipvoice_infer.py

import json
import torch
import torchaudio
from argparse import Namespace
from huggingface_hub import hf_hub_download
from vocos import Vocos
import safetensors.torch

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer, EspeakTokenizer, LibriTTSTokenizer, SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank


HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {"zipvoice": "zipvoice", "zipvoice_distill": "zipvoice_distill"}

# -------------------- 기본값 --------------------
DEFAULT_ARGS = {
    "model_name": "zipvoice_distill",
    "model_dir": None,
    "checkpoint_name": "model.pt",
    "vocoder_path": None,
    "tokenizer": "emilia",
    "lang": "en-us",
    "num_step": None,
    "guidance_scale": None,
    "feat_scale": 0.1,
    "speed": 1.2,
    "t_shift": 0.5,
    "target_rms": 0.1,
}

MODEL_DEFAULTS = {
    "zipvoice": {
        "num_step": 16,
        "guidance_scale": 1.0,
    },
    "zipvoice_distill": {
        "num_step": 8,
        "guidance_scale": 3.0,
    },
}

# -------------------- 유틸 --------------------
def get_vocoder(vocos_local_path: str | None = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def _load_files(params: Namespace):
    if params.model_dir is not None:
        model_ckpt = params.model_dir / params.checkpoint_name
        model_config = params.model_dir / "model.json"
        token_file = params.model_dir / "tokens.txt"
    else:
        model_ckpt = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.pt")
        model_config = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.json")
        token_file = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/tokens.txt")
    return model_ckpt, model_config, token_file

def _build_tokenizer(tokenizer_name: str, token_file: str, lang: str):
    if tokenizer_name == "emilia":
        return EmiliaTokenizer(token_file=token_file)
    elif tokenizer_name == "libritts":
        return LibriTTSTokenizer(token_file=token_file)
    elif tokenizer_name == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=lang)
    else:
        return SimpleTokenizer(token_file=token_file)

def _build_model(model_name: str, model_config: dict, tokenizer):
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    if model_name == "zipvoice":
        return ZipVoice(**model_config["model"], **tokenizer_config)
    else:
        return ZipVoiceDistill(**model_config["model"], **tokenizer_config)

def _load_state(model, ckpt_path: str):
    if ckpt_path.endswith(".safetensors"):
        safetensors.torch.load_model(model, ckpt_path)
    elif ckpt_path.endswith(".pt"):
        load_checkpoint(filename=ckpt_path, model=model, strict=True)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {ckpt_path}")

# -------------------- 초기화 --------------------
def load_infer_context(user_args: dict | None = None):
    """기본값 + 모델별 기본값 + 사용자 지정 인자 합쳐서 context 리턴"""
    args = DEFAULT_ARGS.copy()
    if user_args:
        args.update(user_args)

    # 모델별 디폴트 보정
    for k, v in MODEL_DEFAULTS[args["model_name"]].items():
        if args.get(k) is None:
            args[k] = v

    params = Namespace(**args)

    model_ckpt, model_config_path, token_file = _load_files(params)
    with open(model_config_path, "r") as f:
        model_cfg = json.load(f)

    tokenizer = _build_tokenizer(params.tokenizer, token_file, params.lang)
    model = _build_model(params.model_name, model_cfg, tokenizer)
    _load_state(model, model_ckpt)

    device = _get_device()
    model = model.to(device).eval()
    vocoder = get_vocoder(params.vocoder_path).to(device).eval()
    feature_extractor = VocosFbank()
    sampling_rate = model_cfg["feature"]["sampling_rate"]

    ctx = {
        "params": params,
        "model": model,
        "vocoder": vocoder,
        "tokenizer": tokenizer,
        "feature_extractor": feature_extractor,
        "device": device,
        "sampling_rate": sampling_rate,
    }
    return ctx

# -------------------- 메인 함수 --------------------
@torch.inference_mode()
def generate_sentence(prompt_text: str, prompt_wav_path: str, text: str, ctx: dict, speed=1.0):
    """외부에서 prompt_text, prompt_wav_path, text 만 넣으면 됨"""
    model = ctx["model"]
    vocoder = ctx["vocoder"]
    tokenizer = ctx["tokenizer"]
    feature_extractor = ctx["feature_extractor"]
    device = ctx["device"]
    params = ctx["params"]

    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text]) if prompt_text else None

    # prompt wav 처리
    if prompt_wav_path is not None:
        wav_tensor, wav_sr = torchaudio.load(prompt_wav_path)
        if wav_sr != ctx["sampling_rate"]:
            resampler = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=ctx["sampling_rate"])
            wav_tensor = resampler(wav_tensor)

        prompt_rms = torch.sqrt(torch.mean(torch.square(wav_tensor)))
        if prompt_rms < params.target_rms:
            wav_tensor = wav_tensor * params.target_rms / prompt_rms

        prompt_features = feature_extractor.extract(wav_tensor, sampling_rate=ctx["sampling_rate"]).to(device)
        prompt_features = prompt_features.unsqueeze(0) * params.feat_scale
        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    else:
        # 프롬프트 오디오가 없을 때 → 빈 텐서로 전달
        prompt_features = None
        prompt_features_lens = None
        prompt_rms = params.target_rms

    pred_features, *_ = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=params.speed,
        t_shift=params.t_shift,
        duration="predict",
        num_step=params.num_step,
        guidance_scale=params.guidance_scale,
    )

    pred_features = pred_features.permute(0, 2, 1) / params.feat_scale
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    if prompt_rms < params.target_rms:
        wav = wav * prompt_rms / params.target_rms

    return wav.cpu(), {"sampling_rate": ctx["sampling_rate"]}
