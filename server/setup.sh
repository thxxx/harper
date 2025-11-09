apt-get update
apt-get install -y ffmpeg
apt-get install -y portaudio19-dev python3-dev

pip install --upgrade pip
pip install flask flask-cors --ignore-installed
pip install openai fastapi uvicorn[standard] orjson
pip install faster-whisper soundfile librosa
pip install --upgrade transformers accelerate
pip install pydub hf_transfer

pip install torchaudio numpy lhotse huggingface_hub safetensors tensorboard vocos opuslib
pip install cn2an inflect s3tokenizer diffusers conformer pykakasi resemble-perth
pip install pkuseg
pip install onnx onnxruntime seaborn pyaudio

# Tokenization
pip install jieba piper_phonemize pypinyin
pip install "setuptools<81"

pip install --upgrade "pyzmq<26"
pip install --upgrade --force-reinstall transformers
pip uninstall -y torchvision
python -m pip install --no-cache-dir "numpy<2.0"

wget https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.0.onnx

curl -fsSL https://ollama.com/install.sh | sh
# ollama serve
# ollama run gpt-oss:20b

# uvicorn companionserver:app --host 0.0.0.0 --port 5000