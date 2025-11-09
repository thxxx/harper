# Harper

**Harper** is a conversational AI process inspired by [Sesame](https://sesame.com).  
It implements a full voice-based conversation loop with speech recognition, text generation, and speech synthesis, all running locally or via APIs.
Shout out to Chris Hong! 😊

Please watch [demo video 🎬](https://www.loom.com/share/3ef0ffd2844a4f148e087a7e6bd69b9b)!

<br />

**Features Implemented**
<br />
1. When the user is silent, the system occasionally generates **short self-talk**.  
2. The LLM is **forced to begin with a preset “first word”**, whose audio is pre-generated to reduce TTFT.  
3. It inserts **short silences mid-sentence** for more natural pacing.  
4. **Interruptions** mid-speech are handled; only spoken content is stored in the conversation history.  
5. Using multilingual Chatterbox, it can **speak in any language and any cloned voice** (English works best).  
6. Audio is **encoded/decoded with Opus**.  
7. Smart **turn detection** using <code>silero-vad</code> and <code>pipecat/smart-turn-v3</code>.

<br />

**Notes**

- For **fully local serving**, you can run <code>gpt-oss-20b</code>.  
- For **better quality and latency**, <code>gpt-4.1-mini</code> via API is recommended.  
- Running for 30 minutes of continuous conversation with <code>gpt-4.1-mini</code> costs **less than $0.10**.  
- Other local LLMs may work, but no further experiments were done for optimization.
- On the first run, it will take some time to download the model weights.

<br />

**For voice cloning**

Currently, to reduce TTFT (Time to First Token), the system pre-generates audio for several dozen common “starting words.”
During conversation, the LLM is always prompted to begin with one of these words, and the corresponding pre-generated audio is played immediately to minimize initial delay.

If you want to change the voice, update the new voice audio file path in ```utils/constants.py``` and then run ```voiceprepare.py```.
This will regenerate the pre-generated audio clips using the new voice.


## Modules Used
thanks to the following projects 🔥

- **STT**: [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)  
- **LLM**: <code>gpt-4.1-mini</code> (API) or <code>gpt-oss-20b</code> (via [Ollama](https://ollama.ai))  
- **TTS**: [chatterbox](https://github.com/resemble-ai/chatterbox)
- **Turn Detection**: [silero-vad](https://github.com/snakers4/silero-vad) + [pipecat/smart-turn-v3](https://github.com/pipecat-ai/smart-turn)

<br />

## Server Requirements

- CUDA-compatible GPU  
- Tested with **CUDA 12.4** and **12.6** (other versions may also work)  
- **Python 3.10** recommended (newer versions may work fine)  
- <code>ffmpeg</code> required for some audio operations  

<br />

### Server Setup

```bash
git clone https://github.com/thxxx/harper.git  
cd server  
python3.10 -m venv .venv  
source .venv/bin/activate  
bash setup.sh
```

### Client Setup

```bash
git clone https://github.com/thxxx/harper.git  
cd client  
npm install  
npm run dev  
```


## 🏃 Quickstart

**Run the client**

```bash
npm run dev  
```

<br />

**Run the server**

**1. For Local LLM**

1. In <code>server/utils/constants.py</code>, set  
   <code>LLM_MODEL = "local"</code>  
2. Then start Ollama:

```bash
ollama server  
ollama run gpt-oss:20b
uvicorn companionserver:app --host 0.0.0.0 --port 5000  
```

<br />

**2. For GPT API**

Set your API key in <code>server/utils/constants.py</code>  
   or export it as an environment variable:  
   ```export OPENAI_KEY="sk-xxxxxx"```

```bash

uvicorn companionserver:app --host 0.0.0.0 --port 5000  
```

<br />

---

## 🚧 Not Yet Implemented

1. Context management / long-term memory  
2. Lower TTFT & best-performing local LLM setup
3. CPU support

---

## ⭐️ Ending

Questions or feedback?  
Feel free to reach out: khj605123@gmail.com
<br />
Best regards, From Korea
