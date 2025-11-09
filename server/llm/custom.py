from vllm import LLM, SamplingParams
import os
os.environ["TQDM_DISABLE"] = "1"

class CustomLLM():
    def __init__(self):
        self.llm = LLM(
            model="trillionlabs/Tri-1.8B-Translation",
            dtype="float16",               # GPU라면 권장
            tensor_parallel_size=1,        # 단일 GPU 보장
            enforce_eager=True,            # CUDA graph 캡쳐 이슈 회피
            gpu_memory_utilization=0.5,   # 메모리 여유 확보
            disable_log_stats=True
        )
        self.sp = SamplingParams(temperature=0.1, max_tokens=512)

    def translate(self, text, target="ko"):
        prompt = f"Translate into {target}:\n{text} <{target}>"
        out = self.llm.chat([{"role": "user", "content": prompt}], sampling_params=self.sp)
        return {
            "text": out[0].outputs[0].text.strip()
        }

def load_llm():
    return CustomLLM()