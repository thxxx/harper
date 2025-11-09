import { base64ToUint8Array } from "@/components/VoiceConditionModal";

export type STTTarget = "mic" | "speaker";
export const TARGET_SR = 24000;

export enum LANGUAGE_CODE {
  Arabic = "ar",
  German = "de",
  Greek = "el",
  Danish = "da",
  English = "en",
  Spanish = "es",
  Finnish = "fi",
  Hindi = "hi",
  Hebrew = "he",
  French = "fr",
  Italian = "it",
  Japanese = "ja",
  Korean = "ko",
  Russian = "ru",
  Chinese = "zh",
  Dutch = "nl",
  Norwegian = "no",
  Polish = "pl",
  Portuguese = "pt",
  Swedish = "sv",
  Swahili = "sw",
  Turkish = "tr",
}
export const llmId = "gpt-4.1-mini"; // translator

export const LLM_COST = {
  "gpt-4.1-mini": { cached: 0.1, input: 0.4, output: 1.6 },
  "gpt-4.1-nano": { cached: 0.025, input: 0.1, output: 0.4 },
  "gpt-5-nano": { cached: 0.005, input: 0.05, output: 0.4 },
  "gpt-4o-mini": { cached: 0.075, input: 0.15, output: 0.6 },
};

export type LLMKey = keyof typeof LLM_COST;

// Match your server namespace
export const SERVER_NS = "qi98kgjx2co45a"; // ← change if needed
export const USE_LAT_LOG = false; // send latency ping every 1s

// Target sample rate your server expects (based on your Electron app)

// ---------------------------
// StreamingAudioPlayer (queue-based, gap-minimized)
// ---------------------------
export class StreamingAudioPlayer {
  private audio: HTMLAudioElement;
  private q: { url: string }[] = [];
  private playing = false;
  private _volume = 1;

  constructor() {
    this.audio = new Audio();
    this.audio.addEventListener("ended", this._playNext);
    this.audio.addEventListener("error", this._playNext);
    this.audio.volume = this._volume;
  }

  dispose() {
    this.audio.pause();
    this.audio.src = "";
    this.q.forEach((x) => URL.revokeObjectURL(x.url));
    this.q = [];
  }

  set volume(v: number) {
    this._volume = Math.max(0, Math.min(1, v));
    this.audio.volume = this._volume;
  }

  private _playNext = () => {
    const next = this.q.shift();
    if (!next) {
      this.playing = false;
      return;
    }
    this.audio.src = next.url;
    this.audio.play().catch(() => {
      /* ignore */
    });
  };

  async pushBase64(
    b64: string,
    mime: "audio/mpeg" | "audio/wav" = "audio/mpeg"
  ) {
    const data = base64ToUint8Array(b64);
    const blob = new Blob([data], { type: mime });
    const url = URL.createObjectURL(blob);
    this.q.push({ url });
    if (!this.playing) {
      this.playing = true;
      this._playNext();
    }
  }
}
