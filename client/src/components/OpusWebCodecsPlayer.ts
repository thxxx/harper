// utils/audio/OpusWebCodecsPlayer.ts
export class OpusWebCodecsPlayer {
  private audioCtx: AudioContext;
  private decoder: AudioDecoder | null = null;
  private playHead = 0;
  private primed = false;
  private readonly SR = 24000;
  private readonly CH = 1;
  private readonly SAFETY_BUFFER_SEC = 0.12;
  private readonly PREBUFFER_SEC = 0.25;
  private readonly FRAME_SEC = 0.02;
  private lastSeq = -1;
  private reorderWindow = new Map<number, Uint8Array>();
  private expecting = 0;
  private readonly REORDER_MAX = 8;

  constructor(audioCtx?: AudioContext) {
    this.audioCtx =
      audioCtx ??
      new (window.AudioContext || (window as any).webkitAudioContext)();
    if (!("AudioDecoder" in window)) throw new Error("WebCodecs not supported");
  }

  configure() {
    try {
      this.decoder?.close();
    } catch {}
    this.decoder = new (window as any).AudioDecoder({
      output: (audioData: AudioData) => this.onDecoded(audioData),
      error: (e: any) => console.error("[OpusDecoder] error", e),
    });
    const opusHeader = this.makeOpusHead(this.CH, 0, this.SR, 0, 0);
    this.decoder?.configure({
      codec: "opus",
      sampleRate: this.SR,
      numberOfChannels: this.CH,
      description: opusHeader.buffer,
    });
    if (this.audioCtx.state === "suspended") {
      this.audioCtx.resume().catch(() => {});
    }
    this.primed = false;
    this.playHead = Math.max(this.audioCtx.currentTime + this.PREBUFFER_SEC, 0);
    this.lastSeq = -1;
    this.reorderWindow.clear();
  }

  private onDecoded(audioData: AudioData) {
    const ab = this.audioCtx.createBuffer(
      audioData.numberOfChannels,
      audioData.numberOfFrames,
      audioData.sampleRate
    );
    for (let ch = 0; ch < audioData.numberOfChannels; ch++) {
      const tmp = new Float32Array(audioData.numberOfFrames);
      audioData.copyTo(tmp, { planeIndex: ch });
      ab.getChannelData(ch).set(tmp);
    }
    audioData.close();
    if (!this.primed) {
      this.playHead = Math.max(
        this.audioCtx.currentTime + this.PREBUFFER_SEC,
        this.playHead
      );
      this.primed = true;
    }
    const src = this.audioCtx.createBufferSource();
    src.buffer = ab;
    src.connect(this.audioCtx.destination);
    const when = Math.max(
      this.playHead,
      this.audioCtx.currentTime + this.SAFETY_BUFFER_SEC
    );
    src.start(when);
    this.playHead = when + ab.duration;
    src.onended = () => src.disconnect();
  }

  decodeFrame(payload: Uint8Array, seq: number, timestampUSec?: number) {
    if (!this.decoder) return;

    if (seq < this.expecting) return;

    if (seq === this.expecting) {
      this.doDecode(payload, seq, timestampUSec);
      this.expecting++;
      this.flushReorderWindowInOrder();
      return;
    }

    this.reorderWindow.set(seq, payload);

    if (this.reorderWindow.size > this.REORDER_MAX) {
      const keys = Array.from(this.reorderWindow.keys()).sort((a, b) => a - b);
      while (this.reorderWindow.size > this.REORDER_MAX) {
        this.reorderWindow.delete(keys.shift()!);
      }
    }
  }

  private flushReorderWindowInOrder() {
    while (this.reorderWindow.has(this.expecting)) {
      const p = this.reorderWindow.get(this.expecting)!;
      this.reorderWindow.delete(this.expecting);
      this.doDecode(p, this.expecting);
      this.expecting++;
    }
  }

  private doDecode(payload: Uint8Array, seq: number, timestampUSec?: number) {
    const ts = Math.trunc(timestampUSec ?? seq * this.FRAME_SEC * 1_000_000);
    const chunk = new (window as any).EncodedAudioChunk({
      type: "key",
      timestamp: ts,
      data: payload,
    });
    this.decoder!.decode(chunk);
  }

  async flush() {
    await this.decoder
      ?.flush()
      .catch((e) => console.error("[OpusDecoder] flush error", e));
    this.primed = false;
    this.playHead = Math.max(this.audioCtx.currentTime + this.PREBUFFER_SEC, 0);
    this.lastSeq = -1;
    this.reorderWindow.clear();
  }

  close() {
    try {
      this.decoder?.close();
    } catch {}
    this.decoder = null;
    this.primed = false;
    this.reorderWindow.clear();
  }

  get context() {
    return this.audioCtx;
  }

  private makeOpusHead(
    channels = 1,
    preSkip = 0,
    inputSampleRate = this.SR,
    gain = 0,
    channelMapping = 0
  ): Uint8Array {
    const magic = new TextEncoder().encode("OpusHead");
    const b = new Uint8Array(19);
    b.set(magic, 0);
    b[8] = 1;
    b[9] = channels;
    b[10] = preSkip & 0xff;
    b[11] = (preSkip >> 8) & 0xff;
    b[12] = inputSampleRate & 0xff;
    b[13] = (inputSampleRate >> 8) & 0xff;
    b[14] = (inputSampleRate >> 16) & 0xff;
    b[15] = (inputSampleRate >> 24) & 0xff;
    b[16] = gain & 0xff;
    b[17] = (gain >> 8) & 0xff;
    b[18] = channelMapping;
    return b;
  }
}
