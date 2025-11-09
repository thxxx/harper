class Capture24kWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const ms = (options?.processorOptions?.frameMs ?? 85) | 0;
    const sr = 24000; // fixed 24kHz pipeline
    this.frameSize = Math.max(1, Math.floor((sr * ms) / 1000)); // e.g., 480 samples for 20ms @24k
    this.buf = new Float32Array(this.frameSize); // 85ms, 2040 frame size
    this.i = 0;
  }
  process(inputs) {
    const ch0 = inputs?.[0]?.[0];
    if (!ch0) return true;
    let o = 0;
    const N = ch0.length;
    while (o < N) {
      const remain = this.frameSize - this.i;
      const take = Math.min(remain, N - o);
      this.buf.set(ch0.subarray(o, o + take), this.i);
      this.i += take;
      o += take;
      if (this.i === this.frameSize) {
        // Send a copy to avoid mutation between posts
        this.port.postMessage({ audio: new Float32Array(this.buf) });
        this.i = 0;
      }
    }
    return true;
  }
}
registerProcessor("capture-24k", Capture24kWorklet);
