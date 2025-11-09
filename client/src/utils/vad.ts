// ---------------------------
// VAD: very simple RMS-based start/stop detection
// ---------------------------
interface VADState {
  speaking: boolean;
  silenceMs: number;
  voiceMs: number;
}

export function createVAD(
  thresholdStart = 0.0025,
  thresholdStop = 0.0015,
  minSilenceMs = 250
) {
  const state: VADState = { speaking: false, silenceMs: 0, voiceMs: 0 };
  let lastT = performance.now();
  return (rms: number) => {
    const now = performance.now();
    const dt = now - lastT;
    lastT = now;

    if (rms >= thresholdStart) {
      state.voiceMs += dt;
      state.silenceMs = 0;
      if (!state.speaking && state.voiceMs > 30) state.speaking = true;
    } else if (rms <= thresholdStop) {
      state.silenceMs += dt;
      state.voiceMs = 0;
      if (state.speaking && state.silenceMs >= minSilenceMs) {
        state.speaking = false;
        state.silenceMs = 0;
        return true; // voice end event
      }
    }
    return false;
  };
}
