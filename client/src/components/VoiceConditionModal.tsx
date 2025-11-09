"use client";

import React, { useEffect, useRef, useState } from "react";
import { X, Mic, Square, Play, Pause, Star } from "lucide-react";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";

export type VoiceSample = {
  blob: Blob;
  mimeType: string;
};

type Props = {
  onClose: () => void;
  onUploaded?: (res: { path: string; publicUrl?: string }) => void;
  initialSample?: VoiceSample | null;
  onSampleChange?: (s: VoiceSample | null) => void;
};

export default function VoiceConditionModal({
  onClose,
  onUploaded,
  initialSample,
  onSampleChange,
}: Props) {
  const {
    status,
    durationMs,
    blob,
    mimeType,
    startRecording,
    stopRecording,
    reset,
    setExternalSample,
  } = useAudioRecorder();

  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const injectingRef = useRef(false);
  const currentSigRef = useRef<string | null>(null);
  const makeSig = (b?: Blob | null, mt?: string | null) =>
    b ? `${b.size}:${mt ?? ""}` : "";

  useEffect(() => {
    if (!initialSample) return;
    const nextSig = makeSig(initialSample.blob, initialSample.mimeType);
    if (currentSigRef.current === nextSig) return;

    injectingRef.current = true;
    setExternalSample(initialSample.blob, initialSample.mimeType);
    currentSigRef.current = nextSig;
    queueMicrotask(() => {
      injectingRef.current = false;
    });
  }, [initialSample, setExternalSample]);

  useEffect(() => {
    if (!blob) {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }
      currentSigRef.current = null;
      return;
    }
    const url = URL.createObjectURL(blob);
    setAudioUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [blob]);

  useEffect(() => {
    if (!blob || !mimeType) return;
    if (injectingRef.current) return;

    const sig = makeSig(blob, mimeType);
    if (currentSigRef.current !== sig) {
      currentSigRef.current = sig;
    }
    onSampleChange?.({ blob, mimeType });
  }, [blob, mimeType]);

  const canUse = !!blob && status !== "recording" && !uploading;

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (audioRef.current.paused) {
      audioRef.current.play();
      setIsPlaying(true);
    } else {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };
  const onAudioEnded = () => setIsPlaying(false);

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${r.toString().padStart(2, "0")}`;
  };

  const handleReset = () => {
    reset();
    currentSigRef.current = null;
    onSampleChange?.(null);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 grid place-items-center"
    >
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
        aria-hidden
      />
      <div className="relative mx-4 w-full max-w-lg rounded-2xl border border-neutral-200 bg-white p-5 shadow-xl">
        <div className="mb-3 flex items-center justify-between">
          <div className="font-semibold text-neutral-900 text-lg">
            Make voice condition
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-2 text-neutral-500 hover:bg-neutral-100"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="space-y-4">
          <p className="text-sm text-neutral-500">
            This is test feature. In production, this will be set up on a
            separate page.
            <br />
            To make sure the generated voice matches your own, please record
            about 10 seconds of your voice in a quiet environment.
          </p>

          <div className="rounded-xl border border-neutral-200 bg-neutral-50 p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm text-neutral-600">
                Status:{" "}
                <span className="font-medium text-neutral-900">
                  {status === "idle" && "Idle"}
                  {status === "recording" && "Recording…"}
                  {status === "stopped" && "Recorded"}
                </span>
              </div>
              <div className="text-sm tabular-nums text-neutral-600">
                {formatDuration(durationMs)}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {status !== "recording" ? (
                <button
                  onClick={() => {
                    handleReset();
                    startRecording();
                  }}
                  className="inline-flex items-center gap-2 rounded-lg bg-neutral-900 px-3 py-2 text-sm text-white hover:bg-neutral-800 disabled:opacity-50"
                >
                  <Mic className="size-4" />
                  Start recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="inline-flex items-center gap-2 rounded-lg bg-red-700 px-3 py-2 text-sm text-white hover:bg-red-700/90"
                >
                  <Square className="size-4 fill-white" />
                  Stop recording
                </button>
              )}

              {blob && (
                <button
                  onClick={handleReset}
                  disabled={status === "recording"}
                  className="ml-1 rounded-lg px-3 py-2 text-sm text-neutral-700 hover:bg-neutral-200/70 disabled:opacity-50"
                >
                  Reset
                </button>
              )}
            </div>

            {audioUrl && (
              <div className="mt-3 flex items-center gap-3 rounded-lg border border-neutral-200 bg-white p-3">
                <button
                  onClick={togglePlay}
                  className="rounded-md bg-neutral-800 p-2 text-white hover:bg-neutral-700"
                  aria-label={isPlaying ? "Pause" : "Play"}
                >
                  {isPlaying ? (
                    <Pause className="size-4 fill-white" />
                  ) : (
                    <Play className="size-4 fill-white" />
                  )}
                </button>
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  onEnded={onAudioEnded}
                  preload="metadata"
                  className="hidden"
                />
                <div className="text-sm text-neutral-700">
                  {mimeType} • {Math.round((blob?.size ?? 0) / 1024)} KB
                </div>
              </div>
            )}

            {error && (
              <div className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
                {error}
              </div>
            )}
          </div>
        </div>

        <div className="mt-5 flex items-center justify-between"></div>
      </div>
    </div>
  );
}
export async function arrayBufferToPCM16LE24kBase64(
  arrayBuffer: ArrayBuffer
): Promise<string> {
  const AC = (window.AudioContext || (window as any).webkitAudioContext) as any;
  const ac: AudioContext = new AC();
  const audioBuf: AudioBuffer = await ac.decodeAudioData(arrayBuffer);
  await ac.close();

  // mixdown to mono if needed
  const n = audioBuf.length;
  const chs = audioBuf.numberOfChannels;
  let mono = new Float32Array(n);
  if (chs === 1) {
    audioBuf.copyFromChannel(mono, 0);
  } else {
    const tmp = new Float32Array(n);
    for (let c = 0; c < chs; c++) {
      audioBuf.copyFromChannel(tmp, c);
      for (let i = 0; i < n; i++) mono[i] += tmp[i];
    }
    const inv = 1 / chs;
    for (let i = 0; i < n; i++) mono[i] *= inv;
  }

  // resample to 24kHz
  const mono24k = resampleLinearMono(mono, audioBuf.sampleRate, TARGET_SR);

  // float32 -> int16 -> base64
  const pcm16 = floatTo16BitPCM(mono24k);
  return int16ToBase64(pcm16);
}

// ---- ADD: try to resolve a Blob from VoiceSample or URL ----
export async function resolveSampleBlob(sample: any): Promise<Blob | null> {
  if (!sample) return null;
  if (sample.blob instanceof Blob) return sample.blob;
  if (sample.file instanceof Blob) return sample.file as Blob;
  if (typeof sample.url === "string") {
    const r = await fetch(sample.url);
    return await r.blob();
  }
  return null;
}

// ---------------------------
// Helpers: Base64 / WAV / PCM utils
// ---------------------------
const TARGET_SR = 24000;

export function floatTo16BitPCM(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

export function int16ToBase64(int16: Int16Array): string {
  const bytes = new Uint8Array(int16.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++)
    binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export function arrayBufferToBase64(buf: ArrayBuffer) {
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

export function base64ToUint8Array(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function pcm16ToWav(
  pcm: Int16Array,
  sampleRate = TARGET_SR,
  channels = 1
) {
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + pcm.length * bytesPerSample);
  const view = new DataView(buffer);

  let offset = 0;
  const writeString = (s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i));
  };

  writeString("RIFF");
  view.setUint32(offset, 36 + pcm.length * bytesPerSample, true);
  offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true);
  offset += 4; // subchunk1 size
  view.setUint16(offset, 1, true);
  offset += 2; // PCM
  view.setUint16(offset, channels, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, sampleRate * blockAlign, true);
  offset += 4;
  view.setUint16(offset, blockAlign, true);
  offset += 2;
  view.setUint16(offset, bytesPerSample * 8, true);
  offset += 2;
  writeString("data");
  view.setUint32(offset, pcm.length * bytesPerSample, true);
  offset += 4;

  for (let i = 0; i < pcm.length; i++, offset += 2)
    view.setInt16(offset, pcm[i], true);
  return buffer;
}

// Simple linear resampler for mono Float32
export function resampleLinearMono(
  input: Float32Array,
  inRate: number,
  outRate: number
): Float32Array {
  if (inRate === outRate) return input;
  const ratio = outRate / inRate;
  const newLen = Math.floor(input.length * ratio);
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const srcIndex = i / ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = srcIndex - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}
