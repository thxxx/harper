"use client";

import React, { useRef, useState } from "react";

export function useAudioRecorder() {
  const [status, setStatus] = useState<"idle" | "recording" | "stopped">(
    "idle"
  );
  const [durationMs, setDurationMs] = useState(0);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [mimeType, setMimeType] = useState<string>("audio/webm");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const startAtRef = useRef<number>(0);
  const timerRef = useRef<number | null>(null);

  // 외부 샘플 중복 주입 방지용 시그
  const externalSigRef = useRef<string | null>(null);
  const makeSig = (b?: Blob | null, mt?: string | null) =>
    b ? `${b.size}:${mt ?? ""}` : "";

  const tick = () => {
    if (status === "recording") {
      setDurationMs(Date.now() - startAtRef.current);
      timerRef.current = window.setTimeout(tick, 200) as unknown as number;
    }
  };

  // 외부에서 Blob 주입
  const setExternalSample = (externalBlob: Blob, mt = "audio/webm") => {
    const nextSig = makeSig(externalBlob, mt);
    if (externalSigRef.current === nextSig) return; // 동일 샘플이면 무시
    externalSigRef.current = nextSig;

    // 녹음 중이면 정지
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      try {
        mediaRecorderRef.current.stop();
      } catch {}
    }

    setMimeType(mt);
    setBlob(externalBlob);
    setStatus("stopped");
    setDurationMs(0);
  };

  const startRecording = async () => {
    if (status === "recording") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000, // 브라우저 기본
          sampleSize: 16, // 일부 브라우저만 반영
          echoCancellation: true, // 스피커로 재생도 한다면 켜기 (AEC)
          noiseSuppression: true, // 1차 노이즈 억제
          autoGainControl: false, // 크기 “과다·펌핑” 문제의 주범 → 끄기
        },
        video: false,
      });

      const mt =
        // 최신 크롬/엣지
        (window as any).MediaRecorder?.isTypeSupported?.(
          "audio/webm;codecs=opus"
        )
          ? "audio/webm;codecs=opus"
          : // 파이어폭스
          (window as any).MediaRecorder?.isTypeSupported?.("audio/webm")
          ? "audio/webm"
          : // 사파리 폴백
            "audio/mp4";

      setMimeType(mt.includes("webm") ? "audio/webm" : "audio/mp4");

      const mr = new MediaRecorder(stream, { mimeType: mt });
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      setBlob(null);
      externalSigRef.current = null; // 내부 녹음 시작하면 외부 시그 초기화

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const recorded = new Blob(chunksRef.current, { type: mimeType });
        setBlob(recorded);
        stream.getTracks().forEach((t) => t.stop());
      };

      mr.start();
      startAtRef.current = Date.now();
      setDurationMs(0);
      setStatus("recording");
    } catch (e) {
      console.error("Mic permission / start error:", e);
      setStatus("idle");
    }
  };

  const stopRecording = () => {
    if (status !== "recording" || !mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    setStatus("stopped");
    tick();
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const reset = () => {
    setStatus("idle");
    setDurationMs(0);
    setBlob(null);
    externalSigRef.current = null;
    chunksRef.current = [];
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      try {
        mediaRecorderRef.current.stop();
      } catch {}
    }
  };

  return {
    status,
    durationMs,
    blob,
    mimeType,
    startRecording,
    stopRecording,
    reset,
    setExternalSample,
  };
}
