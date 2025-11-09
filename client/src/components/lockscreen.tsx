import {
  ChevronRight,
  Mic,
  MicOff,
  Phone,
  PhoneCall,
  PhoneOff,
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { MicPulseRings } from "./audio/MicPulseRing";
import { AGENT_NAME } from "@/pages";

type SliderState = "idle" | "dragging" | "answered";

const clamp = (v: number, min: number, max: number) =>
  Math.min(Math.max(v, min), max);

export default function LockScreenCall({
  onAnswer,
  onEnd,
  onMute,
  connected,
  isSessionLoading,
  micLevel,
  isSpeaking,
  recording,
}: {
  onAnswer: () => void;
  onEnd: () => void;
  onMute: () => void;
  connected: boolean;
  isSessionLoading: boolean;
  micLevel: number;
  isSpeaking: boolean;
  recording: boolean;
}) {
  const [sliderState, setSliderState] = useState<SliderState>("idle");
  const [knobX, setKnobX] = useState(0);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const startX = useRef(0);
  const startKnobX = useRef(0);

  const sizes = useMemo(() => {
    const track = trackRef.current;
    const wrap = wrapRef.current;
    const wrapW = wrap?.clientWidth ?? 390;
    const trackW = track?.clientWidth ?? Math.min(320, wrapW - 32);
    const knobW = 72;
    const maxX = trackW - knobW - 8;
    return { wrapW, trackW, knobW, maxX };
  }, [trackRef.current, wrapRef.current, sliderState]);

  useEffect(() => {
    if (sliderState === "idle") setKnobX(0);
  }, [sliderState]);

  const percent = sizes.maxX > 0 ? knobX / sizes.maxX : 0;

  const onPointerDown = (e: React.PointerEvent) => {
    if (sliderState === "answered") return;
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    setSliderState("dragging");
    startX.current = e.clientX;
    startKnobX.current = knobX;
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (sliderState !== "dragging") return;
    const delta = e.clientX - startX.current;
    setKnobX((prev) => clamp(startKnobX.current + delta, 0, sizes.maxX));
  };

  const onPointerUp = () => {
    if (sliderState !== "dragging") return;
    if (percent >= 0.85) {
      setSliderState("answered");
      setKnobX(sizes.maxX);
      onAnswer();
    } else {
      setSliderState("idle");
    }
  };

  return (
    <div className="w-full max-w-xl justify-center items-center mx-auto pt-12">
      {/* Phone frame */}
      <div
        ref={wrapRef}
        className="relative w-[640px] max-w-full aspect-[1] rounded-[1rem] border border-white/10 overflow-hidden bg-black/0"
      >
        <div className="relative flex items-center justify-center pt-4">
          <div
            className={`w-[280px] h-[280px] flex items-center justify-center transition-all cursor-pointer hover:opacity-90 rounded-full ${
              connected
                ? "bg-[linear-gradient(45deg,#6d28d9,#8b5cf6,#c084fc,#e879f9,#f472b6)] bg-[length:300%_300%] animate-gradientx"
                : "bg-[#8b5cf6]/20"
            } ${isSessionLoading ? "bg-green" : ""}`}
          >
            {!connected ? (
              <span className="md:text-2xl text-lg font-medium text-black/50 pointer-events-none">
                {isSessionLoading ? "Connecting..." : "Not connected"}
              </span>
            ) : (
              <span className=""></span>
            )}
          </div>
          {/* 외곽 파동 링 (isSpeaking일 때만) */}
          {isSpeaking && (
            <>
              <MicPulseRings />
            </>
          )}
        </div>

        {/* In-call UI after answer */}
        {sliderState === "answered" && (
          <div className="absolute inset-0 flex flex-col items-center justify-end pb-8">
            <div className="flex gap-8">
              {/* Mute */}
              <button
                onClick={onMute}
                className="group inline-flex flex-col items-center gap-3 cursor-pointer"
                aria-label="Mute"
              >
                <span className="hover:opacity-90 w-20 h-20 rounded-full border-black/30 bg-white/10 backdrop-blur border flex items-center justify-center transition active:scale-95">
                  {recording ? (
                    <Mic className="w-6 h-6 text-black/90" />
                  ) : (
                    <MicOff className="w-6 h-6 text-black/90" />
                  )}
                </span>
                <span className="text-lg text-black/80 font-light">
                  {recording ? "Mute" : "Unmute"}
                </span>
              </button>
              {/* End */}
              <button
                onClick={() => {
                  onEnd();
                  setSliderState("idle");
                }}
                className="group inline-flex flex-col items-center gap-3 cursor-pointer"
                aria-label="End Call"
              >
                <span className="hover:opacity-90 w-20 h-20 rounded-full bg-red-600/90 shadow-lg shadow-red-900/40 flex items-center justify-center transition active:scale-95">
                  <PhoneOff className="w-6 h-6 text-white" />
                </span>
                <span className="text-lg text-black/90 font-light">
                  End Call
                </span>
              </button>
            </div>
          </div>
        )}

        {/* Bottom slide-to-answer (hidden once answered) */}
        {sliderState !== "answered" && (
          <div className="absolute left-0 right-0 bottom-0 px-5 pb-16 select-none">
            <div className="flex flex-col items-center gap-3">
              <div
                ref={trackRef}
                className="relative w-full max-w-[340px] h-20 rounded-full bg-purple-400/10 backdrop-blur border border-black/15 overflow-hidden"
              >
                {/* Fill based on percent */}
                <div
                  className="absolute inset-y-0 left-0 bg-gradient-to-r from-purple-400/20 to-purple-400/0 transition-[width] duration-150"
                  style={{
                    width: `${Math.max(knobX + sizes.knobW * 0.6, 0)}px`,
                  }}
                />

                {percent < 0.15 && (
                  <div className="absolute inset-0 flex items-center pl-16">
                    <ChevronRight className="w-4 h-4 text-black/70 animate-pulse" />
                    <ChevronRight className="w-4 h-4 text-black/60 animate-pulse delay-150" />
                    <ChevronRight className="w-4 h-4 text-black/50 animate-pulse delay-300" />
                  </div>
                )}

                {/* Knob */}
                <div
                  role="button"
                  aria-label="Slide to answer"
                  tabIndex={0}
                  onPointerDown={onPointerDown}
                  onPointerMove={onPointerMove}
                  onPointerUp={onPointerUp}
                  onPointerCancel={onPointerUp}
                  className={[
                    "absolute top-1/2 -translate-y-1/2 w-[72px] h-[72px] rounded-full",
                    "bg-white text-black shadow-xl flex items-center justify-center",
                    "active:scale-[0.98] transition-transform",
                    sliderState === "idle"
                      ? "animate-[nudge_2.2s_ease-in-out_infinite]"
                      : "",
                  ].join(" ")}
                  style={{ left: `${knobX + 6}px` }}
                >
                  <Phone className="w-5 h-5" />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Simple CSS keyframes */}
        <style>{`
          @keyframes pulse {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-2%); }
          }
          @keyframes shimmer {
            0% { transform: translateX(-120%); }
            100% { transform: translateX(120%); }
          }
          @keyframes nudge {
            0%, 60%, 100% { transform: translate(-0%, -50%); }
            30% { transform: translate(8%, -50%); }
          }
        `}</style>
      </div>
    </div>
  );
}
