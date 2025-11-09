import React, { useMemo } from "react";

export function MicPulseRings({ size = 240, count = 4 }) {
  const rings = useMemo(() => {
    return Array.from({ length: count }).map((_, i) => {
      const dur = 900 + Math.random() * 800;
      const delay = -Math.random() * dur;
      const scaleEnd = 1.25 + Math.random() * 0.7;
      const opacity = 0.38 + Math.random() * 0.22;
      const blur = Math.random() < 0.5 ? 0 : 1 + Math.random() * 1.5;
      const soft = Math.random() < 0.5;

      return {
        key: `ring-${i}`,
        className: `ring-center ${
          soft ? "animate-ripple-soft" : "animate-ripple"
        }`,
        style: {
          ["--dur" as any]: `${dur}ms`,
          ["--delay" as any]: `${delay}ms`,
          ["--scale-start" as any]: 1,
          ["--scale-end" as any]: scaleEnd,
          ["--opacity" as any]: opacity,
          ["--blur" as any]: `${blur}px`,
        },
      };
    });
  }, [count]);

  return (
    <div
      className="absolute inset-1/2 -translate-x-1/2 -translate-y-1/2"
      style={{ width: size, height: size }}
    >
      {rings.map(({ key, className, style }) => (
        <div
          key={key}
          className={className}
          style={{
            ...style,
            width: size,
            height: size,
          }}
        />
      ))}
    </div>
  );
}
