// app/page.tsx
"use client";

import { useEffect, useRef, useState } from "react";

export default function Page() {
  const wsRef = useRef<WebSocket | null>(null);
  const [resp, setResp] = useState<any>(null);

  const onPick = async (f: File) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    // optional: send a tiny header first (type, name, mime)
    wsRef.current.send(
      JSON.stringify({ t: "audio", name: f.name, mime: f.type || "audio/wav" })
    );
    // send raw bytes as one WS message
    const buf = await f.arrayBuffer();
    wsRef.current.send(buf);
  };

  return (
    <div style={{ padding: 24 }}>
      <input
        type="file"
        accept="audio/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onPick(file);
        }}
      />
      {resp && (
        <pre style={{ marginTop: 16 }}>
          {JSON.stringify(
            {
              sr: resp.sr,
              length: resp.length,
              audio_head10: resp.audio?.slice(0, 10),
            },
            null,
            2
          )}
        </pre>
      )}
    </div>
  );
}
