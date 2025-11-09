import React from "react";
import { AGENT_NAME } from "@/pages/index";

type Props = {
  responses: {
    type: "user" | typeof AGENT_NAME;
    text: string;
  }[];
  scripting: string;
};

const RoleLabel = ({ label }: { label: string }) => (
  <span className="text-[11px] tracking-wide text-zinc-500 font-medium select-none">
    {label}
  </span>
);

const Line = ({
  role,
  text,
}: {
  role: "You" | typeof AGENT_NAME;
  text: string;
}) => (
  <div className="group">
    <div className="flex items-baseline gap-2">
      <RoleLabel label={role} />
    </div>
    <div
      className={
        "mt-1 pl-3 border-l-2 text-[15px] leading-relaxed whitespace-pre-wrap break-words " +
        (role === "You"
          ? "border-emerald-300/70 hover:border-emerald-400/90"
          : "border-zinc-300 hover:border-zinc-400")
      }
    >
      {text}
    </div>
  </div>
);

const Placeholder = () => (
  <div className="px-4 py-10 text-center text-zinc-400 text-sm">
    Your conversation will appear here.
  </div>
);

const ConversationHistory: React.FC<Props> = ({ responses, scripting }) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null);

  // Build a flat list from responses in given order; append scripting as user's line if present
  const lines = React.useMemo(() => {
    const mapRole = (
      t: "user" | typeof AGENT_NAME
    ): "You" | typeof AGENT_NAME => (t === "user" ? "You" : AGENT_NAME);

    const out = responses.map((r) => ({
      role: mapRole(r.type),
      text: r.text,
    })) as Array<{ role: "You" | typeof AGENT_NAME; text: string }>;

    if (scripting && scripting.trim().length > 0) {
      out.push({ role: "You", text: scripting });
    }
    return out;
  }, [responses, scripting]);

  // Auto scroll to bottom on updates
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const behavior: ScrollBehavior = el.scrollTop > 0 ? "smooth" : "auto";
    el.scrollTo({ top: el.scrollHeight, behavior });
  }, [lines]);

  return (
    <section className="max-w-3xl mx-auto mt-10 px-4">
      <div className="bg-white rounded-2xl border border-zinc-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 text-xs uppercase tracking-wide text-zinc-500 bg-zinc-50">
          Conversation
        </div>

        <div
          ref={containerRef}
          className="h-[52vh] sm:h-[60vh] overflow-y-auto px-4 py-4 space-y-4 divide-y divide-transparent"
        >
          {lines.length === 0 ? (
            <Placeholder />
          ) : (
            lines.map((l, idx) => (
              <Line key={`line_${idx}`} role={l.role} text={l.text} />
            ))
          )}
        </div>
      </div>
    </section>
  );
};

export default React.memo(ConversationHistory);
