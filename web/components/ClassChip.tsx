"use client";

import { BenchPanel } from "@/components/BenchPanel";
import { useCountUp } from "@/lib/motion";

type ClassChipProps = {
  channel: string;
  label: string;
  predictedClass: string;
  confidence: number;
  footnote?: string;
  revealDelayMs?: number;
};

export function ClassChip({ channel, label, predictedClass, confidence, footnote, revealDelayMs = 0 }: ClassChipProps) {
  const target = Math.round(confidence * 100);
  const displayed = useCountUp(target, 500, revealDelayMs);
  const percent = Math.round(displayed);
  const filled = Math.round((percent / 100) * 10);

  return (
    <BenchPanel channel={channel} revealDelayMs={revealDelayMs} style={{ flex: "1 1 240px" }}>
      <div className="mono-nums" style={{ fontSize: 11, letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 8 }}>
        {label.toUpperCase()}
      </div>
      <div className="mono-nums" style={{ fontSize: 30, fontWeight: 600, color: "var(--text-primary)", lineHeight: 1.1 }}>
        {predictedClass}
      </div>

      <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 10 }}>
        <span className="mono-nums" aria-hidden style={{ fontSize: 12, color: "var(--signal)", letterSpacing: "-0.02em" }}>
          [{"█".repeat(filled)}{"░".repeat(10 - filled)}]
        </span>
        <div
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label} confidence`}
          className="mono-nums"
          style={{ fontSize: 13, color: "var(--text-secondary)" }}
        >
          {percent}%
        </div>
      </div>

      {footnote ? <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-muted)" }}>{footnote}</div> : null}
    </BenchPanel>
  );
}
