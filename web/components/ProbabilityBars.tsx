"use client";

// One horizontal bar per class, in ordinal order -- the predicted class
// draws in the signal (amber) hue, the rest in the muted ghost-trace tone,
// matching the growth chart's bold/ghost convention. Rows lock in
// top-to-bottom like a meter reading settling, not fading in as a group.

import type { CSSProperties } from "react";

import { useCountUp } from "@/lib/motion";

type ProbabilityBarsProps = {
  classes: string[];
  distribution: Record<string, number>;
  predictedClass: string;
};

function Row({
  cls,
  value,
  isPredicted,
  delayMs,
}: {
  cls: string;
  value: number;
  isPredicted: boolean;
  delayMs: number;
}) {
  const target = Math.round(value * 100);
  const displayed = useCountUp(target, 450, delayMs);
  const percent = Math.round(displayed);
  const rowStyle = { "--row-delay": `${delayMs}ms` } as CSSProperties;

  return (
    <div
      className="bench-reveal-row"
      style={{ ...rowStyle, display: "grid", gridTemplateColumns: "92px 1fr 48px", gap: 10, alignItems: "center" }}
    >
      <span
        className="mono-nums"
        style={{
          fontSize: 13,
          color: isPredicted ? "var(--text-primary)" : "var(--text-secondary)",
          fontWeight: isPredicted ? 600 : 400,
        }}
      >
        {cls}
      </span>
      <div
        role="img"
        aria-label={`${cls}: ${Math.round(value * 100)}%`}
        style={{ height: 8, border: "1px solid var(--border)", background: "var(--surface-2)", overflow: "hidden" }}
      >
        <div
          style={{
            height: "100%",
            width: `${percent}%`,
            background: isPredicted ? "var(--signal)" : "var(--ghost-trace)",
            transition: "width 450ms var(--ease-settle)",
            transitionDelay: `${delayMs}ms`,
          }}
        />
      </div>
      <span className="mono-nums" style={{ fontSize: 13, color: "var(--text-secondary)", textAlign: "right" }}>
        {percent}%
      </span>
    </div>
  );
}

export function ProbabilityBars({ classes, distribution, predictedClass }: ProbabilityBarsProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {classes.map((cls, i) => (
        <Row
          key={cls}
          cls={cls}
          value={distribution[cls] ?? 0}
          isPredicted={cls === predictedClass}
          delayMs={i * 60}
        />
      ))}
    </div>
  );
}
