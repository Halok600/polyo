"use client";

// The recurring "instrument case" frame: four corner brackets that snap in
// clockwise from top-left when a panel mounts, and a numbered channel tag
// in place of a plain muted caption -- ties the chart, chips, distribution
// and code panel into one visible family instead of four card styles.

import type { CSSProperties, ReactNode } from "react";

type BenchPanelProps = {
  channel: string;
  revealDelayMs?: number;
  style?: CSSProperties;
  children: ReactNode;
};

export function BenchPanel({ channel, revealDelayMs = 0, style, children }: BenchPanelProps) {
  const mergedStyle = { ...style, "--corner-delay": `${revealDelayMs}ms` } as CSSProperties;

  return (
    <div className="bench-reveal bench-panel" style={mergedStyle}>
      <span className="bench-corner tl" />
      <span className="bench-corner tr" />
      <span className="bench-corner br" />
      <span className="bench-corner bl" />
      <span className="bench-channel-label">{channel}</span>
      {children}
    </div>
  );
}
