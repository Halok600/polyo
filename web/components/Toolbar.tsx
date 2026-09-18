import type { ReactNode } from "react";

// Rendered identically by the landing scene and the tool scene -- a View
// Transition cross-fades two pixel-identical regions into a no-op, so the
// header just holds still across the landing->tool morph without needing
// its own view-transition-name.

type ToolbarProps = {
  right: ReactNode;
};

export function Toolbar({ right }: ToolbarProps) {
  return (
    <header className="bench-toolbar">
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span className="mono-nums" style={{ fontSize: 19, fontWeight: 700, letterSpacing: "0.01em", color: "var(--text-primary)" }}>
          POLY·O
        </span>
        <span className="mono-nums" style={{ fontSize: 10, letterSpacing: "0.08em", color: "var(--text-muted)" }}>
          REV 1
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>{right}</div>
    </header>
  );
}
