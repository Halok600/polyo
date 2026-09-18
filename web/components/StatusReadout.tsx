// Shared "instrument light" readout -- a colored dot plus a mono label,
// reused by both scenes so the landing page's toolbar looks like the same
// instrument that's about to power on, not separate chrome.

export type Status = "ready" | "sampling" | "done";

export function StatusReadout({ status }: { status: Status }) {
  const color = status === "sampling" ? "var(--signal)" : status === "done" ? "var(--calibration)" : "var(--text-muted)";
  const label = status === "sampling" ? "SAMPLING" : status === "done" ? "DONE" : "READY";
  return (
    <div className="mono-nums" style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 11, letterSpacing: "0.08em", color }}>
      <span
        aria-hidden
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: color,
          animation: status === "sampling" ? "bench-pulse 900ms ease-in-out infinite" : "none",
        }}
      />
      {label}
    </div>
  );
}
