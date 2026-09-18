// One horizontal bar per class, in ordinal order (plan §11's item 3) --
// the predicted class is drawn in the bold "predicted" hue, the rest in a
// muted ghost tone, matching the growth chart's bold/ghost convention
// rather than a full categorical palette (there's one thing to look at,
// not eight things to tell apart).

type ProbabilityBarsProps = {
  classes: string[];
  distribution: Record<string, number>;
  predictedClass: string;
};

export function ProbabilityBars({ classes, distribution, predictedClass }: ProbabilityBarsProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {classes.map((cls) => {
        const value = distribution[cls] ?? 0;
        const percent = Math.round(value * 100);
        const isPredicted = cls === predictedClass;
        return (
          <div key={cls} style={{ display: "grid", gridTemplateColumns: "88px 1fr 48px", gap: 10, alignItems: "center" }}>
            <span
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
              aria-label={`${cls}: ${percent}%`}
              style={{ height: 10, borderRadius: 4, background: "var(--gridline)", overflow: "hidden" }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${percent}%`,
                  borderRadius: 4,
                  background: isPredicted ? "var(--series-predicted)" : "var(--series-ghost)",
                }}
              />
            </div>
            <span style={{ fontSize: 13, color: "var(--text-secondary)", textAlign: "right" }}>
              {percent}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
