type ClassChipProps = {
  label: string;
  predictedClass: string;
  confidence: number;
  footnote?: string;
};

export function ClassChip({ label, predictedClass, confidence, footnote }: ClassChipProps) {
  const percent = Math.round(confidence * 100);
  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: 12,
        padding: "16px 20px",
        background: "var(--surface-1)",
        flex: "1 1 240px",
      }}
    >
      <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 600, color: "var(--text-primary)" }}>
        {predictedClass}
      </div>
      <div
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label} confidence`}
        style={{
          marginTop: 10,
          height: 8,
          borderRadius: 4,
          background: "var(--gridline)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${percent}%`,
            borderRadius: 4,
            background: "var(--series-predicted)",
          }}
        />
      </div>
      <div style={{ marginTop: 6, fontSize: 13, color: "var(--text-secondary)" }}>
        {percent}% confidence
      </div>
      {footnote ? (
        <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-muted)" }}>{footnote}</div>
      ) : null}
    </div>
  );
}
