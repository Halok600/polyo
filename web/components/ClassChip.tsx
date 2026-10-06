"use client";

import { AnswerBadge } from "@/components/AnswerBadge";
import { BenchPanel } from "@/components/BenchPanel";
import { useCountUp } from "@/lib/motion";

type ClassChipProps = {
  channel: string;
  label: string;
  predictedClass: string;
  confidence: number;
  conformalSet: string[];
  conformalCoverage: number;
  abstain: boolean;
  // v2 -- all optional, so a response from before the symbolic engine still renders as it did.
  engine?: string;
  certainty?: string | null;
  expression?: string | null;
  extendedClass?: string | null;
  projectionLossy?: boolean;
  footnote?: string;
  revealDelayMs?: number;
};

export function ClassChip({
  channel,
  label,
  predictedClass,
  confidence,
  conformalSet,
  conformalCoverage,
  abstain,
  engine,
  certainty,
  expression,
  extendedClass,
  projectionLossy = false,
  footnote,
  revealDelayMs = 0,
}: ClassChipProps) {
  const target = Math.round(confidence * 100);
  const displayed = useCountUp(target, 500, revealDelayMs);
  const percent = Math.round(displayed);
  const filled = Math.round((percent / 100) * 10);
  const coveragePercent = Math.round(conformalCoverage * 100);
  // A one-class set is exactly the point prediction above -- nothing new
  // to say. A wider set is the honest complement to the confidence number:
  // "usually within this range" read as a real range, not a coin flip.
  const isRange = conformalSet.length > 1;
  // The engine's exact expression leads; the 7-class answer becomes the secondary chip. A model
  // answer has no expression, so it keeps the class as the headline exactly as before.
  const headline = expression ?? predictedClass;
  const symbolic = engine === "symbolic";
  const showExtended = Boolean(extendedClass) && extendedClass !== predictedClass;
  // Repeating the headline as a chip says nothing; the class chip appears when it is a different
  // (rounded-up) statement of the answer.
  const showClassChip = Boolean(expression) && (projectionLossy || expression !== predictedClass);

  return (
    <BenchPanel channel={channel} revealDelayMs={revealDelayMs} style={{ flex: "1 1 240px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
        <span className="mono-nums" style={{ fontSize: 11, letterSpacing: "0.08em", color: "var(--text-muted)" }}>
          {label.toUpperCase()}
        </span>
        <AnswerBadge engine={engine} certainty={certainty} />
      </div>
      <div
        className="mono-nums"
        data-testid={`${label.toLowerCase()}-headline`}
        style={{ fontSize: 30, fontWeight: 600, color: "var(--text-primary)", lineHeight: 1.1, overflowWrap: "anywhere" }}
      >
        {headline}
      </div>

      {showClassChip || showExtended ? (
        <div className="mono-nums" style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          {showClassChip ? (
            <span className="class-chip" title={projectionLossy ? "Rounded up from the exact expression to the nearest class on the standard scale." : undefined}>
              CLASS {predictedClass}
              {projectionLossy ? " ↑" : ""}
            </span>
          ) : null}
          {showExtended ? <span className="class-chip">EXACT CLASS {extendedClass}</span> : null}
        </div>
      ) : null}

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
      {symbolic ? (
        <div className="mono-nums" style={{ marginTop: 6, fontSize: 11, color: "var(--text-muted)" }}>
          accuracy of {certainty === "certain" ? "certain" : "assumed"} answers, measured on independent programs
        </div>
      ) : null}

      {isRange ? (
        <div className="mono-nums" style={{ marginTop: 10, fontSize: 13 }}>
          <span style={{ color: "var(--text-primary)" }}>
            {conformalSet[0]} – {conformalSet[conformalSet.length - 1]}
          </span>
          <span style={{ color: "var(--text-muted)", fontSize: 11 }}> at {coveragePercent}% coverage</span>
        </div>
      ) : null}

      {abstain ? (
        <div
          className="mono-nums"
          style={{ marginTop: 8, fontSize: 11, color: "var(--signal)", borderLeft: "2px solid var(--signal)", paddingLeft: 8, lineHeight: 1.5 }}
        >
          [!] too uncertain to narrow down -- {conformalSet.length} classes still possible at {coveragePercent}% coverage
        </div>
      ) : null}

      {footnote ? <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-muted)" }}>{footnote}</div> : null}
    </BenchPanel>
  );
}
