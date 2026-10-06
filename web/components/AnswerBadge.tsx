// The provenance badge on a result: CERTAIN / ASSUMED (the static analyser) or MODEL (the learned
// fallback). Colour is never the only signal: the label says it, and the title explains it.

import type { CSSProperties } from "react";

import { answerSource, type AnswerTone } from "@/lib/derivation";

const TONE_COLOR: Record<AnswerTone, string> = {
  good: "var(--status-good)",
  signal: "var(--signal)",
  model: "var(--calibration)",
};

type AnswerBadgeProps = {
  engine: string | undefined;
  certainty: string | null | undefined;
};

export function AnswerBadge({ engine, certainty }: AnswerBadgeProps) {
  const source = answerSource(engine, certainty);
  const style = { "--badge-color": TONE_COLOR[source.tone] } as CSSProperties;
  return (
    <span className="mono-nums answer-badge" style={style} title={source.detail} data-tone={source.tone}>
      {source.label}
    </span>
  );
}
