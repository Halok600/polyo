// Pure helpers behind the derivation view: which source lines the engine's steps and assumptions
// point at, and how an answer's provenance is labelled. Kept out of the components so the edge
// cases (a model answer with no derivation, an assumption with no line) are unit-tested.

import type { AssumptionItem, DerivationStep } from "@/lib/types";

export type LineMarker = { steps: DerivationStep[]; assumptions: AssumptionItem[] };

// 1-based source line -> what the engine said about it. The closing "time"/"space" totals have
// line 0 and an assumption may too (unknown line): neither belongs to a line, so neither is marked.
export function markersByLine(
  derivation: DerivationStep[] | undefined,
  assumptions: AssumptionItem[] | undefined,
): Map<number, LineMarker> {
  const markers = new Map<number, LineMarker>();
  const at = (line: number): LineMarker => {
    let marker = markers.get(line);
    if (!marker) {
      marker = { steps: [], assumptions: [] };
      markers.set(line, marker);
    }
    return marker;
  };
  for (const step of derivation ?? []) {
    if (step.line > 0) at(step.line).steps.push(step);
  }
  for (const assumption of assumptions ?? []) {
    if (assumption.line > 0) at(assumption.line).assumptions.push(assumption);
  }
  return markers;
}

export function totalSteps(derivation: DerivationStep[] | undefined): DerivationStep[] {
  return (derivation ?? []).filter((step) => step.kind === "total");
}

export type AnswerTone = "good" | "signal" | "model";

export type AnswerSource = {
  label: "CERTAIN" | "ASSUMED" | "MODEL";
  tone: AnswerTone;
  detail: string;
};

// Provenance of one dimension's answer. Never claims more than the API said: a symbolic answer
// without an explicit "certain" is shown as assumed, and a response from before the engine
// existed (no `engine` field) is a model answer.
export function answerSource(engine: string | undefined, certainty: string | null | undefined): AnswerSource {
  if (engine === "symbolic") {
    if (certainty === "certain") {
      return { label: "CERTAIN", tone: "good", detail: "Every loop bound and recursion was proven by the static analyser." };
    }
    return {
      label: "ASSUMED",
      tone: "signal",
      detail: "The analyser had to assume at least one bound, so this may over-estimate. See the assumptions below.",
    };
  }
  if (engine === "ml_fallback") {
    return {
      label: "MODEL",
      tone: "model",
      detail: "The static analyser could not bound this code, so the learned model answered. Treat it as an estimate.",
    };
  }
  return { label: "MODEL", tone: "model", detail: "Answered by the learned model." };
}
