import { describe, expect, it } from "vitest";

import { answerSource, markersByLine, totalSteps } from "@/lib/derivation";
import type { AssumptionItem, DerivationStep } from "@/lib/types";

const STEPS: DerivationStep[] = [
  { line: 3, kind: "loop", text: "loop runs O(n) times; the whole loop costs O(n * m)" },
  { line: 4, kind: "loop", text: "loop runs O(m) times; the whole loop costs O(m)" },
  { line: 4, kind: "alloc", text: "allocates O(m)" },
  { line: 0, kind: "total", text: "time: O(n * m)" },
  { line: 0, kind: "total", text: "space: O(1)" },
];

describe("markersByLine", () => {
  it("groups the steps by their 1-based source line and leaves the totals out", () => {
    const markers = markersByLine(STEPS, []);
    expect([...markers.keys()]).toEqual([3, 4]);
    expect(markers.get(4)?.steps.map((s) => s.kind)).toEqual(["loop", "alloc"]);
  });

  it("attaches assumptions to their line, creating the entry when no step is there", () => {
    const assumptions: AssumptionItem[] = [{ line: 7, reason: "loop bound assumed" }];
    const markers = markersByLine(STEPS, assumptions);
    expect(markers.get(7)).toEqual({ steps: [], assumptions });
    expect(markers.get(3)?.assumptions).toEqual([]);
  });

  it("ignores an assumption with an unknown line (0) instead of marking line 0", () => {
    const markers = markersByLine([], [{ line: 0, reason: "unknown library call" }]);
    expect(markers.size).toBe(0);
  });

  it("is empty for a model answer, which has no derivation", () => {
    expect(markersByLine(undefined, undefined).size).toBe(0);
  });
});

describe("totalSteps", () => {
  it("returns only the closing time / space totals, in order", () => {
    expect(totalSteps(STEPS).map((s) => s.text)).toEqual(["time: O(n * m)", "space: O(1)"]);
  });

  it("tolerates a missing derivation", () => {
    expect(totalSteps(undefined)).toEqual([]);
  });
});

describe("answerSource", () => {
  it("calls a proven symbolic answer certain", () => {
    expect(answerSource("symbolic", "certain")).toMatchObject({ label: "CERTAIN", tone: "good" });
  });

  it("calls a symbolic answer with an assumed bound assumed, never certain", () => {
    expect(answerSource("symbolic", "assumed")).toMatchObject({ label: "ASSUMED", tone: "signal" });
  });

  it("says plainly that the model answered when the engine could not", () => {
    const view = answerSource("ml_fallback", null);
    expect(view.tone).toBe("model");
    expect(view.label).toBe("MODEL");
    expect(view.detail).toMatch(/could not bound/i);
  });

  it("treats a legacy response (no engine field) as a model answer", () => {
    expect(answerSource(undefined, undefined)).toMatchObject({ label: "MODEL", tone: "model" });
  });

  it("does not claim certainty for a symbolic answer without a certainty field", () => {
    expect(answerSource("symbolic", undefined).label).toBe("ASSUMED");
  });
});
