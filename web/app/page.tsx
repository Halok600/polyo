"use client";

import { useEffect, useState } from "react";
import { flushSync } from "react-dom";

import { BenchPanel } from "@/components/BenchPanel";
import { ClassChip } from "@/components/ClassChip";
import { CodeEditor } from "@/components/CodeEditor";
import { CodeWithSpans } from "@/components/CodeWithSpans";
import { GrowthChart } from "@/components/GrowthChart";
import { Landing } from "@/components/Landing";
import { LanguageSelector } from "@/components/LanguageSelector";
import { ProbabilityBars } from "@/components/ProbabilityBars";
import { StatusReadout } from "@/components/StatusReadout";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Toolbar } from "@/components/Toolbar";
import { ApiError, fetchLanguages, predict } from "@/lib/api";
import { usePrefersReducedMotion, withViewTransition } from "@/lib/motion";
import type { LanguageOption, PredictResponse } from "@/lib/types";

const EXAMPLE_CODE = `def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[n] = i
    return []
`;

// "Better" (lower rank, faster) neighbour of a predicted class, if the
// curve series has one -- "what faster looks like".
function betterNeighbour(seriesKeys: string[], predictedClass: string, allRanked: string[]) {
  const predictedRank = allRanked.indexOf(predictedClass);
  if (predictedRank <= 0) return undefined;
  const better = allRanked[predictedRank - 1];
  return seriesKeys.includes(better) ? better : undefined;
}

const TIME_CLASSES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)"];
const SPACE_CLASSES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)"];

type Status = "ready" | "sampling" | "done";

export default function Home() {
  const [entered, setEntered] = useState(false);
  const [languages, setLanguages] = useState<LanguageOption[]>([]);
  const [language, setLanguage] = useState("auto");
  const [code, setCode] = useState(EXAMPLE_CODE);
  const [chartDimension, setChartDimension] = useState<"time" | "space">("time");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [status, setStatus] = useState<Status>("ready");
  const [runId, setRunId] = useState(0);
  const reducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    // Fetched unconditionally, before `entered` flips -- the language list
    // is already warm by the time someone clicks through the landing page.
    fetchLanguages()
      .then(setLanguages)
      .catch(() => setLanguages([]));
  }, []);

  function handleEnter() {
    withViewTransition(() => flushSync(() => setEntered(true)), reducedMotion, "vt-scene");
  }

  if (!entered) {
    return <Landing onEnter={handleEnter} />;
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setStatus("sampling");
    setError(null);
    try {
      const response = await predict(language, code);
      setResult(response);
      setRunId((id) => id + 1);
      setStatus("done");
      window.setTimeout(() => setStatus("ready"), 1400);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "something went wrong");
      setResult(null);
      setStatus("ready");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="bench-shell">
      <Toolbar
        right={
          <>
            <StatusReadout status={status} />
            <ThemeToggle />
          </>
        }
      />

      <div className="bench-grid">
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
            Static, multi-language time &amp; space complexity prediction. No LLM, no code execution.
          </p>
          <LanguageSelector languages={languages} value={language} onChange={setLanguage} />
          <BenchPanel channel="CH.00 — INPUT">
            <CodeEditor value={code} onChange={setCode} language={language} />
          </BenchPanel>
          <button
            type="submit"
            disabled={loading || code.trim().length === 0}
            className="mono-nums vt-run-cta"
            style={{
              alignSelf: "flex-start",
              padding: "11px 22px",
              borderRadius: 0,
              border: "none",
              background: "var(--signal)",
              color: "var(--page-plane)",
              fontSize: 13,
              fontWeight: 700,
              letterSpacing: "0.04em",
              cursor: loading ? "default" : "pointer",
              opacity: loading ? 0.65 : 1,
              transition: "opacity 150ms var(--ease-settle)",
            }}
          >
            {loading ? "SAMPLING…" : "RUN ANALYSIS"}
          </button>
          {error ? (
            <p role="alert" className="mono-nums" style={{ fontSize: 12, color: "var(--status-error)", borderLeft: "2px solid var(--status-error)", paddingLeft: 8 }}>
              [!] {error}
            </p>
          ) : null}
        </form>

        <div>
          {result ? (
            <section key={runId} style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              <p className="mono-nums" style={{ fontSize: 12, color: "var(--text-muted)" }}>
                LANG <span style={{ color: "var(--text-primary)" }}>{result.language_detected}</span>
              </p>

              <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                <ClassChip channel="CH.01 — TIME" label="Time" predictedClass={result.time.class} confidence={result.time.confidence} revealDelayMs={0} />
                <ClassChip
                  channel="CH.02 — SPACE"
                  label="Space"
                  predictedClass={result.space.class}
                  confidence={result.space.confidence}
                  footnote="Auxiliary space, including recursion stack."
                  revealDelayMs={60}
                />
              </div>

              <BenchPanel channel="CH.03 — GROWTH" revealDelayMs={120}>
                <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
                  {(["time", "space"] as const).map((dim) => (
                    <button
                      key={dim}
                      type="button"
                      onClick={() => setChartDimension(dim)}
                      className="mono-nums"
                      style={{
                        padding: "6px 16px",
                        borderRadius: 0,
                        border: "1px solid var(--border-strong)",
                        background: chartDimension === dim ? "var(--signal)" : "var(--surface-1)",
                        color: chartDimension === dim ? "var(--page-plane)" : "var(--text-secondary)",
                        fontSize: 11,
                        letterSpacing: "0.06em",
                        cursor: "pointer",
                      }}
                    >
                      {dim.toUpperCase()}
                    </button>
                  ))}
                </div>
                <GrowthChart
                  key={chartDimension}
                  title={chartDimension === "time" ? "Time complexity vs n" : "Space complexity vs n"}
                  n={result.curve.n}
                  series={result.curve[chartDimension].series}
                  predictedClass={result.curve[chartDimension].predicted_class}
                  betterClass={betterNeighbour(
                    Object.keys(result.curve[chartDimension].series),
                    result.curve[chartDimension].predicted_class,
                    chartDimension === "time" ? TIME_CLASSES : SPACE_CLASSES,
                  )}
                />
              </BenchPanel>

              <BenchPanel channel="CH.04 — DISTRIBUTION" revealDelayMs={180}>
                <div className="mono-nums" style={{ fontSize: 11, letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 14 }}>
                  CLASS PROBABILITY — {chartDimension.toUpperCase()}
                </div>
                <ProbabilityBars
                  classes={chartDimension === "time" ? TIME_CLASSES : SPACE_CLASSES}
                  distribution={result[chartDimension].distribution}
                  predictedClass={result[chartDimension].class}
                />
              </BenchPanel>

              <BenchPanel channel="CH.05 — SOURCE" revealDelayMs={240}>
                <div className="mono-nums" style={{ fontSize: 11, letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 14 }}>
                  DRIVING SPANS HIGHLIGHTED
                </div>
                <CodeWithSpans code={code} attribution={result.attribution} />
              </BenchPanel>

              {result.warnings.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {result.warnings.map((warning) => (
                    <div
                      key={warning}
                      className="mono-nums"
                      style={{ fontSize: 12, color: "var(--text-secondary)", borderLeft: "2px solid var(--signal)", paddingLeft: 8 }}
                    >
                      [!] {warning}
                    </div>
                  ))}
                </div>
              ) : null}
            </section>
          ) : (
            <div
              className="bench-panel"
              style={{ color: "var(--text-muted)", fontSize: 13, lineHeight: 1.6, minHeight: 200, display: "flex", alignItems: "center" }}
            >
              Paste code on the left and run an analysis — the growth curve, confidence, and driving spans will read out here.
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
