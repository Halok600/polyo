"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { BenchPanel } from "@/components/BenchPanel";
import { ClassChip } from "@/components/ClassChip";
import { CodeWithSpans } from "@/components/CodeWithSpans";
import { DerivationPanel } from "@/components/DerivationPanel";
import { GrowthChart } from "@/components/GrowthChart";
import { IdlePlayground } from "@/components/IdlePlayground";
import { Landing } from "@/components/Landing";
import { LanguageSelector } from "@/components/LanguageSelector";
import { ProbabilityBars } from "@/components/ProbabilityBars";
import { StatusReadout } from "@/components/StatusReadout";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Toolbar } from "@/components/Toolbar";
import { ApiError, fetchLanguages, predict } from "@/lib/api";
import { COLLAPSE_MS, remainingScanMs, wait } from "@/lib/handoff";
import { usePrefersReducedMotion, withViewTransition } from "@/lib/motion";
import { useTabStatus } from "@/lib/tabStatus";
import { useMatchBottom } from "@/lib/useMatchBottom";
import type { LanguageOption, PredictResponse } from "@/lib/types";

// CodeMirror (core + language packages) is a ~650KB chunk on its own --
// deferred out of the landing page's bundle entirely via next/dynamic, not
// loaded until the tool scene actually mounts. ssr:false is safe (and
// required): the editor only ever touches the DOM inside its own effects,
// never during render, but its whole import chain still shouldn't be part
// of what the static prerender needs to produce first paint.
const CodeEditor = dynamic(() => import("@/components/CodeEditor").then((mod) => mod.CodeEditor), {
  ssr: false,
  loading: () => <div className="code-editor-frame" style={{ height: 340 }} />,
});

const EXAMPLE_CODE = `def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[n] = i
    return []
`;

// One demo per served language -- the pitch is "multi-language," so the
// demo should let a visitor actually see that instead of only ever
// running the one hardcoded Python snippet.
const EXAMPLES: { language: string; label: string; code: string }[] = [
  { language: "python", label: "Python — hash lookup", code: EXAMPLE_CODE },
  {
    language: "go",
    label: "Go — merge sort",
    code: `package main

func mergeSort(nums []int) []int {
	if len(nums) <= 1 {
		return nums
	}
	mid := len(nums) / 2
	left := mergeSort(nums[:mid])
	right := mergeSort(nums[mid:])
	result := make([]int, 0, len(nums))
	i, j := 0, 0
	for i < len(left) && j < len(right) {
		if left[i] <= right[j] {
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}
	result = append(result, left[i:]...)
	result = append(result, right[j:]...)
	return result
}
`,
  },
  {
    language: "java",
    label: "Java — nested loop",
    code: `public class Solution {
    public static int countPairs(int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == 0) {
                    count++;
                }
            }
        }
        return count;
    }
}
`,
  },
  {
    language: "cpp",
    label: "C++ — binary search",
    code: `#include <vector>
using namespace std;

int binarySearch(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
`,
  },
  {
    language: "javascript",
    label: "JavaScript — memoized recursion",
    code: `function fibMemo(n, memo = new Map()) {
  if (n <= 1) return n;
  if (memo.has(n)) return memo.get(n);
  const result = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
  memo.set(n, result);
  return result;
}
`,
  },
  {
    language: "c",
    label: "C — linear search",
    code: `#include <stddef.h>

int linear_search(int *nums, size_t n, int target) {
    for (size_t i = 0; i < n; i++) {
        if (nums[i] == target) return (int)i;
    }
    return -1;
}
`,
  },
];

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
  // The idle playground plays a hand-off to the first result: it scans while the request is in
  // flight, then collapses to the predicted class before the results replace it (lib/handoff.ts).
  const [handoff, setHandoff] = useState<{ phase: "scanning" | "collapsing"; answer: string | null } | null>(
    null,
  );
  const resultRef = useRef<PredictResponse | null>(null);
  useEffect(() => {
    resultRef.current = result;
  });
  // 1-based source line the reader is pointing at, shared by the code gutter and the step list.
  const [activeLine, setActiveLine] = useState<number | null>(null);
  const [wakingUp, setWakingUp] = useState(false);
  const reducedMotion = usePrefersReducedMotion();
  // The code input (left) and the growth chart (right) end on the same line: see useMatchBottom.
  const inputPanelRef = useRef<HTMLDivElement>(null);
  const growthPanelRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Fetched unconditionally, before `entered` flips -- the language list
    // is already warm by the time someone clicks through the landing page.
    fetchLanguages()
      .then(setLanguages)
      .catch(() => setLanguages([]));
  }, []);

  // Render's free tier sleeps after 15 minutes idle -- the first request
  // after that can take ~40s to wake the container. A generic spinner past
  // a few seconds reads as broken, not slow, so the copy changes once it's
  // clearly not a normal-latency request.
  useEffect(() => {
    if (!loading) return;
    const timeout = window.setTimeout(() => setWakingUp(true), 3000);
    return () => window.clearTimeout(timeout);
  }, [loading]);

  // Cancels any still-in-flight request before starting a new one -- without
  // this, submitting twice in quick succession (a fast double-click, or
  // changing language and resubmitting before the first response lands)
  // races two responses against each other and whichever resolves last wins,
  // even if it's stale.
  const runAnalysis = useCallback(async () => {
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const startedAt = performance.now();
    // the playground is on screen only until the first result; the hand-off is skipped for reduced motion
    const handOff = resultRef.current === null && !reducedMotion;
    setLoading(true);
    setWakingUp(false);
    setStatus("sampling");
    setError(null);
    if (handOff) setHandoff({ phase: "scanning", answer: null });
    try {
      const response = await predict(language, code, controller.signal);
      if (handOff) {
        // a fast answer still lets the scan be seen; a slow one (cold start) is never held back further
        await wait(remainingScanMs(performance.now() - startedAt), controller.signal);
        setHandoff({ phase: "collapsing", answer: response.time.class });
        await wait(COLLAPSE_MS, controller.signal);
      }
      setResult(response);
      setHandoff(null);
      setActiveLine(null);
      setRunId((id) => id + 1);
      setStatus("done");
      window.setTimeout(() => setStatus("ready"), 1400);
    } catch (err) {
      if (controller.signal.aborted) return; // superseded by a newer request
      setHandoff(null);
      setError(err instanceof ApiError ? err.message : "something went wrong");
      setResult(null);
      setStatus("ready");
    } finally {
      if (abortControllerRef.current === controller) {
        setLoading(false);
        setWakingUp(false); // otherwise a cold-start run leaves this caption stuck on afterward
      }
    }
  }, [language, code, reducedMotion]);

  // Reads current values through a ref instead of listing them as effect
  // deps -- `code` changes on every keystroke, and re-subscribing a global
  // window listener that often is pure churn for no behavioural gain.
  const growthMinHeight = useMatchBottom(inputPanelRef, growthPanelRef, entered && result !== null, runId);
  // the tab says what is happening (and the answer, once there is one) and its icon spins meanwhile
  useTabStatus({ analysing: loading, answer: result ? (result.time.expression ?? result.time.class) : null });

  const shortcutStateRef = useRef({ entered, loading, code, runAnalysis });
  useEffect(() => {
    shortcutStateRef.current = { entered, loading, code, runAnalysis };
  });

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.repeat || !(event.metaKey || event.ctrlKey) || event.key !== "Enter") return;
      const { entered, loading, code, runAnalysis } = shortcutStateRef.current;
      if (!entered) return;
      event.preventDefault();
      if (!loading && code.trim().length > 0) void runAnalysis();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  function handleEnter() {
    withViewTransition(() => flushSync(() => setEntered(true)), reducedMotion, "vt-scene");
  }

  if (!entered) {
    return <Landing onEnter={handleEnter} />;
  }

  // The probability bars describe the learned model's distribution; a static-analysis answer has
  // none, so they appear only when the model answered the dimension being charted (and for a
  // response from before the engine, which has no `engine` field).
  const showDistribution = result !== null && result[chartDimension].engine !== "symbolic";
  const hasDerivation = (result?.derivation?.length ?? 0) > 0 || (result?.assumptions?.length ?? 0) > 0;

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    void runAnalysis();
  }

  function loadExample(example: (typeof EXAMPLES)[number]) {
    setLanguage(example.language);
    setCode(example.code);
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
          <div className="mono-nums field-label bench-colhead">
            <span>SETUP</span>
          </div>
          <p className="bench-intro" style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
            Static, multi-language time &amp; space complexity prediction. No LLM, no code execution.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span className="mono-nums field-label">EXAMPLES</span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {EXAMPLES.map((example) => (
                <button
                  key={example.language}
                  type="button"
                  disabled={loading}
                  onClick={() => loadExample(example)}
                  className="mono-nums example-chip"
                >
                  {example.label}
                </button>
              ))}
            </div>
          </div>
          <LanguageSelector languages={languages} value={language} onChange={setLanguage} />
          <div ref={inputPanelRef}>
            <BenchPanel channel="CH.00 — INPUT">
              <CodeEditor value={code} onChange={setCode} language={language} />
            </BenchPanel>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              type="submit"
              disabled={loading || code.trim().length === 0}
              className="mono-nums vt-run-cta cta-button submit-cta"
            >
              {loading ? (wakingUp ? "WAKING SERVER…" : "SAMPLING…") : "RUN ANALYSIS"}
            </button>
            <span className="mono-nums" style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {wakingUp ? "free-tier host was asleep — first request can take ~40s" : "⌘/Ctrl + Enter"}
            </span>
          </div>
          {error ? (
            <p role="alert" className="mono-nums" style={{ fontSize: 12, color: "var(--status-error)", borderLeft: "2px solid var(--status-error)", paddingLeft: 8 }}>
              [!] {error}
            </p>
          ) : null}
        </form>

        <div className="bench-readout">
          <div className="mono-nums field-label bench-colhead">
            <span>READOUT</span>
            {result ? (
              <span>
                LANG <span style={{ color: "var(--text-primary)" }}>{result.language_detected}</span>
              </span>
            ) : null}
          </div>
          {result ? (
            <section key={runId} style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              <div
                className="bench-enter"
                style={{ display: "flex", gap: 16, flexWrap: "wrap", "--enter-delay": "0ms" } as React.CSSProperties}
              >
                <ClassChip
                  channel="CH.01 — TIME"
                  label="Time"
                  predictedClass={result.time.class}
                  confidence={result.time.confidence}
                  conformalSet={result.time.conformal_set}
                  conformalCoverage={result.time.conformal_coverage}
                  abstain={result.time.abstain}
                  engine={result.time.engine}
                  certainty={result.time.certainty}
                  expression={result.time.expression}
                  extendedClass={result.time.extended_class}
                  projectionLossy={result.time.projection_lossy}
                  revealDelayMs={0}
                />
                <ClassChip
                  channel="CH.02 — SPACE"
                  label="Space"
                  predictedClass={result.space.class}
                  confidence={result.space.confidence}
                  conformalSet={result.space.conformal_set}
                  conformalCoverage={result.space.conformal_coverage}
                  abstain={result.space.abstain}
                  engine={result.space.engine}
                  certainty={result.space.certainty}
                  expression={result.space.expression}
                  extendedClass={result.space.extended_class}
                  projectionLossy={result.space.projection_lossy}
                  footnote="Auxiliary space, including recursion stack."
                  revealDelayMs={60}
                />
              </div>

              <div ref={growthPanelRef} className="bench-match" style={{ minHeight: growthMinHeight }}>
              <BenchPanel
                channel="CH.03 — GROWTH"
                revealDelayMs={120}
                className="bench-enter"
                style={{ flex: 1, "--enter-delay": "140ms" } as React.CSSProperties}
              >
                <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
                  {(["time", "space"] as const).map((dim) => (
                    <button
                      key={dim}
                      type="button"
                      onClick={() => setChartDimension(dim)}
                      className={`mono-nums dim-tab${chartDimension === dim ? " is-active" : ""}`}
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
              </div>

              {showDistribution ? (
                <BenchPanel
                  channel="CH.04 — DISTRIBUTION"
                  revealDelayMs={180}
                  className="bench-enter"
                  style={{ "--enter-delay": "240ms" } as React.CSSProperties}
                >
                  <div className="mono-nums field-label" style={{ marginBottom: 14 }}>
                    MODEL CLASS PROBABILITY — {chartDimension.toUpperCase()}
                  </div>
                  <ProbabilityBars
                    classes={chartDimension === "time" ? TIME_CLASSES : SPACE_CLASSES}
                    distribution={result[chartDimension].distribution}
                    predictedClass={result[chartDimension].class}
                  />
                </BenchPanel>
              ) : null}

              <BenchPanel
                channel={showDistribution ? "CH.05 — SOURCE" : "CH.04 — SOURCE"}
                revealDelayMs={240}
                className="bench-enter"
                style={{ "--enter-delay": "320ms" } as React.CSSProperties}
              >
                <div className="mono-nums field-label" style={{ marginBottom: 14 }}>
                  {hasDerivation ? "MARKED LINES DROVE THE COST — HOVER ONE" : "DRIVING SPANS HIGHLIGHTED"}
                </div>
                <CodeWithSpans
                  code={code}
                  attribution={result.attribution}
                  derivation={result.derivation}
                  assumptions={result.assumptions}
                  activeLine={activeLine}
                  onActiveLineChange={setActiveLine}
                />
                <DerivationPanel
                  derivation={result.derivation}
                  assumptions={result.assumptions}
                  entry={result.entry}
                  activeLine={activeLine}
                  onActiveLineChange={setActiveLine}
                />
              </BenchPanel>

              {result.warnings.length > 0 ? (
                <div
                  className="bench-enter"
                  style={{ display: "flex", flexDirection: "column", gap: 6, "--enter-delay": "400ms" } as React.CSSProperties}
                >
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
            <IdlePlayground phase={handoff?.phase ?? "idle"} answer={handoff?.answer ?? null} />
          )}
        </div>
      </div>
    </main>
  );
}
