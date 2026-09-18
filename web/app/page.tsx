"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { BenchPanel } from "@/components/BenchPanel";
import { ClassChip } from "@/components/ClassChip";
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
  const [wakingUp, setWakingUp] = useState(false);
  const reducedMotion = usePrefersReducedMotion();
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
    setLoading(true);
    setWakingUp(false);
    setStatus("sampling");
    setError(null);
    try {
      const response = await predict(language, code, controller.signal);
      setResult(response);
      setRunId((id) => id + 1);
      setStatus("done");
      window.setTimeout(() => setStatus("ready"), 1400);
    } catch (err) {
      if (controller.signal.aborted) return; // superseded by a newer request
      setError(err instanceof ApiError ? err.message : "something went wrong");
      setResult(null);
      setStatus("ready");
    } finally {
      if (abortControllerRef.current === controller) {
        setLoading(false);
        setWakingUp(false); // otherwise a cold-start run leaves this caption stuck on afterward
      }
    }
  }, [language, code]);

  // Reads current values through a ref instead of listing them as effect
  // deps -- `code` changes on every keystroke, and re-subscribing a global
  // window listener that often is pure churn for no behavioural gain.
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
          <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
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
          <BenchPanel channel="CH.00 — INPUT">
            <CodeEditor value={code} onChange={setCode} language={language} />
          </BenchPanel>
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

              <BenchPanel channel="CH.04 — DISTRIBUTION" revealDelayMs={180}>
                <div className="mono-nums field-label" style={{ marginBottom: 14 }}>
                  CLASS PROBABILITY — {chartDimension.toUpperCase()}
                </div>
                <ProbabilityBars
                  classes={chartDimension === "time" ? TIME_CLASSES : SPACE_CLASSES}
                  distribution={result[chartDimension].distribution}
                  predictedClass={result[chartDimension].class}
                />
              </BenchPanel>

              <BenchPanel channel="CH.05 — SOURCE" revealDelayMs={240}>
                <div className="mono-nums field-label" style={{ marginBottom: 14 }}>
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
