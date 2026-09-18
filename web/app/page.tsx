"use client";

import { useEffect, useState } from "react";

import { ClassChip } from "@/components/ClassChip";
import { CodeWithSpans } from "@/components/CodeWithSpans";
import { GrowthChart } from "@/components/GrowthChart";
import { LanguageSelector } from "@/components/LanguageSelector";
import { ProbabilityBars } from "@/components/ProbabilityBars";
import { ApiError, fetchLanguages, predict } from "@/lib/api";
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
// curve series has one -- plan §11's "what faster looks like".
function betterNeighbour(seriesKeys: string[], predictedClass: string, allRanked: string[]) {
  const predictedRank = allRanked.indexOf(predictedClass);
  if (predictedRank <= 0) return undefined;
  const better = allRanked[predictedRank - 1];
  return seriesKeys.includes(better) ? better : undefined;
}

const TIME_CLASSES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)"];
const SPACE_CLASSES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)"];

export default function Home() {
  const [languages, setLanguages] = useState<LanguageOption[]>([]);
  const [language, setLanguage] = useState("auto");
  const [code, setCode] = useState(EXAMPLE_CODE);
  const [chartDimension, setChartDimension] = useState<"time" | "space">("time");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  useEffect(() => {
    fetchLanguages()
      .then(setLanguages)
      .catch(() => setLanguages([]));
  }, []);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await predict(language, code);
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "something went wrong");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: "32px 16px 64px" }}>
      <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 4 }}>PolyO</h1>
      <p style={{ color: "var(--text-secondary)", fontSize: 14, marginBottom: 24 }}>
        Static, multi-language time &amp; space complexity prediction. No LLM, no code execution.
      </p>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <LanguageSelector languages={languages} value={language} onChange={setLanguage} />
        <textarea
          value={code}
          onChange={(event) => setCode(event.target.value)}
          rows={14}
          spellCheck={false}
          style={{
            width: "100%",
            padding: 12,
            borderRadius: 8,
            border: "1px solid var(--border)",
            background: "var(--surface-1)",
            color: "var(--text-primary)",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            fontSize: 13,
            resize: "vertical",
          }}
        />
        <button
          type="submit"
          disabled={loading || code.trim().length === 0}
          style={{
            alignSelf: "flex-start",
            padding: "10px 20px",
            borderRadius: 8,
            border: "none",
            background: "var(--series-predicted)",
            color: "#fff",
            fontSize: 14,
            fontWeight: 600,
            cursor: loading ? "default" : "pointer",
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? "Analyzing…" : "Predict complexity"}
        </button>
      </form>

      {error ? (
        <p role="alert" style={{ marginTop: 20, color: "#d03b3b" }}>
          {error}
        </p>
      ) : null}

      {result ? (
        <section style={{ marginTop: 32, display: "flex", flexDirection: "column", gap: 24 }}>
          <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
            Detected language: <strong>{result.language_detected}</strong>
          </p>

          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <ClassChip label="Time" predictedClass={result.time.class} confidence={result.time.confidence} />
            <ClassChip
              label="Space"
              predictedClass={result.space.class}
              confidence={result.space.confidence}
              footnote="Auxiliary space, including recursion stack."
            />
          </div>

          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
              {(["time", "space"] as const).map((dim) => (
                <button
                  key={dim}
                  type="button"
                  onClick={() => setChartDimension(dim)}
                  style={{
                    padding: "6px 14px",
                    borderRadius: 999,
                    border: "1px solid var(--border)",
                    background: chartDimension === dim ? "var(--series-predicted)" : "var(--surface-1)",
                    color: chartDimension === dim ? "#fff" : "var(--text-secondary)",
                    fontSize: 13,
                    cursor: "pointer",
                  }}
                >
                  {dim === "time" ? "Time" : "Space"}
                </button>
              ))}
            </div>
            <GrowthChart
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
          </div>

          <div>
            <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 8 }}>
              Class probability distribution ({chartDimension})
            </div>
            <ProbabilityBars
              classes={chartDimension === "time" ? TIME_CLASSES : SPACE_CLASSES}
              distribution={result[chartDimension].distribution}
              predictedClass={result[chartDimension].class}
            />
          </div>

          <div>
            <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 8 }}>
              Code with driving spans highlighted
            </div>
            <CodeWithSpans code={code} attribution={result.attribution} />
          </div>

          {result.warnings.length > 0 ? (
            <ul style={{ fontSize: 13, color: "var(--text-muted)", paddingLeft: 18 }}>
              {result.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
