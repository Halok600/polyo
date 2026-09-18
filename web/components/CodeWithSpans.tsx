// Renders the submitted code with the attribution's driving spans
// highlighted. Spans are `[start_line, start_col, end_line, end_col]`,
// 0-indexed (tree-sitter's own convention, passed through unchanged from
// `core/ir.py`'s `IRNode.span`) -- converted to flat character offsets
// here since a span can cross multiple lines.
//
// Each span reveals as a scanner pass, not a block fade: a thin amber
// underline sweeps left-to-right, then the highlight fill lands a beat
// later, staggered in source order -- "now watch, this is what drove it."

import type { CSSProperties } from "react";

import type { AttributionItem } from "@/lib/types";

type CodeWithSpansProps = {
  code: string;
  attribution: AttributionItem[];
};

export type Range = { start: number; end: number; feature: string };

// Exported for direct unit testing (components/CodeWithSpans.test.tsx) --
// this is the one piece of real logic in an otherwise presentational
// component, and its edge cases (overlapping/adjacent/out-of-order spans)
// are worth locking down independently of rendering.
export function lineStartOffsets(code: string): number[] {
  const offsets = [0];
  for (let i = 0; i < code.length; i++) {
    if (code[i] === "\n") offsets.push(i + 1);
  }
  return offsets;
}

export function toOffset(lineStarts: number[], line: number, col: number): number {
  const lineStart = lineStarts[Math.min(line, lineStarts.length - 1)] ?? 0;
  return lineStart + col;
}

export function mergeRanges(ranges: Range[]): Range[] {
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const merged: Range[] = [];
  for (const range of sorted) {
    const last = merged[merged.length - 1];
    if (last && range.start <= last.end) {
      last.end = Math.max(last.end, range.end);
      last.feature = `${last.feature}, ${range.feature}`;
    } else {
      merged.push({ ...range });
    }
  }
  return merged;
}

export function CodeWithSpans({ code, attribution }: CodeWithSpansProps) {
  const lineStarts = lineStartOffsets(code);
  const ranges = mergeRanges(
    attribution.flatMap((item) =>
      item.spans.map(([sl, sc, el, ec]) => ({
        start: toOffset(lineStarts, sl, sc),
        end: toOffset(lineStarts, el, ec),
        feature: item.feature,
      })),
    ),
  );

  const segments: { text: string; feature?: string }[] = [];
  let cursor = 0;
  for (const range of ranges) {
    if (range.start > cursor) {
      segments.push({ text: code.slice(cursor, range.start) });
    }
    segments.push({ text: code.slice(range.start, range.end), feature: range.feature });
    cursor = range.end;
  }
  if (cursor < code.length) {
    segments.push({ text: code.slice(cursor) });
  }

  let spanIndex = 0;

  return (
    <pre
      style={{
        margin: 0,
        padding: 16,
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        overflowX: "auto",
        fontSize: 13,
        lineHeight: 1.6,
        whiteSpace: "pre",
      }}
    >
      <code>
        {segments.map((segment, i) => {
          if (!segment.feature) return <span key={i}>{segment.text}</span>;
          const delayMs = 300 + Math.min(spanIndex, 8) * 50;
          spanIndex += 1;
          const style = { "--span-delay": `${delayMs}ms` } as CSSProperties;
          return (
            <mark
              key={i}
              title={`Drives the prediction via: ${segment.feature}`}
              style={{
                ...style,
                position: "relative",
                background: "transparent",
                color: "var(--text-primary)",
                animation: "bench-fill-fade 200ms linear forwards",
                animationDelay: "var(--span-delay)",
              }}
            >
              {segment.text}
              <span
                aria-hidden
                style={{
                  position: "absolute",
                  left: 0,
                  right: 0,
                  bottom: 0,
                  height: 2,
                  background: "var(--signal)",
                  transform: "scaleX(0)",
                  transformOrigin: "left",
                  animation: "bench-underline-sweep 220ms var(--ease-sweep) forwards",
                  animationDelay: "var(--span-delay)",
                }}
              />
            </mark>
          );
        })}
      </code>
    </pre>
  );
}
