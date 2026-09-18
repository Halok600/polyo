// Renders the submitted code with the attribution's driving spans
// highlighted (plan §11, item 4). Spans are `[start_line, start_col,
// end_line, end_col]`, 0-indexed (tree-sitter's own convention, passed
// through unchanged from `core/ir.py`'s `IRNode.span`) -- converted to
// flat character offsets here since a span can cross multiple lines.

import type { AttributionItem } from "@/lib/types";

type CodeWithSpansProps = {
  code: string;
  attribution: AttributionItem[];
};

type Range = { start: number; end: number; feature: string };

function lineStartOffsets(code: string): number[] {
  const offsets = [0];
  for (let i = 0; i < code.length; i++) {
    if (code[i] === "\n") offsets.push(i + 1);
  }
  return offsets;
}

function toOffset(lineStarts: number[], line: number, col: number): number {
  const lineStart = lineStarts[Math.min(line, lineStarts.length - 1)] ?? 0;
  return lineStart + col;
}

function mergeRanges(ranges: Range[]): Range[] {
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

  return (
    <pre
      style={{
        margin: 0,
        padding: 16,
        borderRadius: 8,
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        overflowX: "auto",
        fontSize: 13,
        lineHeight: 1.6,
        whiteSpace: "pre",
      }}
    >
      <code>
        {segments.map((segment, i) =>
          segment.feature ? (
            <mark
              key={i}
              title={`Drives the prediction via: ${segment.feature}`}
              style={{
                background: "var(--series-predicted-fill)",
                color: "var(--text-primary)",
                borderRadius: 3,
              }}
            >
              {segment.text}
            </mark>
          ) : (
            <span key={i}>{segment.text}</span>
          ),
        )}
      </code>
    </pre>
  );
}
