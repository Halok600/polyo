// Renders the submitted code with the attribution's driving spans
// highlighted. Spans are `[start_line, start_col, end_line, end_col]`,
// 0-indexed (tree-sitter's own convention, passed through unchanged from
// `core/ir.py`'s `IRNode.span`) -- converted to flat character offsets
// here since a span can cross multiple lines.
//
// Each span reveals as a scanner pass, not a block fade: a thin amber
// underline sweeps left-to-right, then the highlight fill lands a beat
// later, staggered in source order -- "now watch, this is what drove it."
//
// v2: when the static engine answered, a line-number gutter marks every
// line the derivation talks about (hover or focus shows the step), and the
// step list under the code lights the same line from the other side.

import type { CSSProperties } from "react";

import { markersByLine } from "@/lib/derivation";
import type { AssumptionItem, AttributionItem, DerivationStep } from "@/lib/types";

type CodeWithSpansProps = {
  code: string;
  attribution: AttributionItem[];
  // The engine's derivation, shown as markers in a line-number gutter. Absent for a model answer
  // (and for a response from before the engine), which keeps the plain code view.
  derivation?: DerivationStep[];
  assumptions?: AssumptionItem[];
  // 1-based line the reader is pointing at (from the step list or a gutter marker).
  activeLine?: number | null;
  onActiveLineChange?: (line: number | null) => void;
};

export type Range = { start: number; end: number; feature: string };

// The code is 13px at line-height 1.6; the gutter rows and the highlight band use the same numbers
// so they stay on the text's baseline grid.
const FONT_SIZE_PX = 13;
const LINE_HEIGHT = 1.6;
const LINE_HEIGHT_PX = FONT_SIZE_PX * LINE_HEIGHT;
const FRAME_PADDING_PX = 16;

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

export function CodeWithSpans({
  code,
  attribution,
  derivation,
  assumptions,
  activeLine = null,
  onActiveLineChange,
}: CodeWithSpansProps) {
  const lineStarts = lineStartOffsets(code);
  const markers = markersByLine(derivation, assumptions);
  const hasGutter = markers.size > 0;
  // A trailing newline ends the last line; it does not start another one.
  const lineCount = code.endsWith("\n") ? lineStarts.length - 1 : lineStarts.length;

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
    <div
      data-testid="code-frame"
      style={{
        position: "relative",
        display: "flex",
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
      }}
    >
      {hasGutter && activeLine ? (
        <div
          aria-hidden
          data-testid="active-line-band"
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: FRAME_PADDING_PX + (activeLine - 1) * LINE_HEIGHT_PX,
            height: LINE_HEIGHT_PX,
            background: "var(--signal-fill)",
            pointerEvents: "none",
          }}
        />
      ) : null}

      {hasGutter ? (
        <div
          className="mono-nums code-gutter"
          style={{
            position: "relative",
            zIndex: 1,
            padding: `${FRAME_PADDING_PX}px 0`,
            fontSize: FONT_SIZE_PX,
            lineHeight: LINE_HEIGHT,
          }}
        >
          {Array.from({ length: lineCount }, (_, i) => {
            const line = i + 1;
            const marker = markers.get(line);
            if (!marker) {
              return (
                <div key={line} className="code-gutter-row">
                  <span className="code-gutter-number">{line}</span>
                  <span className="code-gutter-mark" aria-hidden />
                </div>
              );
            }
            const notes = [
              ...marker.steps.map((step) => step.text),
              ...marker.assumptions.map((a) => `Assumed: ${a.reason}`),
            ];
            return (
              <div
                key={line}
                className={`code-gutter-row has-note${activeLine === line ? " is-active" : ""}`}
                tabIndex={0}
                role="note"
                aria-label={`Line ${line}: ${notes.join("; ")}`}
                title={notes.join("\n")}
                onMouseEnter={() => onActiveLineChange?.(line)}
                onMouseLeave={() => onActiveLineChange?.(null)}
                onFocus={() => onActiveLineChange?.(line)}
                onBlur={() => onActiveLineChange?.(null)}
              >
                <span className="code-gutter-number">{line}</span>
                <span className="code-gutter-mark" aria-hidden>
                  {marker.assumptions.length > 0 ? "?" : "●"}
                </span>
              </div>
            );
          })}
        </div>
      ) : null}

      <pre
        style={{
          margin: 0,
          padding: FRAME_PADDING_PX,
          flex: 1,
          minWidth: 0,
          position: "relative",
          zIndex: 1,
          overflowX: "auto",
          fontSize: FONT_SIZE_PX,
          lineHeight: LINE_HEIGHT,
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
    </div>
  );
}
