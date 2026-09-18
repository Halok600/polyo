import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeWithSpans, lineStartOffsets, mergeRanges, toOffset } from "@/components/CodeWithSpans";

describe("lineStartOffsets", () => {
  it("returns the character offset each line starts at, including the first", () => {
    expect(lineStartOffsets("ab\ncd\nef")).toEqual([0, 3, 6]);
  });

  it("has a single entry for a single-line string", () => {
    expect(lineStartOffsets("no newlines")).toEqual([0]);
  });
});

describe("toOffset", () => {
  const lineStarts = [0, 3, 6]; // "ab\ncd\nef"

  it("converts a (line, col) pair to a flat character offset", () => {
    expect(toOffset(lineStarts, 1, 1)).toBe(4); // 'd' in "cd"
  });

  it("clamps a line index past the end instead of throwing", () => {
    expect(toOffset(lineStarts, 99, 0)).toBe(6);
  });
});

describe("mergeRanges", () => {
  it("merges overlapping ranges into one, combining feature names", () => {
    const merged = mergeRanges([
      { start: 0, end: 5, feature: "LOOP" },
      { start: 3, end: 8, feature: "HASH_LOOKUP" },
    ]);
    expect(merged).toEqual([{ start: 0, end: 8, feature: "LOOP, HASH_LOOKUP" }]);
  });

  it("merges touching ranges (end === next start)", () => {
    const merged = mergeRanges([
      { start: 0, end: 5, feature: "A" },
      { start: 5, end: 10, feature: "B" },
    ]);
    expect(merged).toEqual([{ start: 0, end: 10, feature: "A, B" }]);
  });

  it("keeps non-overlapping ranges separate", () => {
    const merged = mergeRanges([
      { start: 0, end: 5, feature: "A" },
      { start: 10, end: 15, feature: "B" },
    ]);
    expect(merged).toEqual([
      { start: 0, end: 5, feature: "A" },
      { start: 10, end: 15, feature: "B" },
    ]);
  });

  it("sorts out-of-order input before merging", () => {
    const merged = mergeRanges([
      { start: 10, end: 15, feature: "B" },
      { start: 0, end: 5, feature: "A" },
    ]);
    expect(merged.map((r) => r.feature)).toEqual(["A", "B"]);
  });

  it("returns an empty array for no ranges", () => {
    expect(mergeRanges([])).toEqual([]);
  });
});

describe("CodeWithSpans", () => {
  it("highlights a single attributed span and leaves the rest as plain text", () => {
    render(
      <CodeWithSpans
        code={"def f():\n    return 1\n"}
        attribution={[{ feature: "RETURN", contribution: 1, spans: [[1, 4, 1, 12]] }]}
      />,
    );
    const mark = screen.getByTitle("Drives the prediction via: RETURN");
    expect(mark).toHaveTextContent("return 1");
  });

  it("merges two overlapping attribution spans into a single highlighted region", () => {
    render(
      <CodeWithSpans
        code={"for i in range(n):\n    pass\n"}
        attribution={[
          { feature: "LOOP", contribution: 1, spans: [[0, 0, 0, 18]] },
          { feature: "RANGE_CALL", contribution: 1, spans: [[0, 9, 0, 17]] },
        ]}
      />,
    );
    const mark = screen.getByTitle("Drives the prediction via: LOOP, RANGE_CALL");
    expect(mark).toHaveTextContent("for i in range(n):");
  });

  it("renders no <mark> elements when there is no attribution", () => {
    const { container } = render(<CodeWithSpans code="x = 1\n" attribution={[]} />);
    expect(container.querySelectorAll("mark")).toHaveLength(0);
  });
});
