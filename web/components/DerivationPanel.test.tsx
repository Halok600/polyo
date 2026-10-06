import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ClassChip } from "@/components/ClassChip";
import { CodeWithSpans } from "@/components/CodeWithSpans";
import { DerivationPanel } from "@/components/DerivationPanel";
import type { DerivationStep } from "@/lib/types";

const STEPS: DerivationStep[] = [
  { line: 3, kind: "loop", text: "loop runs O(n) times; the whole loop costs O(n * m)" },
  { line: 4, kind: "loop", text: "loop runs O(m) times; the whole loop costs O(m)" },
  { line: 0, kind: "total", text: "time: O(n * m)" },
  { line: 0, kind: "total", text: "space: O(1)" },
];

describe("DerivationPanel", () => {
  it("renders nothing for a model answer with no derivation", () => {
    const { container } = render(
      <DerivationPanel derivation={undefined} assumptions={undefined} activeLine={null} onActiveLineChange={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("lists the steps, then the totals, and no assumptions section when there are none", () => {
    render(<DerivationPanel derivation={STEPS} assumptions={[]} entry="f" activeLine={null} onActiveLineChange={() => {}} />);
    expect(screen.getByLabelText("Derivation steps").querySelectorAll("li")).toHaveLength(2);
    expect(screen.getByLabelText("Totals")).toHaveTextContent("time: O(n * m)");
    expect(screen.queryByLabelText("Assumptions")).toBeNull();
    expect(screen.getByText(/HOW THE COST WAS DERIVED — f/)).toBeInTheDocument();
  });

  it("shows each assumption with its line and says the answer may over-estimate", () => {
    render(
      <DerivationPanel
        derivation={STEPS}
        assumptions={[{ line: 7, reason: "loop bound assumed" }, { line: 0, reason: "unknown library call" }]}
        activeLine={null}
        onActiveLineChange={() => {}}
      />,
    );
    const list = screen.getByLabelText("Assumptions");
    expect(list).toHaveTextContent("L7");
    expect(list).toHaveTextContent("loop bound assumed");
    expect(list).toHaveTextContent("unknown library call");
    expect(screen.getByText(/may over-estimate/)).toBeInTheDocument();
  });

  it("reports the line a step points at when hovered or focused, and clears it on leave", () => {
    const onActiveLineChange = vi.fn();
    render(<DerivationPanel derivation={STEPS} assumptions={[]} activeLine={null} onActiveLineChange={onActiveLineChange} />);
    const row = screen.getByText(/loop runs O\(m\) times/).closest("li") as HTMLElement;
    fireEvent.mouseEnter(row);
    expect(onActiveLineChange).toHaveBeenLastCalledWith(4);
    fireEvent.mouseLeave(row);
    expect(onActiveLineChange).toHaveBeenLastCalledWith(null);
    fireEvent.focus(row);
    expect(onActiveLineChange).toHaveBeenLastCalledWith(4);
  });

  it("marks the active step", () => {
    render(<DerivationPanel derivation={STEPS} assumptions={[]} activeLine={3} onActiveLineChange={() => {}} />);
    const active = screen.getByText(/loop runs O\(n\) times/).closest("li") as HTMLElement;
    expect(active).toHaveClass("is-active");
    expect(screen.getByText(/loop runs O\(m\) times/).closest("li")).not.toHaveClass("is-active");
  });
});

describe("CodeWithSpans gutter", () => {
  const CODE = "def f(a, b):\n    t = 0\n    for x in a:\n        for y in b:\n            t += x * y\n    return t\n";

  it("has no gutter for a model answer: the plain code view is unchanged", () => {
    const { container } = render(<CodeWithSpans code={CODE} attribution={[]} />);
    expect(container.querySelector(".code-gutter")).toBeNull();
  });

  it("numbers every line, not just the marked ones, and marks only the lines with a step", () => {
    const { container } = render(<CodeWithSpans code={CODE} attribution={[]} derivation={STEPS} />);
    expect(container.querySelectorAll(".code-gutter-row")).toHaveLength(6); // trailing newline is not a 7th line
    expect([...container.querySelectorAll(".code-gutter-row.has-note")].map((r) => r.querySelector(".code-gutter-number")?.textContent)).toEqual(["3", "4"]);
  });

  it("puts the step text on the marked line, readable by hover and by assistive tech", () => {
    render(<CodeWithSpans code={CODE} attribution={[]} derivation={STEPS} />);
    const note = screen.getByRole("note", { name: /Line 3: loop runs O\(n\) times/ });
    expect(note).toHaveAttribute("title", "loop runs O(n) times; the whole loop costs O(n * m)");
  });

  it("marks an assumption line with a ? and its reason", () => {
    render(<CodeWithSpans code={CODE} attribution={[]} assumptions={[{ line: 5, reason: "loop bound assumed" }]} />);
    const note = screen.getByRole("note", { name: /Line 5: Assumed: loop bound assumed/ });
    expect(note).toHaveTextContent("?");
  });

  it("draws a highlight band only for an active line, at that line's row", () => {
    const { rerender } = render(<CodeWithSpans code={CODE} attribution={[]} derivation={STEPS} activeLine={null} />);
    expect(screen.queryByTestId("active-line-band")).toBeNull();
    rerender(<CodeWithSpans code={CODE} attribution={[]} derivation={STEPS} activeLine={4} />);
    expect(screen.getByTestId("active-line-band")).toHaveStyle({ top: `${16 + 3 * 13 * 1.6}px` });
  });

  it("tells the parent which line a gutter marker points at", () => {
    const onActiveLineChange = vi.fn();
    render(<CodeWithSpans code={CODE} attribution={[]} derivation={STEPS} onActiveLineChange={onActiveLineChange} />);
    fireEvent.mouseEnter(screen.getByRole("note", { name: /Line 4/ }));
    expect(onActiveLineChange).toHaveBeenLastCalledWith(4);
  });
});

describe("ClassChip v2", () => {
  // jsdom has no matchMedia. Reduced motion on makes the confidence readout show its target value
  // at once instead of counting up over animation frames.
  beforeEach(() => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: true,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    }));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const base = {
    channel: "CH.01 — TIME",
    label: "Time",
    predictedClass: "O(n^2)",
    confidence: 0.92,
    conformalSet: ["O(n^2)"],
    conformalCoverage: 0.92,
    abstain: false,
  };

  it("leads with the exact expression and keeps the class as a secondary chip", () => {
    render(<ClassChip {...base} engine="symbolic" certainty="certain" expression="O(n * m)" extendedClass="O(n^2)" projectionLossy />);
    expect(screen.getByTestId("time-headline")).toHaveTextContent("O(n * m)");
    expect(screen.getByText(/CLASS O\(n\^2\)/)).toBeInTheDocument();
    expect(screen.getByText("CERTAIN")).toBeInTheDocument();
    expect(screen.getByText(/measured on independent programs/)).toBeInTheDocument();
  });

  it("does not repeat the headline as a CLASS chip when the expression is already exactly the class", () => {
    render(<ClassChip {...base} engine="symbolic" certainty="certain" expression="O(n^2)" extendedClass="O(n^2)" />);
    expect(screen.queryByText(/^CLASS /)).toBeNull();
  });

  it("states that the confidence is the measured accuracy of that certainty level", () => {
    const { rerender } = render(<ClassChip {...base} engine="symbolic" certainty="certain" expression="O(n)" predictedClass="O(n)" />);
    expect(screen.getByText(/accuracy of certain answers, measured on independent programs/)).toBeInTheDocument();
    rerender(<ClassChip {...base} engine="symbolic" certainty="assumed" expression="O(n)" predictedClass="O(n)" />);
    expect(screen.getByText(/accuracy of assumed answers/)).toBeInTheDocument();
  });

  it("shows the exact class only when it differs from the legacy class", () => {
    const { rerender } = render(
      <ClassChip {...base} engine="symbolic" certainty="certain" expression="O(n^2 log n)" predictedClass="O(n^3)" extendedClass="O(n^2 log n)" projectionLossy />,
    );
    expect(screen.getByText(/EXACT CLASS O\(n\^2 log n\)/)).toBeInTheDocument();
    rerender(<ClassChip {...base} engine="symbolic" certainty="certain" expression="O(n^2)" extendedClass="O(n^2)" />);
    expect(screen.queryByText(/EXACT CLASS/)).toBeNull();
  });

  it("labels an assumed answer ASSUMED, never CERTAIN", () => {
    render(<ClassChip {...base} engine="symbolic" certainty="assumed" expression="O(n^2)" />);
    expect(screen.getByText("ASSUMED")).toBeInTheDocument();
    expect(screen.queryByText("CERTAIN")).toBeNull();
  });

  it("keeps a model answer exactly as before, labelled MODEL, class as the headline", () => {
    render(<ClassChip {...base} engine="ml_fallback" certainty={null} expression={null} />);
    expect(screen.getByTestId("time-headline")).toHaveTextContent("O(n^2)");
    expect(screen.getByText("MODEL")).toBeInTheDocument();
    expect(screen.queryByText(/measured on independent programs/)).toBeNull();
    expect(screen.queryByText(/^CLASS /)).toBeNull();
  });

  it("renders a legacy response (no v2 fields at all) as a model answer without crashing", () => {
    render(<ClassChip {...base} />);
    expect(screen.getByTestId("time-headline")).toHaveTextContent("O(n^2)");
    expect(screen.getByText("MODEL")).toBeInTheDocument();
  });
});
