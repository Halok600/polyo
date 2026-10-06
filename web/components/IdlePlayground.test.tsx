import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { IdlePlayground } from "@/components/IdlePlayground";

function stubMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: reduced,
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

const lane = (label: string) => screen.getByRole("button", { name: new RegExp(`^${label.replace(/[()^]/g, "\\$&")},`) });
const nReadout = () => screen.getByTestId("playground-n").textContent;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("IdlePlayground with reduced motion (nothing moves by itself)", () => {
  beforeEach(() => stubMotion(true));

  it("shows the seven classes and a starting n", () => {
    render(<IdlePlayground />);
    expect(screen.getAllByRole("button").filter((b) => b.classList.contains("pg-lane"))).toHaveLength(7);
    expect(nReadout()).toBe("n = 24");
  });

  it("does not offer autoplay when the visitor asked for no motion", () => {
    render(<IdlePlayground />);
    expect(screen.getByRole("button", { name: /AUTO/ })).toBeDisabled();
  });

  it("moves n with the slider, on a log scale (position 500 is n = 100)", () => {
    render(<IdlePlayground />);
    fireEvent.change(screen.getByRole("slider", { name: "Input size n" }), { target: { value: "500" } });
    expect(nReadout()).toBe("n = 100");
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuetext", "n = 100");
  });

  it("marks each class FITS, SLOW or NEVER for the chosen n", () => {
    render(<IdlePlayground />);
    fireEvent.change(screen.getByRole("slider"), { target: { value: "500" } }); // n = 100
    expect(within(lane("O(2^n)")).getByText("NEVER")).toBeInTheDocument();
    expect(within(lane("O(n^3)")).getByText("FITS")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("slider"), { target: { value: "1000" } }); // n = 10,000
    expect(within(lane("O(n^3)")).getByText("SLOW")).toBeInTheDocument();
  });

  it("scales a bar to the share of the axis its operation count reaches", () => {
    render(<IdlePlayground />);
    fireEvent.change(screen.getByRole("slider"), { target: { value: "750" } }); // n = 1000
    const bar = (label: string) => lane(label).querySelector<HTMLElement>(".pg-bar")!;
    expect(bar("O(1)").style.transform).toBe("scaleX(0)");
    expect(bar("O(n^2)").style.transform).toBe(`scaleX(${6 / 18})`);
    // 2^1000 is far off the axis: clamped to the full track, not overflowing it
    expect(bar("O(2^n)").style.transform).toBe("scaleX(1)");
  });

  it("describes a lane in the footer while it is hovered, and goes back to the summary", () => {
    render(<IdlePlayground />);
    const footer = screen.getByTestId("playground-footer");
    const idle = footer.textContent;
    fireEvent.mouseEnter(lane("O(2^n)"));
    expect(footer).toHaveTextContent(/O\(2\^n\).*trying every subset.*Passes 1 second at n = 30/);
    fireEvent.mouseLeave(lane("O(2^n)"));
    expect(footer.textContent).toBe(idle);
  });

  it("says plainly when a class never leaves the budget in range", () => {
    render(<IdlePlayground />);
    fireEvent.focus(lane("O(n log n)"));
    expect(screen.getByTestId("playground-footer")).toHaveTextContent(/Stays under 1 second all the way to n = 10,000/);
  });

  it("jumps to the n where a class passes one second when its lane is clicked", () => {
    render(<IdlePlayground />);
    fireEvent.click(lane("O(2^n)"));
    expect(nReadout()).toBe("n = 30");
    fireEvent.click(lane("O(n^3)"));
    expect(nReadout()).toBe("n = 1,001");
  });

  it("does nothing when a class has no breaking point", () => {
    render(<IdlePlayground />);
    fireEvent.click(lane("O(n)"));
    expect(nReadout()).toBe("n = 24");
  });

  it("keeps the original empty-state instruction visible", () => {
    render(<IdlePlayground />);
    expect(screen.getByText(/Paste code on the left and run an analysis/)).toBeInTheDocument();
  });

  it("gives screen readers a full sentence per lane", () => {
    render(<IdlePlayground />);
    expect(lane("O(log n)")).toHaveAccessibleName(/binary search.*operations.*fits/i);
  });
});

describe("IdlePlayground autoplay", () => {
  beforeEach(() => {
    stubMotion(false);
    // keep the sweep from running during the test: only the button state is under test
    vi.stubGlobal("requestAnimationFrame", () => 0);
    vi.stubGlobal("cancelAnimationFrame", () => {});
  });

  it("starts running and can be paused and resumed", () => {
    render(<IdlePlayground />);
    const toggle = screen.getByRole("button", { name: /PAUSE|AUTO/ });
    expect(toggle).toHaveTextContent("PAUSE");
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(toggle);
    expect(toggle).toHaveTextContent("AUTO");
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveTextContent("PAUSE");
  });

  it("stops for good when the visitor takes the slider", () => {
    render(<IdlePlayground />);
    fireEvent.change(screen.getByRole("slider"), { target: { value: "300" } });
    expect(screen.getByRole("button", { name: /AUTO/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("holds still while a lane is hovered, then carries on", () => {
    render(<IdlePlayground />);
    const toggle = screen.getByRole("button", { name: /PAUSE|AUTO/ });
    fireEvent.mouseEnter(lane("O(n^2)"));
    expect(toggle).toHaveTextContent("AUTO"); // not running while hovered
    fireEvent.mouseLeave(lane("O(n^2)"));
    expect(toggle).toHaveTextContent("PAUSE");
  });
});
