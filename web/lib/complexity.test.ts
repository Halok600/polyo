import { describe, expect, it } from "vitest";

import {
  AXIS_MAX_LOG10,
  COMPLEXITY_CLASSES,
  MAX_N,
  SLIDER_STEPS,
  barFraction,
  breakingPoint,
  budgetFraction,
  formatDuration,
  formatOps,
  nFromSlider,
  sliderFromN,
  summary,
  verdict,
} from "@/lib/complexity";

const ops = (label: string, n: number) => {
  const found = COMPLEXITY_CLASSES.find((c) => c.label === label);
  if (!found) throw new Error(`no class ${label}`);
  return found.log10Ops(n);
};

describe("the slider is logarithmic in n", () => {
  it("runs from 1 to MAX_N", () => {
    expect(nFromSlider(0)).toBe(1);
    expect(nFromSlider(SLIDER_STEPS)).toBe(MAX_N);
  });

  it("maps equal slider distances to equal factors", () => {
    expect(nFromSlider(250)).toBe(10);
    expect(nFromSlider(500)).toBe(100);
    expect(nFromSlider(750)).toBe(1000);
  });

  it("clamps a position outside the range instead of throwing or returning NaN", () => {
    expect(nFromSlider(-50)).toBe(1);
    expect(nFromSlider(SLIDER_STEPS * 3)).toBe(MAX_N);
  });

  it("round-trips through sliderFromN", () => {
    expect(sliderFromN(100)).toBe(500);
    expect(nFromSlider(sliderFromN(37))).toBe(37);
  });
});

describe("operation counts, as log10", () => {
  it("has the seven classes in order of growth", () => {
    expect(COMPLEXITY_CLASSES.map((c) => c.label)).toEqual([
      "O(1)",
      "O(log n)",
      "O(n)",
      "O(n log n)",
      "O(n^2)",
      "O(n^3)",
      "O(2^n)",
    ]);
  });

  it("gets the powers right at n = 1000", () => {
    expect(ops("O(1)", 1000)).toBe(0);
    expect(ops("O(n)", 1000)).toBeCloseTo(3, 9);
    expect(ops("O(n^2)", 1000)).toBeCloseTo(6, 9);
    expect(ops("O(n^3)", 1000)).toBeCloseTo(9, 9);
    expect(ops("O(n log n)", 1000)).toBeCloseTo(3 + Math.log10(Math.log2(1000)), 9);
  });

  it("makes log n the number of halvings, and 2^n explode", () => {
    expect(ops("O(log n)", 1024)).toBeCloseTo(1, 9); // 10 steps
    expect(ops("O(2^n)", 10)).toBeCloseTo(Math.log10(1024), 9);
    expect(ops("O(2^n)", 100)).toBeCloseTo(100 * Math.log10(2), 9);
  });

  it("never reports fewer than one operation, so the first bar is not NaN or negative", () => {
    for (const c of COMPLEXITY_CLASSES) {
      const value = c.log10Ops(1);
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
  });

  it("keeps every class monotonic in n", () => {
    for (const c of COMPLEXITY_CLASSES) {
      let previous = -1;
      for (const n of [1, 2, 5, 10, 100, 1000, 10000]) {
        const value = c.log10Ops(n);
        expect(value).toBeGreaterThanOrEqual(previous);
        previous = value;
      }
    }
  });
});

describe("verdict against the time budgets (1e9 operations per second)", () => {
  it("is ok up to one second, slow up to a year, never beyond", () => {
    expect(verdict(0)).toBe("ok");
    expect(verdict(9)).toBe("ok"); // exactly one second still fits the budget
    expect(verdict(9.01)).toBe("slow");
    expect(verdict(16.4)).toBe("slow");
    expect(verdict(16.6)).toBe("never");
  });

  it("treats n^3 at n = 1000 as exactly on the one-second line, not over it", () => {
    expect(verdict(ops("O(n^3)", 1000))).toBe("ok");
  });
});

describe("bar and budget geometry", () => {
  it("scales a bar by log10 ops over the axis and clamps it to the track", () => {
    expect(barFraction(0)).toBe(0);
    expect(barFraction(AXIS_MAX_LOG10 / 2)).toBeCloseTo(0.5, 9);
    expect(barFraction(AXIS_MAX_LOG10 * 5)).toBe(1);
    expect(barFraction(Number.POSITIVE_INFINITY)).toBe(1);
  });

  it("places the one-second line where a bar with 1e9 operations ends", () => {
    expect(budgetFraction("second")).toBeCloseTo(barFraction(9), 9);
    expect(budgetFraction("year")).toBeGreaterThan(budgetFraction("second"));
    expect(budgetFraction("year")).toBeLessThan(1);
  });
});

describe("formatOps", () => {
  it("writes small counts in full, with separators", () => {
    expect(formatOps(0)).toBe("1");
    expect(formatOps(Math.log10(1234))).toBe("1,234");
    expect(formatOps(Math.log10(999999))).toBe("999,999");
  });

  it("switches to scientific notation from a million up", () => {
    expect(formatOps(6)).toBe("1.0e6");
    expect(formatOps(9)).toBe("1.0e9");
    expect(formatOps(Math.log10(2.5e12))).toBe("2.5e12");
  });

  it("does not print a mantissa of 10.0", () => {
    expect(formatOps(Math.log10(9.96e7))).toBe("1.0e8");
  });

  it("says infinity for counts no number type holds", () => {
    expect(formatOps(5000)).toBe("∞");
    expect(formatOps(Number.POSITIVE_INFINITY)).toBe("∞");
  });
});

describe("formatDuration", () => {
  it("covers the human scales", () => {
    expect(formatDuration(0)).toBe("< 1 µs");
    expect(formatDuration(Math.log10(5e4))).toBe("50 µs");
    expect(formatDuration(6)).toBe("1.0 ms");
    expect(formatDuration(Math.log10(4.5e7))).toBe("45 ms");
    expect(formatDuration(9)).toBe("1.0 s");
    expect(formatDuration(Math.log10(4.2e10))).toBe("42 s");
    expect(formatDuration(12)).toBe("17 min");
    expect(formatDuration(Math.log10(5 * 3600e9))).toBe("5.0 h");
    expect(formatDuration(Math.log10(3 * 86400e9))).toBe("3.0 days");
  });

  it("counts in years, then compares with the age of the universe", () => {
    expect(formatDuration(Math.log10(41 * 31_557_600e9))).toBe("41 years");
    expect(formatDuration(30)).toMatch(/age of the universe/);
    expect(formatDuration(30)).not.toMatch(/Infinity|NaN/);
  });

  it("gives up gracefully where the count is beyond any number", () => {
    expect(formatDuration(5000)).toBe("beyond counting");
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe("beyond counting");
  });
});

describe("summary line", () => {
  it("says everything fits the budget for a tiny input, without claiming it is instant", () => {
    expect(summary(5)).toMatch(/fits in a second/i);
    expect(summary(5)).not.toMatch(/instant/i);
  });

  it("counts the classes that blow the one-second budget", () => {
    // n = 100: 2^n is far past a year; n^3 = 1e6 is fine
    const text = summary(100);
    expect(text).toMatch(/1 of 7/);
  });

  it("reports more than one when several classes are out of budget", () => {
    // n = 10000: n^3 = 1e12 (17 min, slow) and 2^n (never)
    expect(summary(10000)).toMatch(/2 of 7/);
  });
});

describe("breakingPoint: the smallest n whose run no longer fits in a second", () => {
  const point = (label: string) => breakingPoint(COMPLEXITY_CLASSES.find((c) => c.label === label)!);

  it("finds where exponential growth gives out", () => {
    expect(point("O(2^n)")).toBe(30); // 2^29 = 5.4e8 fits, 2^30 = 1.07e9 does not
  });

  it("finds the first n past exactly one second for cubic", () => {
    expect(point("O(n^3)")).toBe(1001); // 1000^3 is exactly 1e9, which still fits
  });

  it("is null for a class that stays inside the budget across the whole range", () => {
    for (const label of ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)"]) {
      expect(point(label)).toBeNull();
    }
  });

  it("agrees with verdict on both sides of the point", () => {
    const cubic = COMPLEXITY_CLASSES.find((c) => c.label === "O(n^3)")!;
    expect(verdict(cubic.log10Ops(1000))).toBe("ok");
    expect(verdict(cubic.log10Ops(1001))).not.toBe("ok");
  });
});
