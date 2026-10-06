// The numbers behind the idle playground: how many operations each complexity class needs at an input
// size n, and what that costs in wall-clock time on a machine doing 1e9 simple operations a second.
// Everything is kept as log10 so O(2^n) at n = 10000 (10^3010) is a number, not Infinity, and the bars
// can sit on one logarithmic axis.

export const MAX_N = 10_000;
export const SLIDER_STEPS = 1000;

// Bars are drawn on a log axis from 1 operation (0) to 1e18 (the right edge).
export const AXIS_MAX_LOG10 = 18;

const OPS_PER_SECOND_LOG10 = 9;
const SECONDS_PER_YEAR = 31_557_600;
const YEAR_LOG10 = OPS_PER_SECOND_LOG10 + Math.log10(SECONDS_PER_YEAR);
const AGE_OF_UNIVERSE_YEARS = 13.8e9;
// beyond this a count is not worth printing: it is not "slow", it is not going to happen
const UNCOUNTABLE_LOG10 = 300;
const EPSILON = 1e-9;

const log2 = (n: number) => Math.log2(Math.max(1, n));

export type ComplexityClass = {
  label: string;
  example: string;
  /** log10 of the operation count at input size n (at least 0: one operation) */
  log10Ops: (n: number) => number;
};

export const COMPLEXITY_CLASSES: readonly ComplexityClass[] = [
  { label: "O(1)", example: "hash lookup", log10Ops: () => 0 },
  { label: "O(log n)", example: "binary search", log10Ops: (n) => Math.log10(Math.max(1, log2(n))) },
  { label: "O(n)", example: "one pass over the data", log10Ops: (n) => Math.log10(n) },
  {
    label: "O(n log n)",
    example: "merge sort",
    log10Ops: (n) => Math.log10(n) + Math.log10(Math.max(1, log2(n))),
  },
  { label: "O(n^2)", example: "a loop inside a loop", log10Ops: (n) => 2 * Math.log10(n) },
  { label: "O(n^3)", example: "three nested loops", log10Ops: (n) => 3 * Math.log10(n) },
  { label: "O(2^n)", example: "trying every subset", log10Ops: (n) => n * Math.log10(2) },
];

/** Which of the seven lanes an answer such as "O(n^2)" belongs to, or -1 if it is none of them. */
export function laneIndex(label: string | null | undefined): number {
  if (!label) return -1;
  return COMPLEXITY_CLASSES.findIndex((c) => c.label === label);
}

// --- the slider --------------------------------------------------------------------------------

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** Slider position 0..SLIDER_STEPS to an input size 1..MAX_N, spaced logarithmically. */
export function nFromSlider(position: number): number {
  const t = clamp(position / SLIDER_STEPS, 0, 1);
  return Math.max(1, Math.round(10 ** (t * Math.log10(MAX_N))));
}

export function sliderFromN(n: number): number {
  return Math.round((Math.log10(clamp(n, 1, MAX_N)) / Math.log10(MAX_N)) * SLIDER_STEPS);
}

// --- budgets and bars --------------------------------------------------------------------------

export type Verdict = "ok" | "slow" | "never";

/** Against 1e9 operations a second: fits in a second, fits in a year, or never finishes. */
export function verdict(log10Ops: number): Verdict {
  if (log10Ops <= OPS_PER_SECOND_LOG10 + EPSILON) return "ok";
  if (log10Ops <= YEAR_LOG10) return "slow";
  return "never";
}

/** How far along the track a bar with this many operations reaches, 0..1. */
export function barFraction(log10Ops: number): number {
  if (Number.isNaN(log10Ops)) return 0;
  return clamp(log10Ops / AXIS_MAX_LOG10, 0, 1);
}

export function budgetFraction(budget: "second" | "year"): number {
  return barFraction(budget === "second" ? OPS_PER_SECOND_LOG10 : YEAR_LOG10);
}

/**
 * The smallest n in 1..MAX_N at which the class no longer fits in a second, or null if it never leaves
 * the budget in that range. Every class is monotonic in n, so a binary search finds it.
 */
export function breakingPoint(cls: ComplexityClass): number | null {
  if (verdict(cls.log10Ops(MAX_N)) === "ok") return null;
  let low = 1;
  let high = MAX_N;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (verdict(cls.log10Ops(mid)) === "ok") low = mid + 1;
    else high = mid;
  }
  return low;
}

// --- text --------------------------------------------------------------------------------------

function scientific(value: number, exponent: number): string {
  let mantissa = Number((value / 10 ** exponent).toFixed(1));
  let power = exponent;
  if (mantissa >= 10) {
    mantissa = 1;
    power += 1;
  }
  return `${mantissa.toFixed(1)}e${power}`;
}

function isUncountable(log10Value: number): boolean {
  return !Number.isFinite(log10Value) || log10Value > UNCOUNTABLE_LOG10;
}

export function formatOps(log10Ops: number): string {
  if (isUncountable(log10Ops)) return "∞";
  const ops = 10 ** log10Ops;
  if (ops < 999_999.5) return Math.round(ops).toLocaleString("en-US");
  return scientific(ops, Math.floor(log10Ops + 1e-12));
}

function withUnit(value: number, unit: string): string {
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${unit}`;
}

export function formatDuration(log10Ops: number): string {
  if (isUncountable(log10Ops)) return "beyond counting";
  const seconds = 10 ** (log10Ops - OPS_PER_SECOND_LOG10);
  if (seconds < 1e-6) return "< 1 µs";
  if (seconds < 1e-3) return withUnit(seconds * 1e6, "µs");
  if (seconds < 1) return withUnit(seconds * 1e3, "ms");
  if (seconds < 60) return withUnit(seconds, "s");
  if (seconds < 3600) return withUnit(seconds / 60, "min");
  if (seconds < 86_400) return withUnit(seconds / 3600, "h");
  if (seconds < SECONDS_PER_YEAR) return withUnit(seconds / 86_400, "days");

  const years = seconds / SECONDS_PER_YEAR;
  if (years < 1e7) return `${Math.round(years).toLocaleString("en-US")} years`;
  if (years < AGE_OF_UNIVERSE_YEARS) return `${scientific(years, Math.floor(Math.log10(years)))} years`;
  const universes = years / AGE_OF_UNIVERSE_YEARS;
  const how = universes < 1e6 ? Math.round(universes).toLocaleString("en-US") : scientific(universes, Math.floor(Math.log10(universes)));
  return `${how}× the age of the universe`;
}

/** One line under the lanes: how many classes have already blown the one-second budget at this n. */
export function summary(n: number): string {
  const over = COMPLEXITY_CLASSES.filter((c) => verdict(c.log10Ops(n)) !== "ok").length;
  if (over === 0) return "Everything fits in a second. Small inputs hide bad algorithms.";
  return `${over} of ${COMPLEXITY_CLASSES.length} classes blow the 1-second budget.`;
}
