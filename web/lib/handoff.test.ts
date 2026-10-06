import { describe, expect, it } from "vitest";

import { COLLAPSE_MS, SCAN_MIN_MS, remainingScanMs } from "@/lib/handoff";

describe("remainingScanMs", () => {
  it("makes a fast answer wait out the rest of the minimum scan, so the scan is seen", () => {
    expect(remainingScanMs(0)).toBe(SCAN_MIN_MS);
    expect(remainingScanMs(300)).toBe(SCAN_MIN_MS - 300);
  });

  it("adds no delay once the request has already taken longer than the minimum", () => {
    expect(remainingScanMs(SCAN_MIN_MS)).toBe(0);
    expect(remainingScanMs(40_000)).toBe(0); // a cold-start wake-up is never slowed further
  });

  it("never returns a negative wait for a clock that went backwards", () => {
    expect(remainingScanMs(-50)).toBe(SCAN_MIN_MS);
  });

  it("keeps the whole hand-off short enough not to feel like waiting", () => {
    expect(SCAN_MIN_MS + COLLAPSE_MS).toBeLessThanOrEqual(2000);
  });
});
