import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MARK, drawBrandMark } from "@/lib/brandMark";

const iconSvg = readFileSync(resolve(__dirname, "../app/icon.svg"), "utf-8");

describe("app/icon.svg and the animated favicon share one geometry", () => {
  // The static icon is a file Next serves; the animated one is drawn in code. If one is edited
  // without the other the tab icon would visibly change shape the moment an analysis starts.
  it("has the ring where the code draws it", () => {
    expect(iconSvg).toContain(`cx="${MARK.ring.cx}"`);
    expect(iconSvg).toContain(`cy="${MARK.ring.cy}"`);
    expect(iconSvg).toContain(`r="${MARK.ring.r}"`);
    expect(iconSvg).toContain(`stroke-width="${MARK.ring.width}"`);
    expect(iconSvg).toContain(`stroke-dasharray="${MARK.ring.dash.join(" ")}"`);
    expect(iconSvg).toContain(`rotate(${MARK.ring.rotate} ${MARK.ring.cx} ${MARK.ring.cy})`);
  });

  it("has the swoosh and both dots where the code draws them", () => {
    expect(iconSvg).toContain(`d="${MARK.swoosh.d}"`);
    expect(iconSvg).toContain(`stroke-width="${MARK.swoosh.width}"`);
    expect(iconSvg).toContain(`cx="${MARK.endDot.cx}" cy="${MARK.endDot.cy}" r="${MARK.endDot.r}"`);
    expect(iconSvg).toContain(`cx="${MARK.startDot.cx}" cy="${MARK.startDot.cy}" r="${MARK.startDot.r}"`);
  });

  it("uses the same colours and the same dark rounded square", () => {
    for (const colour of [MARK.background, MARK.ring.from, MARK.ring.to, MARK.swoosh.color]) {
      expect(iconSvg.toLowerCase()).toContain(colour.toLowerCase());
    }
    expect(iconSvg).toContain(`rx="${MARK.radius}"`);
    expect(iconSvg).toContain(`viewBox="0 0 ${MARK.size} ${MARK.size}"`);
  });

  it("is a standalone svg with no script or external reference", () => {
    expect(iconSvg).toContain('xmlns="http://www.w3.org/2000/svg"');
    expect(iconSvg).not.toMatch(/<script|href=|xlink:|url\(http/i);
  });
});

function fakeContext() {
  const calls: { name: string; args: unknown[] }[] = [];
  const record = (name: string) => (...args: unknown[]) => void calls.push({ name, args });
  const gradient = { addColorStop: vi.fn() };
  const ctx = {
    calls,
    save: record("save"),
    restore: record("restore"),
    scale: record("scale"),
    translate: record("translate"),
    rotate: record("rotate"),
    beginPath: record("beginPath"),
    arc: record("arc"),
    fill: record("fill"),
    stroke: record("stroke"),
    clearRect: record("clearRect"),
    setLineDash: record("setLineDash"),
    roundRect: record("roundRect"),
    createLinearGradient: () => gradient,
    set fillStyle(_: unknown) {},
    set strokeStyle(_: unknown) {},
    set lineWidth(_: number) {},
    set lineCap(_: string) {},
    set globalAlpha(_: number) {},
  };
  return ctx as unknown as CanvasRenderingContext2D & { calls: typeof calls };
}

class FakePath2D {
  constructor(readonly d: string) {}
}

describe("drawBrandMark", () => {
  beforeEach(() => {
    vi.stubGlobal("Path2D", FakePath2D);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("strokes the swoosh from the same path string the svg carries", () => {
    const ctx = fakeContext();
    drawBrandMark(ctx, 64);
    const strokedPaths = ctx.calls.filter((c) => c.name === "stroke" && c.args[0] instanceof FakePath2D);
    expect(strokedPaths).toHaveLength(1);
    expect((strokedPaths[0].args[0] as FakePath2D).d).toBe(MARK.swoosh.d);
  });

  it("scales the 64-unit design to the canvas size", () => {
    const ctx = fakeContext();
    drawBrandMark(ctx, 32);
    expect(ctx.calls.find((c) => c.name === "scale")?.args).toEqual([0.5, 0.5]);
  });

  it("draws the ring on the shared geometry and dashes it like the svg", () => {
    const ctx = fakeContext();
    drawBrandMark(ctx, 64);
    const arc = ctx.calls.find((c) => c.name === "arc" && c.args[2] === MARK.ring.r);
    expect(arc).toBeDefined();
    expect(ctx.calls.some((c) => c.name === "setLineDash" && JSON.stringify(c.args[0]) === JSON.stringify(MARK.ring.dash))).toBe(true);
  });

  it("turns the ring by the spin angle on top of the design's own rotation", () => {
    const still = fakeContext();
    drawBrandMark(still, 64);
    const spun = fakeContext();
    drawBrandMark(spun, 64, { spin: 90 });
    const angle = (ctx: ReturnType<typeof fakeContext>) => ctx.calls.find((c) => c.name === "rotate")!.args[0] as number;
    expect(angle(spun) - angle(still)).toBeCloseTo((90 * Math.PI) / 180, 9);
  });

  it("clears first, so a frame never draws over the previous one", () => {
    const ctx = fakeContext();
    drawBrandMark(ctx, 32);
    expect(ctx.calls[0].name).toBe("clearRect");
  });

  it("does not throw for a tiny or zero size, and leaves the context state balanced", () => {
    for (const size of [0, 1, 16]) {
      const ctx = fakeContext();
      expect(() => drawBrandMark(ctx, size)).not.toThrow();
      const saves = ctx.calls.filter((c) => c.name === "save").length;
      const restores = ctx.calls.filter((c) => c.name === "restore").length;
      expect(restores).toBe(saves);
    }
  });
});
