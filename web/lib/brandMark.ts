// The PolyO mark: an "O" (a ring, open at the upper right) with a growth curve breaking out of it,
// on a near-black rounded square, in the bench's amber. Two things draw it and they must agree:
//   * app/icon.svg, the file Next serves as the tab icon;
//   * drawBrandMark(), which paints the same shapes on a canvas so the favicon can spin while an
//     analysis runs (lib/useTabStatus.ts).
// The numbers live here; lib/brandMark.test.ts fails if icon.svg drifts from them.

export const MARK = {
  size: 64,
  radius: 15,
  background: "#0a0b0d",
  ring: { cx: 29, cy: 36, r: 17, width: 8.5, dash: [84, 22.8], rotate: -20, from: "#ff7a1a", to: "#ffc46b" },
  swoosh: { d: "M29 36 C37 36 47 28 56 9", width: 5.5, color: "#fff3e0" },
  endDot: { cx: 56, cy: 9, r: 5 },
  startDot: { cx: 29, cy: 36, r: 3.5 },
} as const;

export type MarkFrame = {
  /** extra degrees to turn the ring by (the loader spin); 0 draws the static icon */
  spin?: number;
  /** 0..1, how bright the end dot glows (1 = as in the static icon) */
  glow?: number;
};

const degrees = (value: number) => (value * Math.PI) / 180;

/** Paints the mark on a square canvas context, replacing whatever was there. */
export function drawBrandMark(ctx: CanvasRenderingContext2D, size: number, { spin = 0, glow = 1 }: MarkFrame = {}) {
  const { ring, swoosh, endDot, startDot } = MARK;
  ctx.clearRect(0, 0, size, size);

  ctx.save();
  ctx.scale(size / MARK.size, size / MARK.size);

  // the dark rounded square
  ctx.beginPath();
  ctx.roundRect(0, 0, MARK.size, MARK.size, MARK.radius);
  ctx.fillStyle = MARK.background;
  ctx.fill();

  // the ring: amber gradient, round-capped dashes (that is what leaves the gap), turned by `spin`
  const gradient = ctx.createLinearGradient(ring.cx - ring.r, ring.cy + ring.r, ring.cx + ring.r, ring.cy - ring.r);
  gradient.addColorStop(0, ring.from);
  gradient.addColorStop(1, ring.to);
  ctx.save();
  ctx.translate(ring.cx, ring.cy);
  ctx.rotate(degrees(ring.rotate + spin));
  ctx.beginPath();
  ctx.arc(0, 0, ring.r, 0, Math.PI * 2);
  ctx.setLineDash([...ring.dash]);
  ctx.lineWidth = ring.width;
  ctx.lineCap = "round";
  ctx.strokeStyle = gradient;
  ctx.stroke();
  ctx.restore();

  // the swoosh breaking out of the ring, from the very same path string icon.svg carries
  ctx.setLineDash([]);
  ctx.lineWidth = swoosh.width;
  ctx.lineCap = "round";
  ctx.strokeStyle = swoosh.color;
  ctx.stroke(new Path2D(swoosh.d));

  // both dots; the end one can glow
  ctx.fillStyle = swoosh.color;
  ctx.beginPath();
  ctx.arc(startDot.cx, startDot.cy, startDot.r, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 0.35 + 0.65 * Math.min(1, Math.max(0, glow));
  ctx.beginPath();
  ctx.arc(endDot.cx, endDot.cy, endDot.r, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}
