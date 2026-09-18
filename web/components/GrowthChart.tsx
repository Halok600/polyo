"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { usePrefersReducedMotion } from "@/lib/motion";

// Cost vs n on a log x-axis -- log-y too, not just log-x: the series span
// many orders of magnitude (O(n) next to O(n^3)), and a linear y-axis
// would flatten every smaller series to the baseline. Still one axis per
// dimension (the dataviz skill's non-negotiable is about dual Y-SCALES on
// one chart, not about which single scale that axis uses). The predicted
// class is bold; every other class in `series` is ghosted behind it, not
// a categorical multi-series comparison.
//
// On mount (a fresh run, or a Time/Space toggle -- the parent keys this
// component by dimension so both replay the reveal), every path
// stroke-draws in via dashoffset rather than appearing whole: the
// predicted trace draws last and heaviest, with a small glowing dot
// riding its leading edge like a plotter pen, exactly the "gridlines are
// structure, the data is the reading" split a bench instrument has.

type GrowthChartProps = {
  title: string;
  n: number[];
  series: Record<string, number[]>;
  predictedClass: string;
  betterClass?: string;
};

const WIDTH = 560;
const HEIGHT = 300;
const MARGIN = { top: 16, right: 92, bottom: 36, left: 16 };
const PREDICTED_DRAW_MS = 750;
const GHOST_DRAW_MS = 500;
const GHOST_BASE_DELAY_MS = 200;
const GHOST_STAGGER_MS = 100;

function logScale(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain.map(Math.log);
  const [r0, r1] = range;
  return (value: number) => {
    const t = (Math.log(value) - d0) / (d1 - d0);
    return r0 + t * (r1 - r0);
  };
}

function logTicks(minVal: number, maxVal: number) {
  const lo = Math.max(minVal, 1e-9);
  const startExp = Math.floor(Math.log10(lo));
  const endExp = Math.ceil(Math.log10(maxVal));
  const major: number[] = [];
  const minor: number[] = [];
  for (let exp = startExp; exp <= endExp; exp++) {
    const base = Math.pow(10, exp);
    if (base >= lo && base <= maxVal) major.push(base);
    for (const mult of [2, 5]) {
      const v = base * mult;
      if (v >= lo && v <= maxVal) minor.push(v);
    }
  }
  return { major, minor };
}

function formatTick(value: number): string {
  if (value >= 1000) return value.toLocaleString(undefined, { maximumSignificantDigits: 2 });
  return value.toLocaleString(undefined, { maximumSignificantDigits: 3 });
}

// Approximates cubic-bezier(0.4, 0, 0.2, 1) closely enough for a
// JS-driven dot -- the CSS transition on the path itself is the source of
// truth for the line; this only has to look like it's riding the same
// curve.
function easeSweep(t: number): number {
  return t * t * (3 - 2 * t);
}

export function GrowthChart({ title, n, series, predictedClass, betterClass }: GrowthChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [lengths, setLengths] = useState<Record<string, number>>({});
  const pathRefs = useRef<Record<string, SVGPathElement | null>>({});
  const dotRef = useRef<SVGCircleElement | null>(null);
  const reducedMotion = usePrefersReducedMotion();

  const classNames = Object.keys(series);
  const ghostClasses = classNames.filter((cls) => cls !== predictedClass);
  const allValues = classNames.flatMap((cls) => series[cls]).filter((v) => v > 0);
  const yMin = Math.min(...allValues);
  const yMax = Math.max(...allValues);

  const plotLeft = MARGIN.left;
  const plotRight = WIDTH - MARGIN.right;
  const plotTop = MARGIN.top;
  const plotBottom = HEIGHT - MARGIN.bottom;

  const xScale = useMemo(
    () => logScale([n[0], n[n.length - 1]], [plotLeft, plotRight]),
    [n, plotLeft, plotRight],
  );
  const yScale = useMemo(
    () => logScale([Math.max(yMin, 1e-6), yMax], [plotBottom, plotTop]),
    [yMin, yMax, plotBottom, plotTop],
  );

  const xTicks = useMemo(() => logTicks(n[0], n[n.length - 1]), [n]);
  const yTicks = useMemo(() => logTicks(Math.max(yMin, 1e-6), yMax), [yMin, yMax]);

  const pathFor = (values: number[]) =>
    values.map((v, i) => `${i === 0 ? "M" : "L"} ${xScale(n[i]).toFixed(1)} ${yScale(v).toFixed(1)}`).join(" ");

  // Measure real path lengths before first paint so the initial dashoffset
  // equals the full length -- flipping `drawing` a beat later is what
  // actually animates the stroke in. This is the standard
  // measure-in-useLayoutEffect-then-setState pattern for DOM measurements
  // (react.dev/reference/react/useLayoutEffect#measuring-layout) -- there's
  // no external store to subscribe to instead, `getTotalLength()` is a
  // one-shot synchronous read.
  useLayoutEffect(() => {
    const map: Record<string, number> = {};
    for (const cls of classNames) {
      const el = pathRefs.current[cls];
      if (el) map[cls] = el.getTotalLength();
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see comment above
    setLengths(map);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classNames.join("|"), n.join("|")]);

  // `isDrawing` starts true immediately under reduced motion (derived at
  // render time, no state to set); otherwise a double rAF flips `drawing`
  // a frame after mount so the browser paints the "fully hidden" dashoffset
  // first and the transition to 0 actually animates.
  useEffect(() => {
    if (reducedMotion) return;
    const raf1 = requestAnimationFrame(() => {
      const raf2 = requestAnimationFrame(() => setDrawing(true));
      return raf2;
    });
    return () => cancelAnimationFrame(raf1);
  }, [reducedMotion]);
  const isDrawing = reducedMotion || drawing;

  // The traveling pen-dot riding the predicted path's leading edge --
  // imperative rAF against refs so a 750ms sweep doesn't push 45 React
  // re-renders through state.
  useEffect(() => {
    if (reducedMotion || !isDrawing) return;
    const path = pathRefs.current[predictedClass];
    const dot = dotRef.current;
    const length = lengths[predictedClass];
    if (!path || !dot || !length) return;

    let start: number | undefined;
    let frame: number;
    const tick = (now: number) => {
      if (start === undefined) start = now;
      const t = Math.min(1, (now - start) / PREDICTED_DRAW_MS);
      const point = path.getPointAtLength(easeSweep(t) * length);
      dot.setAttribute("cx", String(point.x));
      dot.setAttribute("cy", String(point.y));
      dot.style.opacity = "1";
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [isDrawing, reducedMotion, predictedClass, lengths]);

  const handleMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * WIDTH;
    let closest = 0;
    let closestDist = Infinity;
    for (let i = 0; i < n.length; i++) {
      const dist = Math.abs(xScale(n[i]) - px);
      if (dist < closestDist) {
        closestDist = dist;
        closest = i;
      }
    }
    setHoverIndex(closest);
  };

  return (
    <div>
      <div className="mono-nums" style={{ fontSize: 11, letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 8 }}>
        {title.toUpperCase()}
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        height={HEIGHT}
        role="img"
        aria-label={`${title}: growth curve, predicted class ${predictedClass}`}
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {/* Graticule -- structure, never animated. */}
        {yTicks.minor.map((v) => (
          <line key={`ym-${v}`} x1={plotLeft} x2={plotRight} y1={yScale(v)} y2={yScale(v)} stroke="var(--grid-minor)" strokeWidth={1} />
        ))}
        {yTicks.major.map((v) => (
          <line key={`yM-${v}`} x1={plotLeft} x2={plotRight} y1={yScale(v)} y2={yScale(v)} stroke="var(--grid-major)" strokeWidth={1} />
        ))}
        {xTicks.minor.map((v) => (
          <line key={`xm-${v}`} x1={xScale(v)} x2={xScale(v)} y1={plotTop} y2={plotBottom} stroke="var(--grid-minor)" strokeWidth={1} />
        ))}
        {xTicks.major.map((v) => (
          <line key={`xM-${v}`} x1={xScale(v)} x2={xScale(v)} y1={plotTop} y2={plotBottom} stroke="var(--grid-major)" strokeWidth={1} />
        ))}
        {xTicks.major.map((v) => (
          <text key={`xt-${v}`} x={xScale(v)} y={plotBottom + 16} fontSize={10} textAnchor="middle" fill="var(--text-muted)" className="mono-nums">
            {formatTick(v)}
          </text>
        ))}
        <line x1={plotLeft} x2={plotRight} y1={plotBottom} y2={plotBottom} stroke="var(--grid-major)" strokeWidth={1} />

        {ghostClasses.map((cls, i) => {
          const isBetter = cls === betterClass;
          const length = lengths[cls];
          const delayMs = GHOST_BASE_DELAY_MS + i * GHOST_STAGGER_MS;
          // The "better" neighbour is solid, so it can stroke-draw exactly
          // like the predicted line. A repeating dash pattern can't also
          // encode a dashoffset-based reveal (the offset would just slide
          // the pattern's phase, not hide/show it) -- those fade in via
          // opacity instead, which reads as "quieter context," fittingly.
          if (isBetter) {
            return (
              <path
                key={cls}
                ref={(el) => {
                  pathRefs.current[cls] = el;
                }}
                d={pathFor(series[cls])}
                fill="none"
                stroke="var(--ghost-trace)"
                strokeWidth={1.5}
                style={
                  length
                    ? {
                        strokeDasharray: length,
                        strokeDashoffset: isDrawing ? 0 : length,
                        transition: `stroke-dashoffset ${GHOST_DRAW_MS}ms var(--ease-snap)`,
                        transitionDelay: `${delayMs}ms`,
                      }
                    : undefined
                }
              />
            );
          }
          return (
            <path
              key={cls}
              d={pathFor(series[cls])}
              fill="none"
              stroke="var(--ghost-trace)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              style={{
                opacity: isDrawing ? 0.7 : 0,
                transition: `opacity ${GHOST_DRAW_MS}ms var(--ease-snap)`,
                transitionDelay: `${delayMs}ms`,
              }}
            />
          );
        })}
        <path
          ref={(el) => {
            pathRefs.current[predictedClass] = el;
          }}
          d={pathFor(series[predictedClass])}
          fill="none"
          stroke="var(--signal)"
          strokeWidth={2.5}
          strokeLinecap="round"
          style={
            lengths[predictedClass]
              ? {
                  strokeDasharray: lengths[predictedClass],
                  strokeDashoffset: isDrawing ? 0 : lengths[predictedClass],
                  transition: `stroke-dashoffset ${PREDICTED_DRAW_MS}ms var(--ease-sweep)`,
                }
              : undefined
          }
        />
        <circle ref={dotRef} r={4} fill="var(--signal)" opacity={0} style={{ filter: "drop-shadow(0 0 4px var(--signal))" }} />

        {classNames.map((cls) => {
          const values = series[cls];
          const lastX = xScale(n[n.length - 1]);
          const lastY = yScale(values[values.length - 1]);
          const isPredicted = cls === predictedClass;
          return (
            <text
              key={cls}
              x={lastX + 6}
              y={lastY}
              fontSize={11}
              dominantBaseline="middle"
              fill={isPredicted ? "var(--text-primary)" : "var(--text-muted)"}
              fontWeight={isPredicted ? 600 : 400}
              className="mono-nums"
            >
              {cls}
            </text>
          );
        })}
        {hoverIndex !== null ? (
          <>
            <line
              x1={xScale(n[hoverIndex])}
              x2={xScale(n[hoverIndex])}
              y1={plotTop}
              y2={plotBottom}
              stroke="var(--grid-major)"
              strokeWidth={1}
              style={reducedMotion ? undefined : { transition: "x1 140ms var(--ease-settle), x2 140ms var(--ease-settle)" }}
            />
            {classNames.map((cls) => (
              <circle
                key={cls}
                cx={xScale(n[hoverIndex])}
                cy={yScale(series[cls][hoverIndex])}
                r={3}
                fill={cls === predictedClass ? "var(--signal)" : "var(--ghost-trace)"}
                style={reducedMotion ? undefined : { transition: "cx 140ms var(--ease-settle), cy 140ms var(--ease-settle)" }}
              />
            ))}
          </>
        ) : null}
      </svg>
      {hoverIndex !== null ? (
        <div className="mono-nums" style={{ fontSize: 12, color: "var(--text-secondary)", display: "flex", gap: 14, flexWrap: "wrap" }}>
          <span style={{ color: "var(--text-muted)" }}>n = {n[hoverIndex].toLocaleString()}</span>
          {classNames.map((cls) => (
            <span key={cls}>
              {cls}: {series[cls][hoverIndex].toLocaleString(undefined, { maximumSignificantDigits: 3 })}
            </span>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Hover the chart to inspect values.</div>
      )}
      {betterClass ? (
        <div style={{ marginTop: 6, fontSize: 12, color: "var(--status-good)" }}>What faster looks like: {betterClass}</div>
      ) : null}
    </div>
  );
}
