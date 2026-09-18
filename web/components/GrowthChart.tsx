"use client";

import { useMemo, useState } from "react";

// Cost vs n on a log x-axis (plan §11) -- log-y too, not just log-x: the
// series span many orders of magnitude (O(n) next to O(n^3)), and a linear
// y-axis would flatten every smaller series to the baseline. Still one
// axis per dimension (the dataviz skill's non-negotiable is about dual
// Y-SCALES on one chart, not about which single scale that axis uses).
// The predicted class is bold; every other class in `series` is ghosted
// behind it (plan §11) -- not a categorical multi-series comparison, so
// this deliberately doesn't reach for the eight-hue categorical palette.

type GrowthChartProps = {
  title: string;
  n: number[];
  series: Record<string, number[]>;
  predictedClass: string;
  betterClass?: string;
};

const WIDTH = 560;
const HEIGHT = 280;
const MARGIN = { top: 16, right: 88, bottom: 32, left: 16 };

function logScale(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain.map(Math.log);
  const [r0, r1] = range;
  return (value: number) => {
    const t = (Math.log(value) - d0) / (d1 - d0);
    return r0 + t * (r1 - r0);
  };
}

export function GrowthChart({ title, n, series, predictedClass, betterClass }: GrowthChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const classNames = Object.keys(series);
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

  const pathFor = (values: number[]) =>
    values.map((v, i) => `${i === 0 ? "M" : "L"} ${xScale(n[i]).toFixed(1)} ${yScale(v).toFixed(1)}`).join(" ");

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
      <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 4 }}>{title}</div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        height={HEIGHT}
        role="img"
        aria-label={`${title}: growth curve, predicted class ${predictedClass}`}
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <line
          x1={plotLeft}
          x2={plotRight}
          y1={plotBottom}
          y2={plotBottom}
          stroke="var(--baseline)"
          strokeWidth={1}
        />
        {classNames
          .filter((cls) => cls !== predictedClass)
          .map((cls) => (
            <path
              key={cls}
              d={pathFor(series[cls])}
              fill="none"
              stroke="var(--series-ghost)"
              strokeWidth={1.5}
              strokeDasharray={cls === betterClass ? undefined : "4 3"}
              opacity={0.7}
            />
          ))}
        <path
          d={pathFor(series[predictedClass])}
          fill="none"
          stroke="var(--series-predicted)"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
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
              stroke="var(--gridline)"
              strokeWidth={1}
            />
            {classNames.map((cls) => (
              <circle
                key={cls}
                cx={xScale(n[hoverIndex])}
                cy={yScale(series[cls][hoverIndex])}
                r={3}
                fill={cls === predictedClass ? "var(--series-predicted)" : "var(--series-ghost)"}
              />
            ))}
          </>
        ) : null}
      </svg>
      {hoverIndex !== null ? (
        <div
          style={{
            fontSize: 12,
            color: "var(--text-secondary)",
            display: "flex",
            gap: 14,
            flexWrap: "wrap",
          }}
        >
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
        <div style={{ marginTop: 6, fontSize: 12, color: "var(--status-good)" }}>
          What faster looks like: {betterClass}
        </div>
      ) : null}
    </div>
  );
}
