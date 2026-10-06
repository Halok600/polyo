"use client";

// What the readout column shows before anything has been analysed: a small toy for the idle moments.
// Seven complexity classes race on one logarithmic axis as the input size n grows. Dashed lines mark
// "1 second" and "1 year" of work at 1e9 operations a second, so you can watch each class run out of
// budget in turn. It sweeps n on its own until the visitor takes the slider (or hovers a lane, which
// holds still so the lane can be read), and clicking a lane jumps to the n where it passes 1 second.
// With reduced motion it does not move by itself; every control still works.

import { useEffect, useRef, useState } from "react";

import { BenchPanel } from "@/components/BenchPanel";
import {
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
  type Verdict,
} from "@/lib/complexity";
import { usePrefersReducedMotion } from "@/lib/motion";

const SWEEP_MS = 18_000;
const START_N = 24;
const SLIDER_TICKS = [
  { n: 1, label: "1" },
  { n: 10, label: "10" },
  { n: 100, label: "100" },
  { n: 1000, label: "1k" },
  { n: 10_000, label: "10k" },
];
const VERDICT_WORD: Record<Verdict, string> = { ok: "FITS", slow: "SLOW", never: "NEVER" };
// computed once: every class is monotonic, so these never change
const BREAKING_POINTS = COMPLEXITY_CLASSES.map(breakingPoint);

const opsText = (log10: number) => `${formatOps(log10)} ${log10 === 0 ? "op" : "ops"}`;
const operationsText = (log10: number) => `${formatOps(log10)} ${log10 === 0 ? "operation" : "operations"}`;

function laneHint(index: number): string {
  const { label, example } = COMPLEXITY_CLASSES[index];
  const point = BREAKING_POINTS[index];
  const tail =
    point === null
      ? `Stays under 1 second all the way to n = ${MAX_N.toLocaleString("en-US")}.`
      : `Passes 1 second at n = ${point.toLocaleString("en-US")}. Click to jump there.`;
  return `${label} — ${example}. ${tail}`;
}

export function IdlePlayground() {
  const reducedMotion = usePrefersReducedMotion();
  const [n, setN] = useState(START_N);
  const [auto, setAuto] = useState(true);
  const [active, setActive] = useState<number | null>(null);
  const nRef = useRef(n);
  useEffect(() => {
    nRef.current = n;
  });
  // holding still while a lane is hovered lets it be read; it carries on from the same n afterwards
  const running = auto && !reducedMotion && active === null;

  useEffect(() => {
    if (!running) return;
    // continue the sweep from the current n instead of jumping back to the start
    const fraction = Math.min(1, Math.max(0, sliderFromN(nRef.current) / SLIDER_STEPS));
    const base = Math.acos(1 - 2 * fraction);
    let frame = 0;
    let startedAt: number | null = null;
    const tick = (now: number) => {
      if (startedAt === null) startedAt = now;
      const angle = base + ((now - startedAt) / SWEEP_MS) * 2 * Math.PI;
      setN(nFromSlider(SLIDER_STEPS * (0.5 - 0.5 * Math.cos(angle))));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [running]);

  const lanes = COMPLEXITY_CLASSES.map((cls) => {
    const log10 = cls.log10Ops(n);
    return { cls, log10, state: verdict(log10) };
  });

  function jumpToBreakingPoint(index: number) {
    const point = BREAKING_POINTS[index];
    if (point === null) return;
    setAuto(false);
    setN(point);
  }

  return (
    <BenchPanel channel="CH.03 — PLAYGROUND" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
      <div data-testid="playground" className="pg">
        <div className="pg-top">
          <div>
            <div className="mono-nums field-label">INPUT SIZE</div>
            <div className="mono-nums pg-n" data-testid="playground-n">
              n = {n.toLocaleString("en-US")}
            </div>
          </div>
          <button
            type="button"
            className={`mono-nums dim-tab pg-auto${running ? " is-active" : ""}`}
            aria-pressed={auto && !reducedMotion}
            disabled={reducedMotion}
            onClick={() => setAuto((value) => !value)}
          >
            {running ? "❚❚ PAUSE" : "▶ AUTO"}
          </button>
        </div>

        <div className="pg-slider-wrap">
          <input
            className="pg-slider"
            type="range"
            min={0}
            max={SLIDER_STEPS}
            step={1}
            value={sliderFromN(n)}
            aria-label="Input size n"
            aria-valuetext={`n = ${n.toLocaleString("en-US")}`}
            onChange={(event) => {
              setAuto(false);
              setN(nFromSlider(Number(event.target.value)));
            }}
          />
          <div className="pg-ticks mono-nums" aria-hidden>
            {SLIDER_TICKS.map((tick) => (
              <span key={tick.n} style={{ left: `calc(7px + (100% - 14px) * ${sliderFromN(tick.n) / SLIDER_STEPS})` }}>
                {tick.label}
              </span>
            ))}
          </div>
        </div>

        <div className="pg-axis mono-nums" aria-hidden>
          <span style={{ left: 0 }}>1 op</span>
          <span style={{ left: `${budgetFraction("second") * 100}%` }}>1 s</span>
          <span style={{ left: `${budgetFraction("year") * 100}%` }}>1 year</span>
        </div>

        <div className="pg-lanes">
          <ul className="pg-lane-list" aria-label="Operations needed by each complexity class at this n">
            {lanes.map(({ cls, log10, state }, index) => (
              <li key={cls.label}>
                <button
                  type="button"
                  className="pg-lane"
                  data-verdict={state}
                  data-active={active === index ? "true" : undefined}
                  aria-label={`${cls.label}, ${cls.example}: about ${operationsText(log10)}, ${formatDuration(log10)}, ${VERDICT_WORD[state].toLowerCase()}. ${laneHint(index)}`}
                  onMouseEnter={() => setActive(index)}
                  onMouseLeave={() => setActive(null)}
                  onFocus={() => setActive(index)}
                  onBlur={() => setActive(null)}
                  onClick={() => jumpToBreakingPoint(index)}
                >
                  <span className="pg-lane-head">
                    <span className="mono-nums pg-lane-name">{cls.label}</span>
                    <span className="pg-lane-example">{cls.example}</span>
                    <span className="mono-nums pg-lane-meta">
                      {opsText(log10)} · <span className="pg-lane-verdict">{VERDICT_WORD[state]}</span>{" "}
                      {formatDuration(log10)}
                    </span>
                  </span>
                  <span className="pg-track">
                    <span className="pg-bar" style={{ transform: `scaleX(${barFraction(log10)})` }} />
                    <span className="pg-mark" style={{ left: `${budgetFraction("second") * 100}%` }} aria-hidden />
                    <span className="pg-mark" style={{ left: `${budgetFraction("year") * 100}%` }} aria-hidden />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        <p className="pg-footer" data-testid="playground-footer">
          {active === null ? summary(n) : laneHint(active)}
        </p>
        <p className="pg-hint">
          Paste code on the left and run an analysis — the exact cost expression, how it was derived line by
          line, and the growth curve will read out here.
        </p>
      </div>
    </BenchPanel>
  );
}
