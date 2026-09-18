// Shared motion primitives for the "Bench" instrument-panel reveal --
// count-up readouts and a reduced-motion escape hatch, used by every
// result component so a run always looks like a fresh reading landing,
// not a page that merely re-rendered.

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

function subscribeReducedMotion(callback: () => void) {
  const query = window.matchMedia("(prefers-reduced-motion: reduce)");
  query.addEventListener("change", callback);
  return () => query.removeEventListener("change", callback);
}

function getReducedMotionSnapshot() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function getReducedMotionServerSnapshot() {
  return false;
}

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribeReducedMotion, getReducedMotionSnapshot, getReducedMotionServerSnapshot);
}

// Matches --ease-settle (cubic-bezier(0.16, 1, 0.3, 1)) closely enough for
// a JS-driven readout -- a fast start with a long, decelerating settle.
function easeSettle(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Counts a number up from 0 to `target` like a digital meter locking onto a
 * reading, instead of a static value that simply appears. Restart by
 * remounting the owning component (the result panel remounts per run via a
 * `key` on its container), not by calling this imperatively.
 */
export function useCountUp(target: number, durationMs: number, delayMs = 0): number {
  const reducedMotion = usePrefersReducedMotion();
  const [value, setValue] = useState(0);
  const frame = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (reducedMotion) return;
    let start: number | undefined;
    const timeout = window.setTimeout(() => {
      const tick = (now: number) => {
        if (start === undefined) start = now;
        const t = Math.min(1, (now - start) / durationMs);
        setValue(target * easeSettle(t));
        if (t < 1) frame.current = requestAnimationFrame(tick);
      };
      frame.current = requestAnimationFrame(tick);
    }, delayMs);
    return () => {
      window.clearTimeout(timeout);
      if (frame.current) cancelAnimationFrame(frame.current);
    };
  }, [target, durationMs, delayMs, reducedMotion]);

  return reducedMotion ? target : value;
}

type StartViewTransition = (callback: () => void) => { finished: Promise<void> };

/**
 * Runs `update` inside the native View Transitions API, tagging <html> with
 * `transitionClass` for CSS to key off (the theme wipe and the landing->tool
 * morph both use this API but need different root behaviour, see
 * globals.css). Falls back to a plain synchronous update when the browser
 * lacks the API or the user prefers reduced motion -- same escape hatch
 * every other animation in this app uses.
 */
export function withViewTransition(update: () => void, reducedMotion: boolean, transitionClass: string): void {
  const doc = document as Document & { startViewTransition?: StartViewTransition };
  if (reducedMotion || typeof doc.startViewTransition !== "function") {
    update();
    return;
  }
  doc.documentElement.classList.add(transitionClass);
  const transition = doc.startViewTransition(update);
  transition.finished.finally(() => doc.documentElement.classList.remove(transitionClass));
}
