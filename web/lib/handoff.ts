// Timing of the hand-off from the idle playground to the results.
//
// While a request is in flight the playground "scans"; when the answer arrives its lane stays lit and
// the others collapse away; then the result panels take over. The CSS durations are read from these
// same numbers (they are set as custom properties on the playground), so the page never swaps the
// results in before the animation has finished.

/** The scan always runs at least this long, so a fast answer is still seen arriving. */
export const SCAN_MIN_MS = 800;

/** How long the collapse-to-the-answer takes before the results replace the playground. */
export const COLLAPSE_MS = 1000;

/** How much longer the scan must run, given how long the request has already taken. */
export function remainingScanMs(elapsedMs: number): number {
  return Math.max(0, SCAN_MIN_MS - Math.max(0, elapsedMs));
}

/** A promise that resolves after `ms`, or rejects if the signal aborts first (a newer run began). */
export function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}
