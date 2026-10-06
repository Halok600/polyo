// Lines the bottom edge of one panel up with the bottom edge of another that sits in the other column.
//
// The two columns of the bench are independent stacks, so a panel on the right (the growth chart) and
// a panel on the left (the code input) end wherever their own content ends, a few pixels apart. This
// measures how tall the follower must be to end exactly where the target does and returns it for the
// caller to apply as a min-height. The follower's top does not depend on its own height, so the
// measurement is the same whether or not a min-height is already applied, which keeps it stable
// (applying it does not change what is measured). It only acts in the two-column layout: on a phone
// the panels are stacked and there is nothing to align (it returns undefined).

import { useEffect, useState, type RefObject } from "react";

const TWO_COLUMNS = "(min-width: 761px)";

export function useMatchBottom(
  target: RefObject<HTMLElement | null>,
  follower: RefObject<HTMLElement | null>,
  enabled: boolean,
  remeasureOn: unknown,
): number | undefined {
  const [minHeight, setMinHeight] = useState<number | undefined>(undefined);

  useEffect(() => {
    const targetEl = target.current;
    const followerEl = follower.current;
    if (!enabled || !targetEl || !followerEl) return;

    const query = window.matchMedia(TWO_COLUMNS);
    const measure = () => {
      let next: number | undefined;
      if (query.matches) {
        const needed = targetEl.getBoundingClientRect().bottom - followerEl.getBoundingClientRect().top;
        next = needed > 0 ? Math.round(needed * 100) / 100 : undefined; // sub-pixel: edges meet exactly
      }
      setMinHeight((previous) => (previous === next ? previous : next));
    };

    // always from a frame callback: layout is settled, and a resize observer never changes layout
    // inside its own callback
    let frame = 0;
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    };

    schedule();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(schedule);
    observer?.observe(targetEl);
    if (targetEl.parentElement) observer?.observe(targetEl.parentElement);
    if (followerEl.parentElement) observer?.observe(followerEl.parentElement);
    query.addEventListener("change", schedule);
    void document.fonts?.ready.then(schedule);

    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
      query.removeEventListener("change", schedule);
    };
  }, [target, follower, enabled, remeasureOn]);

  return enabled ? minHeight : undefined;
}
