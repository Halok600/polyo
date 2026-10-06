"use client";

// What the browser tab says about PolyO: a real title instead of the bare name, the answer in the
// title once there is one (so it can be read from another tab), and the icon spinning while a request
// is in flight. The spinner repaints the same mark the static icon is (lib/brandMark.ts) and puts the
// original icons back exactly afterwards; it does nothing for visitors who prefer reduced motion.

import { useEffect } from "react";

import { drawBrandMark } from "@/lib/brandMark";
import { usePrefersReducedMotion } from "@/lib/motion";
import { SITE_TITLE } from "@/lib/siteMeta";

export { SITE_TITLE };

const PRODUCT = "PolyO";
const FRAME_MS = 90;
const FRAME_SIZE = 32;
const SPIN_DEGREES_PER_FRAME = 24;

export type TabState = {
  /** a request is running (an older answer may still be on screen) */
  analysing: boolean;
  /** the headline answer, e.g. "O(n * m)", or null when there is none */
  answer: string | null;
};

export function tabTitle({ analysing, answer }: TabState): string {
  if (analysing) return `Analysing… · ${PRODUCT}`;
  if (answer) return `${answer} · ${PRODUCT}`;
  return SITE_TITLE;
}

export function useFaviconSpinner(active: boolean): void {
  const reducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (!active || reducedMotion) return;
    // the tab icons only: not the apple touch icon (rel="apple-touch-icon" is a different token)
    const links = Array.from(document.querySelectorAll<HTMLLinkElement>('link[rel~="icon"]'));
    if (links.length === 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = FRAME_SIZE;
    canvas.height = FRAME_SIZE;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const originals = links.map((link) => ({
      link,
      href: link.getAttribute("href"),
      type: link.getAttribute("type"),
      sizes: link.getAttribute("sizes"),
    }));

    let frame = 0;
    const paint = () => {
      drawBrandMark(ctx, FRAME_SIZE, { spin: frame * SPIN_DEGREES_PER_FRAME, glow: 0.5 + 0.5 * Math.sin(frame / 3) });
      const url = canvas.toDataURL("image/png");
      for (const { link } of originals) {
        link.setAttribute("href", url);
        link.setAttribute("type", "image/png");
        link.removeAttribute("sizes");
      }
      frame += 1;
    };
    paint();
    const timer = window.setInterval(paint, FRAME_MS);

    return () => {
      window.clearInterval(timer);
      for (const { link, href, type, sizes } of originals) {
        const restore = (name: string, value: string | null) =>
          value === null ? link.removeAttribute(name) : link.setAttribute(name, value);
        restore("href", href);
        restore("type", type);
        restore("sizes", sizes);
      }
    };
  }, [active, reducedMotion]);
}

export function useTabStatus(state: TabState): void {
  const title = tabTitle(state);
  useEffect(() => {
    document.title = title;
    return () => {
      document.title = SITE_TITLE;
    };
  }, [title]);
  useFaviconSpinner(state.analysing);
}
