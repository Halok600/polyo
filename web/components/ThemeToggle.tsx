"use client";

// A literal two-position rocker, not a sun/moon pill -- the user asked for
// an explicit switch, not another icon button. `app/layout.tsx`'s inline
// script already stamps `data-theme` on <html> before first paint (no
// flash); the current theme lives on that DOM attribute, not in React
// state, so this subscribes to it via useSyncExternalStore rather than
// mirroring it into a `useState` + effect.

import { useSyncExternalStore } from "react";

import { usePrefersReducedMotion, withViewTransition } from "@/lib/motion";

type Theme = "light" | "dark";

function subscribe(callback: () => void) {
  const observer = new MutationObserver(callback);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => observer.disconnect();
}

function getSnapshot(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function getServerSnapshot(): Theme {
  return "dark";
}

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const reducedMotion = usePrefersReducedMotion();

  function toggle(event: React.MouseEvent<HTMLButtonElement>) {
    const next: Theme = theme === "dark" ? "light" : "dark";

    // Origin + reach for the circular wipe -- the circle has to grow past
    // the corner farthest from the click for it to cover the whole
    // viewport before the transition ends.
    const x = event.clientX;
    const y = event.clientY;
    const radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    const root = document.documentElement.style;
    root.setProperty("--vt-x", `${x}px`);
    root.setProperty("--vt-y", `${y}px`);
    root.setProperty("--vt-r", `${radius}px`);

    withViewTransition(
      () => {
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem("polyo-theme", next);
        } catch {
          // Private-browsing/storage-denied -- the click still flips the
          // theme for this page view, it just won't persist across visits.
        }
      },
      reducedMotion,
      "vt-theme",
    );
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      aria-pressed={theme === "dark"}
      style={{
        position: "relative",
        display: "inline-flex",
        alignItems: "center",
        width: 76,
        height: 26,
        padding: 2,
        border: "1px solid var(--border-strong)",
        borderRadius: 3,
        background: "var(--surface-2)",
        cursor: "pointer",
        boxShadow: "inset 0 1px 0 var(--border)",
      }}
    >
      <span
        aria-hidden
        style={{
          position: "absolute",
          top: 2,
          bottom: 2,
          left: theme === "dark" ? 2 : "50%",
          width: "calc(50% - 2px)",
          background: "var(--surface-1)",
          border: "1px solid var(--border-strong)",
          borderRadius: 2,
          transition: "left 150ms var(--ease-snap)",
        }}
      />
      <span
        className="mono-nums"
        style={{
          position: "relative",
          flex: 1,
          textAlign: "center",
          fontSize: 9,
          letterSpacing: "0.08em",
          color: theme === "dark" ? "var(--text-primary)" : "var(--text-muted)",
          zIndex: 1,
        }}
      >
        DARK
      </span>
      <span
        className="mono-nums"
        style={{
          position: "relative",
          flex: 1,
          textAlign: "center",
          fontSize: 9,
          letterSpacing: "0.08em",
          color: theme === "light" ? "var(--text-primary)" : "var(--text-muted)",
          zIndex: 1,
        }}
      >
        LIGHT
      </span>
    </button>
  );
}
