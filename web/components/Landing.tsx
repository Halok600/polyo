"use client";

// The entry scene -- reuses the exact same toolbar as the tool view (see
// components/Toolbar.tsx) so the header holds still across the morph, and
// keeps every capability claim traceable back to README.md rather than
// inventing marketing copy.

import { BenchPanel } from "@/components/BenchPanel";
import { StatusReadout } from "@/components/StatusReadout";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Toolbar } from "@/components/Toolbar";

type LandingProps = {
  onEnter: () => void;
};

const CAPABILITIES = [
  {
    channel: "CH.01",
    label: "Time and space, together",
    body: "Most tools predict one. PolyO predicts both — a class and a calibrated confidence for each.",
  },
  {
    channel: "CH.02",
    label: "No LLM, no execution",
    body: "A graph neural network reads the code once, statically. Nothing runs, nothing gets sent to a language model.",
  },
  {
    channel: "CH.03",
    label: "Shows its work",
    body: "The exact code spans that drove the prediction are highlighted afterward — not just a number, a reason.",
  },
  {
    channel: "CH.04",
    label: "Six languages, one model",
    body: "Python, C++, Java, JavaScript, C, and Go all normalise into one intermediate representation first.",
  },
];

export function Landing({ onEnter }: LandingProps) {
  return (
    <main className="bench-shell">
      <Toolbar
        right={
          <>
            <StatusReadout status="ready" />
            <ThemeToggle />
          </>
        }
      />

      <section className="landing-hero bench-reveal-row" style={{ "--row-delay": "0ms" } as React.CSSProperties}>
        <p className="mono-nums landing-eyebrow">STATIC ANALYSIS BENCH</p>
        <h1 className="landing-title">
          Know the <span style={{ color: "var(--signal)" }}>Big-O</span> before you run it.
        </h1>
        <p className="landing-sub">
          Paste code and get its predicted worst-case time and space complexity — computed by a model that only
          ever reads the code.
        </p>
      </section>

      <section className="landing-grid">
        {CAPABILITIES.map((cap, i) => (
          <BenchPanel key={cap.channel} channel={cap.channel} revealDelayMs={80 + i * 70}>
            <div style={{ fontWeight: 600, marginBottom: 6, color: "var(--text-primary)" }}>{cap.label}</div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>{cap.body}</div>
          </BenchPanel>
        ))}
      </section>

      <div className="landing-cta-row bench-reveal-row" style={{ "--row-delay": "360ms" } as React.CSSProperties}>
        <button type="button" onClick={onEnter} className="mono-nums cta-button landing-cta vt-run-cta">
          RUN THE BENCH →
        </button>
      </div>
    </main>
  );
}
