"use client";

// The step list under the code: how the static engine arrived at the cost (one row per loop, costly
// call, allocation or recursion, then the totals) and every bound it had to assume. Pointing at a
// row lights the same line in the code gutter, and the other way round, through `activeLine`.
// Renders nothing for a model answer, which has no derivation.

import { totalSteps } from "@/lib/derivation";
import type { AssumptionItem, DerivationStep } from "@/lib/types";

type DerivationPanelProps = {
  derivation: DerivationStep[] | undefined;
  assumptions: AssumptionItem[] | undefined;
  entry?: string | null;
  activeLine: number | null;
  onActiveLineChange: (line: number | null) => void;
};

function LineLink({
  line,
  activeLine,
  onActiveLineChange,
  children,
}: {
  line: number;
  activeLine: number | null;
  onActiveLineChange: (line: number | null) => void;
  children: React.ReactNode;
}) {
  if (line <= 0) return <li className="derivation-row">{children}</li>;
  return (
    <li
      className={`derivation-row is-linked${activeLine === line ? " is-active" : ""}`}
      tabIndex={0}
      onMouseEnter={() => onActiveLineChange(line)}
      onMouseLeave={() => onActiveLineChange(null)}
      onFocus={() => onActiveLineChange(line)}
      onBlur={() => onActiveLineChange(null)}
    >
      {children}
    </li>
  );
}

export function DerivationPanel({ derivation, assumptions, entry, activeLine, onActiveLineChange }: DerivationPanelProps) {
  const steps = (derivation ?? []).filter((step) => step.kind !== "total");
  const totals = totalSteps(derivation);
  const assumed = assumptions ?? [];
  if (steps.length === 0 && totals.length === 0 && assumed.length === 0) return null;

  return (
    <details open className="derivation-panel">
      <summary className="mono-nums field-label derivation-summary">
        HOW THE COST WAS DERIVED{entry ? ` — ${entry}` : ""}
      </summary>

      {steps.length > 0 ? (
        <ol className="derivation-list" aria-label="Derivation steps">
          {steps.map((step, i) => (
            <LineLink key={`${step.line}-${i}`} line={step.line} activeLine={activeLine} onActiveLineChange={onActiveLineChange}>
              <span className="mono-nums derivation-line">L{step.line}</span>
              <span className="mono-nums derivation-kind">{step.kind.toUpperCase()}</span>
              <span className="derivation-text">{step.text}</span>
            </LineLink>
          ))}
        </ol>
      ) : null}

      {totals.length > 0 ? (
        <ul className="derivation-list derivation-totals" aria-label="Totals">
          {totals.map((step, i) => (
            <li key={i} className="derivation-row">
              <span className="mono-nums derivation-text">{step.text}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {assumed.length > 0 ? (
        <>
          <div className="mono-nums field-label derivation-subhead">ASSUMPTIONS — the answer may over-estimate</div>
          <ul className="derivation-list" aria-label="Assumptions">
            {assumed.map((a, i) => (
              <LineLink key={`${a.line}-${i}`} line={a.line} activeLine={activeLine} onActiveLineChange={onActiveLineChange}>
                <span className="mono-nums derivation-line">{a.line > 0 ? `L${a.line}` : "—"}</span>
                <span className="mono-nums derivation-kind derivation-assumed">ASSUMED</span>
                <span className="derivation-text">{a.reason}</span>
              </LineLink>
            ))}
          </ul>
        </>
      ) : null}
    </details>
  );
}
