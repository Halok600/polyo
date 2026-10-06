// Mirrors api/main.py's Pydantic response models (plan §10) -- kept as a
// hand-written type, not generated, since the API is small and stable
// enough that a codegen step would be more machinery than value here.

export type SupportedLanguage = "python" | "cpp" | "java" | "javascript" | "c" | "go";

export type LanguageOption = {
  id: SupportedLanguage;
  tier: 1 | 2;
};

export type ClassPrediction = {
  class: string;
  rank: number;
  confidence: number;
  distribution: Record<string, number>;
  // Split-conformal prediction set: the smallest ordinally contiguous run
  // of classes guaranteed (marginally, distribution-free) to contain the
  // true class conformal_coverage of the time -- see models/conformal.py.
  conformal_set: string[];
  conformal_coverage: number;
  abstain: boolean;
  // --- v2 (all optional: a response from before the symbolic engine omits them) ---
  // Who answered: the static cost engine, the GNN because the engine could not bound the code, or
  // the GNN alone.
  engine?: "symbolic" | "ml_fallback" | "ml";
  // Only for a symbolic answer. "assumed" means a bound was assumed and the answer may over-estimate.
  certainty?: "certain" | "assumed" | null;
  // The exact cost expression, e.g. "O(n * m)". `class` is that expression rounded UP to the
  // 7-class (time) / 5-class (space) scale when projection_lossy is true.
  expression?: string | null;
  extended_class?: string | null;
  projection_lossy?: boolean;
};

export type AssumptionItem = {
  line: number; // 1-based, 0 when unknown
  reason: string;
};

export type DerivationStep = {
  line: number; // 1-based; 0 for the closing "time" / "space" totals
  kind: "loop" | "call" | "recursion" | "alloc" | "total" | string;
  text: string;
};

export type AttributionItem = {
  feature: string;
  contribution: number;
  spans: number[][];
};

export type CurveDimension = {
  predicted_class: string;
  series: Record<string, number[]>;
};

export type Curve = {
  n: number[];
  time: CurveDimension;
  space: CurveDimension;
};

export type IrSummary = {
  nodes: number;
  edges: number;
  histogram: Record<string, number>;
};

export type PredictResponse = {
  language_detected: string;
  time: ClassPrediction;
  space: ClassPrediction;
  attribution: AttributionItem[];
  curve: Curve;
  ir: IrSummary;
  warnings: string[];
  // --- v2 (optional, as above) ---
  engine?: "symbolic" | "ml_fallback" | "ml";
  entry?: string | null;
  assumptions?: AssumptionItem[];
  derivation?: DerivationStep[];
};

export type PredictErrorBody = {
  detail: string;
};
