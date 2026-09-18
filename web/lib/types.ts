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
};

export type PredictErrorBody = {
  detail: string;
};
