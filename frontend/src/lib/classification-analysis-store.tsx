import { createContext, useContext, useState, type ReactNode } from "react";
import type { ClassificationChartData, ClassificationMetrics, ClassificationStatisticalAnalysis, DataSplitConfig } from "./api-types";
import type { Dataset } from "./analysis-store";
import { toBackendSplitConfig } from "./analysis-store";

// Parallel to analysis-store.tsx's AnalysisState — kept as a genuinely
// separate Context rather than adding a taskType branch to the regression
// store, since several fields have fundamentally different shapes here
// (prediction is a class label + probabilities, not a single number;
// results/statisticalAnalysis/chartData are classification-shaped) and the
// regression wizard's existing pages are hard-typed against the regression
// shapes already — forking the store avoids touching any of that.
export type ClassificationAnalysisState = {
  datasetId: string | null;
  preprocessedDatasetId: string | null;
  modelId: string | null;

  dataset: Dataset | null;
  datasetStats: { missingValues: number; dataTypes: Record<string, string> } | null;
  features: string[];
  target: string | null;
  featureKinds: Record<string, "numerical" | "categorical"> | null;
  preprocessing: {
    missing: string;
    dedupe: boolean;
    outlier: string;
    scaling: string;
  };
  preprocessingSummary: {
    initial_rows: number;
    final_rows: number;
    missing_imputed: number;
    duplicates_removed: number;
    outliers_removed: number;
  } | null;
  split: { train: number; test: number; val: number; useVal: boolean; randomState: number };
  model: string | null;
  metrics: string[];
  trained: boolean;
  trainingTimeMs: number;
  trainedRowCounts: { total: number; train: number; test: number; val: number } | null;
  results: Partial<ClassificationMetrics> & Record<string, unknown>;
  statisticalAnalysis: ClassificationStatisticalAnalysis | null;
  chartData: ClassificationChartData | null;
  classes: string[] | null;
  prediction: { label: string; probabilities: Record<string, number> | null } | null;
};

const defaultState: ClassificationAnalysisState = {
  datasetId: null,
  preprocessedDatasetId: null,
  modelId: null,

  dataset: null,
  datasetStats: null,
  features: [],
  target: null,
  featureKinds: null,
  preprocessing: {
    missing: "mean",
    dedupe: true,
    outlier: "iqr",
    scaling: "standard",
  },
  preprocessingSummary: null,
  split: { train: 80, test: 20, val: 0, useVal: false, randomState: 42 },
  model: null,
  metrics: ["Accuracy", "Precision", "Recall", "F1"],
  trained: false,
  trainingTimeMs: 0,
  trainedRowCounts: null,
  results: {},
  statisticalAnalysis: null,
  chartData: null,
  classes: null,
  prediction: null,
};

export { toBackendSplitConfig };
export type { DataSplitConfig };

type Ctx = {
  state: ClassificationAnalysisState;
  update: (patch: Partial<ClassificationAnalysisState>) => void;
  reset: () => void;
};

const ClassificationAnalysisCtx = createContext<Ctx | null>(null);

export function ClassificationAnalysisProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ClassificationAnalysisState>(defaultState);
  const update = (patch: Partial<ClassificationAnalysisState>) => setState((s) => ({ ...s, ...patch }));
  const reset = () => setState(defaultState);
  return (
    <ClassificationAnalysisCtx.Provider value={{ state, update, reset }}>
      {children}
    </ClassificationAnalysisCtx.Provider>
  );
}

export function useClassificationAnalysis() {
  const ctx = useContext(ClassificationAnalysisCtx);
  if (!ctx) throw new Error("useClassificationAnalysis must be inside ClassificationAnalysisProvider");
  return ctx;
}
