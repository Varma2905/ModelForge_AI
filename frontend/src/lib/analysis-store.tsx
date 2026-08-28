import { createContext, useContext, useState, type ReactNode } from "react";
import type { ChartData, DataSplitConfig, ModelMetrics, StatisticalAnalysis } from "./api-types";

export type Dataset = {
  name: string;
  columns: string[];
  rows: (string | number | null)[][];
  // The dataset's true total row count, from the backend's row_count/total_rows
  // (not `rows.length` — `rows` here may be a preview-capped slice, see
  // new.upload.tsx). Training itself always uses the full dataset server-side
  // regardless of this value; this is display-only.
  totalRows: number;
};

export type AnalysisState = {
  // Real backend identifiers threaded through the wizard — populated as each
  // step completes, consumed by the steps that follow.
  datasetId: string | null;
  preprocessedDatasetId: string | null;
  modelId: string | null;

  dataset: Dataset | null;
  datasetStats: { missingValues: number; dataTypes: Record<string, string> } | null;
  features: string[];
  target: string | null;
  // Set from the Variables step's /datasets/{id}/profile lookup — which of
  // the selected `features` are numerical vs. categorical, so later steps
  // don't have to re-derive it.
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
  // Actual rows the training run used, as reported back by the backend after
  // train_test_split — not derived from any client-side/preview-capped array.
  trainedRowCounts: { total: number; train: number; test: number; val: number } | null;
  results: Partial<ModelMetrics> & Record<string, number | null | undefined>;
  statisticalAnalysis: StatisticalAnalysis | null;
  chartData: ChartData | null;
  prediction: number | null;
};

const defaultState: AnalysisState = {
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
  metrics: ["MSE", "RMSE", "R2"],
  trained: false,
  trainingTimeMs: 0,
  trainedRowCounts: null,
  results: {},
  statisticalAnalysis: null,
  chartData: null,
  prediction: null,
};

// Converts the wizard's percentage-based split UI state into the fraction-based
// shape the backend's DataSplitConfig expects. `test` is derived from
// train/val rather than trusted directly — the split UI only ever adjusts
// the train and val sliders, so a stored `test` value can go stale.
export function toBackendSplitConfig(split: AnalysisState["split"]): DataSplitConfig {
  const valPct = split.useVal ? split.val : 0;
  const testPct = 100 - split.train - valPct;
  return {
    test_size: Math.max(0.05, Math.min(0.95, testPct / 100)),
    val_size: split.useVal ? Math.max(0, Math.min(0.45, valPct / 100)) : 0,
    random_state: split.randomState,
  };
}

type Ctx = {
  state: AnalysisState;
  update: (patch: Partial<AnalysisState>) => void;
  reset: () => void;
};

const AnalysisCtx = createContext<Ctx | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AnalysisState>(defaultState);
  const update = (patch: Partial<AnalysisState>) => setState((s) => ({ ...s, ...patch }));
  const reset = () => setState(defaultState);
  return <AnalysisCtx.Provider value={{ state, update, reset }}>{children}</AnalysisCtx.Provider>;
}

export function useAnalysis() {
  const ctx = useContext(AnalysisCtx);
  if (!ctx) throw new Error("useAnalysis must be inside AnalysisProvider");
  return ctx;
}
