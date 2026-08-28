import { createContext, useContext, useState, type ReactNode } from "react";
import type { ClusteringMetrics } from "./api-types";
import type { Dataset } from "./analysis-store";

// Clustering counterpart to analysis-store.tsx / classification-analysis-store.tsx
// — kept as a genuinely separate Context rather than a taskType branch on
// either existing store, since clustering has no target/split/predict
// concept at all (it's unsupervised: Dataset -> Features -> Algorithm ->
// Clustering -> Analysis -> Insights -> Report, see cluster.*.tsx routes).
export type ClusteringAnalysisState = {
  datasetId: string | null;
  modelId: string | null;

  dataset: Dataset | null;
  datasetStats: { missingValues: number; dataTypes: Record<string, string> } | null;
  features: string[];
  featureKinds: Record<string, "numerical" | "categorical"> | null;
  model: string | null;
  hyperparameters: Record<string, unknown>;
  trained: boolean;
  trainingTimeMs: number;
  results: Partial<ClusteringMetrics> & Record<string, unknown>;
};

const defaultState: ClusteringAnalysisState = {
  datasetId: null,
  modelId: null,

  dataset: null,
  datasetStats: null,
  features: [],
  featureKinds: null,
  model: null,
  hyperparameters: {},
  trained: false,
  trainingTimeMs: 0,
  results: {},
};

type Ctx = {
  state: ClusteringAnalysisState;
  update: (patch: Partial<ClusteringAnalysisState>) => void;
  reset: () => void;
};

const ClusteringAnalysisCtx = createContext<Ctx | null>(null);

export function ClusteringAnalysisProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ClusteringAnalysisState>(defaultState);
  const update = (patch: Partial<ClusteringAnalysisState>) => setState((s) => ({ ...s, ...patch }));
  const reset = () => setState(defaultState);
  return (
    <ClusteringAnalysisCtx.Provider value={{ state, update, reset }}>
      {children}
    </ClusteringAnalysisCtx.Provider>
  );
}

export function useClusteringAnalysis() {
  const ctx = useContext(ClusteringAnalysisCtx);
  if (!ctx) throw new Error("useClusteringAnalysis must be inside ClusteringAnalysisProvider");
  return ctx;
}
