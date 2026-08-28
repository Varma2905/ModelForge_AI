import type { ClusteringAnalysisState } from "@/lib/clustering-analysis-store";

/**
 * Clustering counterpart to apply-new-dataset.ts / apply-new-classification-dataset.ts
 * — same "a new dataset invalidates everything downstream" reset, typed
 * against ClusteringAnalysisState (no target/split/prediction fields to reset,
 * since clustering is unsupervised).
 */
export function applyNewClusteringDataset(
  update: (patch: Partial<ClusteringAnalysisState>) => void,
  args: {
    datasetId: string;
    name: string;
    totalRows: number;
    missingValues: number;
    dataTypes: Record<string, string>;
  },
) {
  update({
    datasetId: args.datasetId,
    modelId: null,
    dataset: { name: args.name, columns: [], rows: [], totalRows: args.totalRows },
    datasetStats: { missingValues: args.missingValues, dataTypes: args.dataTypes },
    features: [],
    model: null,
    hyperparameters: {},
    trained: false,
    trainingTimeMs: 0,
    results: {},
  });
}
