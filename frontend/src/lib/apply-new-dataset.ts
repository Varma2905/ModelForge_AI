import type { AnalysisState } from "@/lib/analysis-store";

/**
 * A new dataset — from any source — invalidates everything downstream in
 * the wizard (preprocessing, features, training, results). All source
 * panels (local upload, manual creation, Kaggle import, Google Drive
 * import) funnel through this same reset so the wizard behaves identically
 * regardless of where the dataset came from.
 */
export function applyNewDataset(
  update: (patch: Partial<AnalysisState>) => void,
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
    preprocessedDatasetId: null,
    modelId: null,
    dataset: { name: args.name, columns: [], rows: [], totalRows: args.totalRows },
    datasetStats: { missingValues: args.missingValues, dataTypes: args.dataTypes },
    features: [],
    target: null,
    preprocessingSummary: null,
    trained: false,
    trainedRowCounts: null,
    results: {},
    statisticalAnalysis: null,
    chartData: null,
    prediction: null,
  });
}
