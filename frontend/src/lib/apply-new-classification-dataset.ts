import type { ClassificationAnalysisState } from "@/lib/classification-analysis-store";

/**
 * Classification counterpart to apply-new-dataset.ts — same "a new dataset
 * invalidates everything downstream" reset, typed against
 * ClassificationAnalysisState instead of the regression AnalysisState
 * (their prediction/results/chartData/statisticalAnalysis shapes differ).
 */
export function applyNewClassificationDataset(
  update: (patch: Partial<ClassificationAnalysisState>) => void,
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
    classes: null,
    prediction: null,
  });
}
