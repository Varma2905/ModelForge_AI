import { useCallback, useEffect, useState } from "react";

// Single structured localStorage key for every ModelForge AI Studio
// preference added by the Settings page — see section 8 of the settings
// spec. These are display/behavior PREFERENCES only: nothing here changes
// the ML engine, training pipeline, or report-generation engine itself
// (see the "supported" / "preference only" notes on each section below).
// Never store secrets here (API keys, tokens, passwords) — see Backend/
// Account tabs, which read/write those through their own existing
// mechanisms, not this store.
const STORAGE_KEY = "modelforge_settings";

export type MlType = "regression" | "classification";
export type SplitRatio = "70/30" | "75/25" | "80/20" | "85/15" | "90/10";
export type CrossValidation = "disabled" | "3" | "5" | "10";

export type MlPreferences = {
  defaultMlType: MlType;
  defaultSplit: SplitRatio;
  randomSeed: number;
  crossValidation: CrossValidation;
  autoModelSelection: boolean;
};

// Per-model enable/disable overrides, keyed by the model's registry `id`
// (the exact string the training APIs expect — see backend
// regression_registry.py / classification_registry.py). A model absent
// from these maps is treated as enabled by default; only registry entries
// with status "available" can ever be turned on (see ModelPreferencesTab).
export type ModelPreferences = {
  regression: Record<string, boolean>;
  classification: Record<string, boolean>;
};

export type RegressionMetricKey =
  | "MAE"
  | "MSE"
  | "RMSE"
  | "MAPE"
  | "MSLE"
  | "RMSLE"
  | "Median Absolute Error"
  | "Max Error"
  | "R2"
  | "Adjusted R2"
  | "Explained Variance"
  | "Pearson Correlation"
  | "Spearman Correlation";

export type ClassificationMetricKey = "Accuracy" | "Precision" | "Recall" | "F1" | "ConfusionMatrix" | "ROC_AUC";

export type PrimaryRegressionMetric = "R2" | "RMSE" | "MAE";
export type PrimaryClassificationMetric = "Accuracy" | "F1" | "Precision" | "Recall";

export type MetricsPreferences = {
  regressionMetrics: RegressionMetricKey[];
  classificationMetrics: ClassificationMetricKey[];
  primaryRegressionMetric: PrimaryRegressionMetric;
  primaryClassificationMetric: PrimaryClassificationMetric;
};

export type MaxCharts = "4" | "6" | "8" | "10" | "unlimited";
export type RegressionPlotKey =
  | "Actual vs Predicted"
  | "Residual Plot"
  | "Feature Importance"
  | "Prediction Error Distribution"
  | "Correlation Heatmap";
export type ClassificationPlotKey =
  | "Confusion Matrix"
  | "ROC Curve"
  | "Precision-Recall Curve"
  | "Feature Importance"
  | "Class Distribution";

export type VisualizationPreferences = {
  autoGenerate: boolean;
  interactiveCharts: boolean;
  maxCharts: MaxCharts;
  regressionPlots: RegressionPlotKey[];
  classificationPlots: ClassificationPlotKey[];
};

export type ReportPreferences = {
  includeDatasetSummary: boolean;
  includeDataQuality: boolean;
  includePreprocessing: boolean;
  includeModelInfo: boolean;
  includeModelEquation: boolean;
  includeErrorMetrics: boolean;
  includeStatisticalMetrics: boolean;
  includeVisualizationPlots: boolean;
  includeAiInsights: boolean;
  includeFeatureImportance: boolean;
};

export type ModelForgeSettings = {
  mlPreferences: MlPreferences;
  modelPreferences: ModelPreferences;
  metrics: MetricsPreferences;
  visualization: VisualizationPreferences;
  reportPreferences: ReportPreferences;
};

export const DEFAULT_ML_PREFERENCES: MlPreferences = {
  defaultMlType: "regression",
  defaultSplit: "80/20",
  randomSeed: 42,
  crossValidation: "5",
  autoModelSelection: true,
};

export const DEFAULT_MODEL_PREFERENCES: ModelPreferences = {
  regression: {},
  classification: {},
};

export const ALL_REGRESSION_METRICS: RegressionMetricKey[] = [
  "MAE",
  "MSE",
  "RMSE",
  "MAPE",
  "MSLE",
  "RMSLE",
  "Median Absolute Error",
  "Max Error",
  "R2",
  "Adjusted R2",
  "Explained Variance",
  "Pearson Correlation",
  "Spearman Correlation",
];

export const ALL_CLASSIFICATION_METRICS: ClassificationMetricKey[] = [
  "Accuracy",
  "Precision",
  "Recall",
  "F1",
  "ConfusionMatrix",
  "ROC_AUC",
];

export const DEFAULT_METRICS: MetricsPreferences = {
  regressionMetrics: ALL_REGRESSION_METRICS,
  classificationMetrics: ALL_CLASSIFICATION_METRICS,
  primaryRegressionMetric: "R2",
  primaryClassificationMetric: "Accuracy",
};

export const ALL_REGRESSION_PLOTS: RegressionPlotKey[] = [
  "Actual vs Predicted",
  "Residual Plot",
  "Feature Importance",
  "Prediction Error Distribution",
  "Correlation Heatmap",
];

export const ALL_CLASSIFICATION_PLOTS: ClassificationPlotKey[] = [
  "Confusion Matrix",
  "ROC Curve",
  "Precision-Recall Curve",
  "Feature Importance",
  "Class Distribution",
];

export const DEFAULT_VISUALIZATION: VisualizationPreferences = {
  autoGenerate: true,
  interactiveCharts: true,
  maxCharts: "6",
  regressionPlots: ALL_REGRESSION_PLOTS,
  classificationPlots: ALL_CLASSIFICATION_PLOTS,
};

export const DEFAULT_REPORT_PREFERENCES: ReportPreferences = {
  includeDatasetSummary: true,
  includeDataQuality: true,
  includePreprocessing: true,
  includeModelInfo: true,
  includeModelEquation: true,
  includeErrorMetrics: true,
  includeStatisticalMetrics: true,
  includeVisualizationPlots: true,
  includeAiInsights: true,
  includeFeatureImportance: true,
};

export const DEFAULT_SETTINGS: ModelForgeSettings = {
  mlPreferences: DEFAULT_ML_PREFERENCES,
  modelPreferences: DEFAULT_MODEL_PREFERENCES,
  metrics: DEFAULT_METRICS,
  visualization: DEFAULT_VISUALIZATION,
  reportPreferences: DEFAULT_REPORT_PREFERENCES,
};

function loadSettings(): ModelForgeSettings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<ModelForgeSettings>;
    // Shallow-merge each section over its defaults, so a stored object from
    // before a new field existed never crashes or silently loses it.
    return {
      mlPreferences: { ...DEFAULT_ML_PREFERENCES, ...parsed.mlPreferences },
      modelPreferences: {
        regression: { ...DEFAULT_MODEL_PREFERENCES.regression, ...parsed.modelPreferences?.regression },
        classification: { ...DEFAULT_MODEL_PREFERENCES.classification, ...parsed.modelPreferences?.classification },
      },
      metrics: { ...DEFAULT_METRICS, ...parsed.metrics },
      visualization: { ...DEFAULT_VISUALIZATION, ...parsed.visualization },
      reportPreferences: { ...DEFAULT_REPORT_PREFERENCES, ...parsed.reportPreferences },
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function persistSettings(settings: ModelForgeSettings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // storage full/unavailable — preferences degrade to in-memory only for this session
  }
}

export function useModelForgeSettings() {
  const [settings, setSettings] = useState<ModelForgeSettings>(DEFAULT_SETTINGS);

  // Read persisted settings after hydration, matching lib/theme.tsx's
  // SSR-safe pattern — localStorage doesn't exist during server render.
  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  const updateSection = useCallback(
    <K extends keyof ModelForgeSettings>(key: K, value: ModelForgeSettings[K]) => {
      setSettings((prev) => {
        const next = { ...prev, [key]: value };
        persistSettings(next);
        return next;
      });
    },
    [],
  );

  const resetSection = useCallback(
    <K extends keyof ModelForgeSettings>(key: K) => {
      updateSection(key, DEFAULT_SETTINGS[key]);
    },
    [updateSection],
  );

  return { settings, updateSection, resetSection };
}
