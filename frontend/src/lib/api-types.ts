// TypeScript types mirroring the backend's Pydantic models and response envelope.

export type ApiEnvelope<T> =
  | { success: true; data: T; message?: string | null }
  | { success: false; error: { code: number; message: string } };

// --- Auth ---
export type AuthUser = {
  id: string;
  name: string;
  email: string;
  created_at: string;
};

export type AuthPayload = {
  user: AuthUser;
  token: string;
};

// --- Datasets ---
export type DatasetSourceType = "local" | "google_drive" | "kaggle" | "manual";

export type DatasetSummary = {
  dataset_id: string;
  dataset_name: string;
  rows: number;
  columns: number;
  column_names: string[];
  data_types: Record<string, string>;
  missing_values: number;
  source?: DatasetSourceType | null;
};

export type DatasetPreview = {
  name: string;
  columns: string[];
  rows: (string | number | null)[][];
  total_rows: number;
  total_columns: number;
  data_types: Record<string, string>;
  missing_values: number;
  source?: DatasetSourceType | null;
};

export type DatasetListItem = {
  dataset_id: string;
  name: string;
  rows: number;
  columns: number;
  source?: DatasetSourceType | null;
  created_at?: string | null;
  size_bytes?: number | null;
};

// --- Dataset profile (feature classification) ---
export type ColumnKind = "numerical" | "categorical" | "datetime";

export type ColumnProfile = {
  name: string;
  kind: ColumnKind;
  missing_count: number;
  unique_count: number;
  high_cardinality: boolean;
  is_identifier?: boolean;
  // True for a genuinely integer-valued numerical column (including an int
  // column pandas upcast to float64 due to NaNs) — used to decide whether a
  // "numerical" column could plausibly be an integer-encoded classification
  // target (0/1/2) rather than a continuous regression target.
  is_integer_like: boolean;
};

// --- Model type discriminator ---
export type ModelType = "regression" | "classification" | "clustering";

export type DatasetProfile = {
  dataset_id: string;
  name: string;
  rows: number;
  columns: number;
  features: ColumnProfile[];
};

// --- Dataset Sources (Kaggle / Google Drive) ---
export type DatasetSourcesStatus = {
  google_drive_configured: boolean;
};

// Kaggle has no third-party OAuth — each platform user connects their own
// Kaggle account (username + personal API key), so its availability is a
// per-user connection state, not a server-wide "configured" flag like
// Google Drive.
export type KaggleStatus = { connected: boolean; kaggle_username: string | null };

export type KaggleFile = { name: string; size: number };

export type KaggleResolveResult = { dataset_ref: string; files: KaggleFile[] };

export type GoogleDriveFile = { id: string; name: string; size?: number; mimeType: string };

export type DatasetModelListItem = {
  model_id: string;
  model: string;
  // Absent on a model doc trained before this field existed — every
  // consumer must treat a missing value as "regression".
  model_type?: ModelType;
  metrics: ModelMetrics;
  created_at: string;
  source?: "huggingface" | null;
  hf_model_id?: string | null;
};

// --- Preprocessing ---
export type PreprocessConfig = {
  missing: "remove" | "mean" | "median" | "mode" | "none";
  dedupe: boolean;
  outlier: "iqr" | "zscore" | "none";
  scaling: "standard" | "minmax" | "none";
};

export type PreprocessSummary = {
  initial_rows: number;
  final_rows: number;
  missing_imputed: number;
  duplicates_removed: number;
  outliers_removed: number;
  scaling_deferred_to_training: string;
};

export type PreprocessResult = {
  preprocessed_dataset_id: string;
  summary: PreprocessSummary;
};

export type SelectFeaturesResult = {
  status: string;
  features: string[];
  target: string;
  numerical_features: string[];
  categorical_features: string[];
  // High-cardinality (likely-identifier) warnings — informational only,
  // never causes automatic exclusion of the flagged column.
  warnings: string[];
};

// --- Split ---
export type DataSplitConfig = {
  test_size: number;
  val_size: number;
  random_state: number;
};

export type SplitPreview = {
  train_rows: number;
  test_rows: number;
  val_rows: number;
  random_state: number;
};

// --- Training ---
export type ModelMetrics = {
  MSE: number;
  MAE: number;
  RMSE: number;
  R2: number;
  "Adjusted R2": number;
  // Each of these is null whenever it's mathematically undefined for the
  // given actual/predicted values (division by zero, a negative value under
  // a log, a constant array, ...) rather than NaN/Infinity — see backend
  // calculate_evaluation_metrics for each metric's specific validity condition.
  MAPE?: number | null;
  MSLE?: number | null;
  RMSLE?: number | null;
  "Median Absolute Error"?: number | null;
  "Max Error"?: number | null;
  "Explained Variance"?: number | null;
  "Pearson Correlation"?: number | null;
  "Spearman Correlation"?: number | null;
};

export type StatisticalAnalysis = {
  coefficients: Record<string, number>;
  p_values: Record<string, number>;
  standard_errors: Record<string, number>;
  t_statistics: Record<string, number>;
  f_statistic: number;
  f_pvalue: number;
};

export type ChartData = {
  actual: number[];
  predicted: number[];
  residuals: number[];
  // Each entry's "feature" is a display label — a plain column name for
  // numerical features, or "column = category" for a one-hot-expanded
  // categorical level (see backend map_expanded_coefficients).
  feature_importance: { feature: string; source_feature: string; value: number }[];
  correlation_matrix: { columns: string[]; matrix: number[][] };
};

export type TrainModelResult = {
  status: string;
  model: string;
  model_id: string;
  metrics: ModelMetrics;
  total_rows: number;
  train_rows: number;
  test_rows: number;
  val_rows: number;
};

export type ModelMetricsResult = {
  model_id: string;
  model: string;
  model_type?: ModelType;
  classes?: string[] | null;
  features: string[];
  numerical_features: string[];
  categorical_features: string[];
  // Values actually observed for each categorical feature in the training
  // split — populates the Predict page's per-feature dropdowns so a
  // prediction can never submit a category the model was never fit on.
  categorical_options: Record<string, string[]>;
  target: string;
  metrics: ModelMetrics;
  chart_data: ChartData;
  statistical_analysis: StatisticalAnalysis;
};

// Registry-backed catalog from GET /regression-models — see backend
// regression_registry.py. "available" means it can actually be trained right
// now; "unavailable" means it's implemented but its optional package isn't
// installed; "coming_soon" means no pipeline exists yet (the time-series
// models). Never treat an entry as trainable unless status === "available".
export type RegressionModelStatus = "available" | "unavailable" | "coming_soon";

export type RegressionModelEntry = {
  id: string;
  display_name: string;
  category: string;
  description: string;
  pros: string[];
  status: RegressionModelStatus;
  reason: string | null;
};

export type ModelListItem = {
  model_id: string;
  dataset_id: string;
  dataset_name: string;
  model: string;
  model_type?: ModelType;
  metrics: ModelMetrics;
  created_at: string;
  source?: "huggingface" | null;
  hf_model_id?: string | null;
};

// --- Classification ---
export type ClassificationSelectFeaturesResult = SelectFeaturesResult;

export type ClassificationMetrics = {
  Accuracy: number;
  Precision: number;
  Recall: number;
  F1: number;
  // null when it couldn't be computed (e.g. no predict_proba, or a class
  // missing from the test split) — see backend calculate_classification_metrics.
  ROC_AUC: number | null;
  ConfusionMatrix: number[][];
  Classes: string[];
  // Per-class precision/recall/f1-score/support, from sklearn's
  // classification_report(output_dict=True).
  ClassificationReport: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
  // Always computable from hard labels alone — present for both binary and
  // multiclass. MCC/LogLoss are null when undefined (LogLoss needs
  // predict_proba); see backend calculate_classification_metrics.
  BalancedAccuracy?: number | null;
  MCC?: number | null;
  LogLoss?: number | null;
  // Binary only — true negative rate.
  Specificity?: number | null;
  // Multiclass only — macro treats every class equally regardless of size;
  // weighted mirrors the plain Precision/Recall/F1 above.
  "Precision Macro"?: number | null;
  "Recall Macro"?: number | null;
  "F1 Macro"?: number | null;
  "Precision Weighted"?: number | null;
  "Recall Weighted"?: number | null;
  "F1 Weighted"?: number | null;
};

// Registry-backed catalog from GET /classify/classification-models — see
// backend classification_registry.py. "unavailable" means the model is
// implemented but its optional package (xgboost/lightgbm/catboost) isn't
// installed; never treat an entry as trainable unless status === "available".
export type ClassificationModelEntry = {
  id: string;
  display_name: string;
  description: string;
  pros: string[];
  status: "available" | "unavailable";
  reason: string | null;
};

// --- Clustering ---

export type ClusteringParameter = {
  name: string;
  label: string;
  default: number | string;
  type: "int" | "float" | "select";
  options?: string[];
};

// Registry-backed catalog from GET /cluster/clustering-models — see backend
// clustering_registry.py. No "coming_soon" tier — every listed algorithm has
// a real pipeline; "unavailable" means its optional package isn't installed.
export type ClusteringModelEntry = {
  id: string;
  display_name: string;
  algorithm_type: string;
  description: string;
  parameters: ClusteringParameter[];
  status: "available" | "unavailable";
  reason: string | null;
};

export type ClusteringSelectFeaturesResult = {
  status: string;
  features: string[];
  numerical_features: string[];
  categorical_features: string[];
  missing_counts: Record<string, number>;
  warnings: string[];
};

export type ClusteringMetrics = {
  ClusterCount: number;
  NoisePoints: number | null;
  // null when fewer than 2 real clusters exist (see backend
  // calculate_clustering_metrics) — never a fabricated value.
  Silhouette: number | null;
  CalinskiHarabasz: number | null;
  DaviesBouldin: number | null;
};

export type ClusterProfile = {
  cluster: string;
  samples: number;
  feature_means: Record<string, number | null>;
  feature_medians: Record<string, number | null>;
  categorical_modes: Record<string, string | null>;
  important_characteristics: string[];
};

export type ClusteringVisualizations = {
  pca: {
    x: number[];
    y: number[];
    z?: number[];
    labels: string[];
    explained_variance_2d: number[];
    explained_variance_3d?: number[];
  } | null;
  silhouette_plot: { by_cluster: Record<string, number[]> } | null;
  elbow: { k_values: number[]; values: number[]; metric_label: string } | null;
  dendrogram: { icoord: number[][]; dcoord: number[][]; sampled: boolean; sample_size: number; total_size: number } | null;
  cluster_sizes: { labels: string[]; values: number[] };
};

export type TrainClusteringModelResult = {
  status: string;
  model: string;
  model_id: string;
  metrics: ClusteringMetrics;
  cluster_sizes: Record<string, number>;
  total_rows: number;
};

export type ClusteringModelMetricsResult = {
  model_id: string;
  model: string;
  model_type: "clustering";
  algorithm_type: string | null;
  dataset_name: string;
  features: string[];
  numerical_features: string[];
  categorical_features: string[];
  hyperparameters: Record<string, unknown>;
  total_rows: number;
  metrics: ClusteringMetrics;
  cluster_sizes: Record<string, number>;
  cluster_profiles: ClusterProfile[];
  visualizations: ClusteringVisualizations;
  created_at: string;
};

// Same key shape as StatisticalAnalysis (see backend
// calculate_classification_statistical_properties) — a multiclass fit's
// coefficient keys carry a "__class_{k}" suffix, already folded into the
// display label by the shared map_expanded_coefficients() helper.
export type ClassificationStatisticalAnalysis = StatisticalAnalysis;

export type ClassificationChartData = {
  actual: string[];
  predicted: string[];
  feature_importance: { feature: string; source_feature: string; value: number }[];
  correlation_matrix: { columns: string[]; matrix: number[][] };
  confusion_matrix: { classes: string[]; matrix: number[][] };
};

// Dedicated data for the Classification Visualization dashboard (see backend
// classification_evaluation.py's extract_* helpers) — every field is
// independently optional/null when unsupported by the trained model or
// dataset (e.g. no predict_proba, a non-binary target, or a model with
// neither feature_importances_ nor coef_), never just absent from a crash.
export type ClassificationVisualizations = {
  class_distribution: { labels: string[]; values: number[] } | null;
  confusion_matrix: { labels: string[]; matrix: number[][] } | null;
  feature_importance: { features: string[]; importance: number[] } | null;
  roc_curve: { fpr: number[]; tpr: number[]; auc: number | null } | null;
  precision_recall_curve: { precision: number[]; recall: number[] } | null;
};

export type TrainClassificationModelResult = {
  status: string;
  model: string;
  model_id: string;
  metrics: ClassificationMetrics;
  total_rows: number;
  train_rows: number;
  test_rows: number;
  val_rows: number;
};

export type ClassificationModelMetricsResult = {
  model_id: string;
  model: string;
  model_type: "classification";
  classes: string[];
  features: string[];
  numerical_features: string[];
  categorical_features: string[];
  categorical_options: Record<string, string[]>;
  target: string;
  metrics: ClassificationMetrics;
  chart_data: ClassificationChartData;
  visualizations?: ClassificationVisualizations | null;
  statistical_analysis: ClassificationStatisticalAnalysis;
};

// --- Prediction ---
export type PredictionResult = {
  prediction: number;
  value: number;
};

export type ClassPredictionResult = {
  prediction: string;
  probabilities: Record<string, number> | null;
};

// A prediction payload can mix numeric feature values with categorical
// (string) ones — the backend coerces each value according to which
// features the model was actually trained on as numeric vs. categorical.
export type PredictionValues = Record<string, number | string>;

// --- AI ---
export type AIExplainResult = {
  summary: string;
  insights: string[];
  full_report: string;
};

export type ChatResult = {
  reply: string;
  source: "llm" | "fallback";
};

// --- AI Assistant (streaming chat) ---
export type AIChatRole = "user" | "assistant";

export type AIChatTurn = {
  role: AIChatRole;
  content: string;
};

export type AIChatStreamRequest = {
  messages: AIChatTurn[];
  model_id?: string | null;
  provider?: string;
  model?: string;
};

export type AIChatErrorType = "auth" | "rate_limit" | "network" | "server" | "config";

// --- Dashboard ---
export type DashboardActivityItem = {
  type: "dataset" | "analysis" | "report";
  title: string;
  subtitle: string;
  action: string;
  timestamp: string;
};

export type DashboardSummary = {
  total_datasets: number;
  total_analyses: number;
  total_reports: number;
  total_classification_analyses: number;
  total_clustering_analyses: number;
  best_model: { model_id: string; model: string; dataset_name: string; r2: number } | null;
  best_accuracy_model: { model_id: string; model: string; dataset_name: string; accuracy: number } | null;
  best_clustering_model: { model_id: string; model: string; dataset_name: string; silhouette: number } | null;
  avg_r2: number | null;
  avg_rmse: number | null;
  avg_accuracy: number | null;
  avg_silhouette: number | null;
  recent_analyses: {
    model_id: string;
    dataset_name: string;
    model: string;
    model_type?: ModelType;
    r2: number | null;
    accuracy: number | null;
    silhouette?: number | null;
    created_at: string;
  }[];
  top_regression_models: {
    model_id: string;
    model: string;
    dataset_name: string;
    r2: number | null;
    rmse: number | null;
  }[];
  top_classification_models: {
    model_id: string;
    model: string;
    dataset_name: string;
    accuracy: number | null;
    f1_score: number | null;
  }[];
  top_clustering_models: {
    model_id: string;
    model: string;
    dataset_name: string;
    silhouette: number | null;
    davies_bouldin: number | null;
    calinski_harabasz: number | null;
    cluster_count: number | null;
  }[];
  trend_data: {
    date: string;
    regression: number;
    classification: number;
    clustering: number;
  }[];
  datasets_sparkline: number[];
  regression_sparkline: number[];
  classification_sparkline: number[];
  clustering_sparkline: number[];
  reports_sparkline: number[];
  r2_sparkline: number[];
  accuracy_sparkline: number[];
  silhouette_sparkline: number[];
  // Percent change vs. the prior 30-day period; null when there's no prior
  // data to compare against (not enough history yet).
  trend_datasets_pct: number | null;
  trend_analyses_pct: number | null;
  trend_reports_pct: number | null;
  // Absolute point change vs. the prior period (neither metric is
  // meaningfully a percent-of-itself change).
  avg_r2_delta: number | null;
  avg_accuracy_delta: number | null;
  avg_silhouette_delta: number | null;
  recent_activity: DashboardActivityItem[];
};

// --- Graphs ---
export type GraphsResult = {
  model_id: string;
  graphs: Record<string, string>;
};

// --- Hugging Face Models ---
export type HFModelSummary = {
  model_id: string;
  author: string | null;
  pipeline_tag: string | null;
  library_name: string | null;
  tags: string[];
  downloads: number | null;
  likes: number | null;
  last_modified: string | null;
};

export type HFModelDetails = HFModelSummary & {
  license: string | null;
  files: string[];
  gated: boolean;
};

export type HFExecutionMode = "inference_only" | "fine_tune_required" | "unsupported";

export type HFCompatibilityResult = {
  model_id: string;
  dataset_id: string;
  compatible: boolean;
  execution_mode: HFExecutionMode;
  confidence: "high" | "medium" | "low";
  reasons: string[];
  artifact_file: string | null;
};

export type HFRunModelRequest = {
  dataset_id: string;
  features: string[];
  target: string;
};

export type HFRunModelResult = {
  status: string;
  model_id: string;
  hf_model_id: string;
  source: "huggingface";
  metrics: ModelMetrics;
  version_warning: string | null;
};
