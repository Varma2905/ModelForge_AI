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
export type DatasetSummary = {
  dataset_id: string;
  dataset_name: string;
  rows: number;
  columns: number;
  column_names: string[];
  data_types: Record<string, string>;
  missing_values: number;
};

export type DatasetPreview = {
  name: string;
  columns: string[];
  rows: (string | number | null)[][];
  total_rows: number;
  total_columns: number;
  data_types: Record<string, string>;
  missing_values: number;
};

export type DatasetListItem = {
  dataset_id: string;
  name: string;
  rows: number;
  columns: number;
};

export type DatasetModelListItem = {
  model_id: string;
  model: string;
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
  feature_importance: { feature: string; value: number }[];
  correlation_matrix: { columns: string[]; matrix: number[][] };
};

export type TrainModelResult = {
  status: string;
  model: string;
  model_id: string;
  metrics: ModelMetrics;
};

export type ModelMetricsResult = {
  model_id: string;
  model: string;
  features: string[];
  target: string;
  metrics: ModelMetrics;
  chart_data: ChartData;
  statistical_analysis: StatisticalAnalysis;
};

export type ModelListItem = {
  model_id: string;
  dataset_id: string;
  dataset_name: string;
  model: string;
  metrics: ModelMetrics;
  created_at: string;
  source?: "huggingface" | null;
  hf_model_id?: string | null;
};

// --- Prediction ---
export type PredictionResult = {
  prediction: number;
  value: number;
};

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
export type DashboardSummary = {
  total_datasets: number;
  total_analyses: number;
  total_reports: number;
  best_model: { model_id: string; model: string; dataset_name: string; r2: number } | null;
  avg_r2: number | null;
  avg_rmse: number | null;
  recent_analyses: {
    model_id: string;
    dataset_name: string;
    model: string;
    r2: number | null;
    created_at: string;
  }[];
};

// --- Graphs ---
export type GraphsResult = {
  model_id: string;
  graphs: Record<string, string>;
};

// --- Data Sources (database connectors) ---
export type DataSourceEngine = "mysql" | "postgresql" | "mongodb";

export type SQLConnectionInput = {
  engine: "mysql" | "postgresql";
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl_enabled?: boolean;
  connection_timeout_sec?: number;
};

export type MongoConnectionInput = {
  engine: "mongodb";
  uri: string;
  database: string;
  connection_timeout_sec?: number;
};

export type ConnectionInput = SQLConnectionInput | MongoConnectionInput;

// Never includes a password/URI — the backend's response model omits it entirely.
export type DataSourceSummary = {
  id: string;
  name: string;
  engine: DataSourceEngine;
  host: string | null;
  port: number | null;
  database: string;
  username: string | null;
  ssl_enabled: boolean;
  created_at: string;
  updated_at: string;
  last_tested_at: string | null;
  last_test_status: "ok" | "failed" | null;
};

export type TestConnectionResult = {
  status: "ok";
  server_version: string | null;
};

export type ColumnSchema = {
  name: string;
  type: string;
  nullable: boolean;
  is_pk: boolean;
};

export type TableSchema = {
  name: string;
  row_count_estimate: number | null;
  columns: ColumnSchema[];
};

export type TablesResult = {
  tables: TableSchema[];
};

export type ColumnStat = {
  column: string;
  dtype: string;
  null_count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
};

export type DataSourcePreview = {
  columns: string[];
  rows: (string | number | null)[][];
  row_count_returned: number;
  sampled: boolean;
  column_stats: ColumnStat[];
};

export type ImportTableRequest = {
  database: string;
  table?: string;
  row_limit?: number;
  dataset_name?: string;
  custom_sql?: string;
  filter?: Record<string, unknown>;
  sort?: [string, 1 | -1][];
  fields?: string[];
};

export type ImportDatasetResult = DatasetSummary & {
  truncated: boolean;
  row_limit_applied: number;
};

// --- Database AI Query ---
export type NLQueryRequest = {
  data_source_id: string;
  database: string;
  question: string;
};

export type NLQuerySqlQuery = string;

export type NLQueryMongoQuery = {
  collection: string;
  filter: Record<string, unknown>;
  sort: [string, 1 | -1][];
  fields: string[] | null;
};

export type NLQueryResult = {
  engine: DataSourceEngine;
  query: NLQuerySqlQuery | NLQueryMongoQuery;
  columns: string[];
  rows: (string | number | null)[][];
  row_count_returned: number;
  truncated: boolean;
  explanation: string | null;
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
