// HTTP client for the FastAPI backend. Unwraps the {success,data,message}
// response envelope and surfaces failures as a structured ApiError.
import { clearToken, getToken } from "./auth-store";
import type {
  AIChatErrorType,
  AIChatStreamRequest,
  AIExplainResult,
  AuthPayload,
  AuthUser,
  ChartData,
  ChatResult,
  ClassificationModelEntry,
  ClassificationModelMetricsResult,
  ClassificationSelectFeaturesResult,
  ClassPredictionResult,
  ClusteringModelEntry,
  ClusteringModelMetricsResult,
  ClusteringSelectFeaturesResult,
  TrainClusteringModelResult,
  DashboardSummary,
  DataSplitConfig,
  DatasetListItem,
  DatasetModelListItem,
  DatasetPreview,
  DatasetProfile,
  DatasetSourcesStatus,
  DatasetSummary,
  GoogleDriveFile,
  GraphsResult,
  KaggleResolveResult,
  KaggleStatus,
  HFCompatibilityResult,
  HFModelDetails,
  HFModelSummary,
  HFRunModelRequest,
  HFRunModelResult,
  ModelListItem,
  ModelMetricsResult,
  PredictionResult,
  PredictionValues,
  PreprocessConfig,
  PreprocessResult,
  RegressionModelEntry,
  SelectFeaturesResult,
  SplitPreview,
  TrainClassificationModelResult,
  TrainModelResult,
} from "./api-types";

const API_BASE_URL_KEY = "regression-studio:api-base-url";

function isBrowser() {
  return typeof window !== "undefined";
}

export function getApiBaseUrl(): string {
  if (isBrowser()) {
    try {
      const override = localStorage.getItem(API_BASE_URL_KEY);
      if (override && override.trim()) return override.trim();
    } catch {
      // ignore
    }
  }
  return import.meta.env.VITE_API_URL || "http://localhost:8001";
}

export function setApiBaseUrl(url: string) {
  if (!isBrowser()) return;
  try {
    const trimmed = url.trim();
    if (trimmed) {
      localStorage.setItem(API_BASE_URL_KEY, trimmed);
    } else {
      localStorage.removeItem(API_BASE_URL_KEY);
    }
  } catch {
    // ignore
  }
}

export class ApiError extends Error {
  code: number;
  constructor(code: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

export function isUnauthorizedError(err: unknown): boolean {
  return err instanceof ApiError && err.code === 401;
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function unwrap<T>(res: Response): Promise<T> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new ApiError(res.status, `Request failed with status ${res.status}`);
  }

  if (body && typeof body === "object" && "success" in (body as Record<string, unknown>)) {
    const envelope = body as
      | { success: true; data: T; message?: string | null }
      | { success: false; error: { code: number; message: string } };
    if (envelope.success) return envelope.data;
    if (envelope.error.code === 401) clearToken();
    throw new ApiError(envelope.error.code, envelope.error.message);
  }

  // Non-enveloped endpoints (health check, /chat)
  if (!res.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    if (res.status === 401) clearToken();
    throw new ApiError(res.status, detail ?? `Request failed with status ${res.status}`);
  }
  return body as T;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = init?.body instanceof FormData;
  const res = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !isFormData ? { "Content-Type": "application/json" } : {}),
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  return unwrap<T>(res);
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

function get<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

function postForm<T>(path: string, form: FormData): Promise<T> {
  return request<T>(path, { method: "POST", body: form });
}

function del<T>(path: string): Promise<T> {
  return request<T>(path, { method: "DELETE" });
}

async function getBlob(path: string): Promise<Blob> {
  const res = await fetch(`${getApiBaseUrl()}${path}`, { headers: authHeaders() });
  if (!res.ok) {
    let message = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (body?.error?.message) message = body.error.message;
    } catch {
      // ignore
    }
    if (res.status === 401) clearToken();
    throw new ApiError(res.status, message);
  }
  return res.blob();
}

// --- AI Assistant streaming chat ---
// Not routed through request()/unwrap() like the rest of the client: a
// successful response here is a raw text/event-stream body, not the usual
// {success,data} JSON envelope, so it needs its own fetch + SSE parsing.
export type ChatStreamCallbacks = {
  onDelta: (delta: string) => void;
  onDone: () => void;
  onError: (message: string, type: AIChatErrorType) => void;
};

export async function streamChat(
  payload: AIChatStreamRequest,
  callbacks: ChatStreamCallbacks,
  signal: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${getApiBaseUrl()}/ai/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    callbacks.onError("Connection failed. Please try again.", "network");
    return;
  }

  if (!res.ok) {
    let message = `Request failed with status ${res.status}`;
    let errorType: AIChatErrorType = "server";
    try {
      const body = await res.json();
      if (body?.error?.message) message = body.error.message;
    } catch {
      // ignore — use the generic message above
    }
    if (res.status === 401) {
      clearToken();
      errorType = "auth";
    } else if (res.status === 429) {
      errorType = "rate_limit";
    } else if (res.status === 503) {
      errorType = "config";
    }
    callbacks.onError(message, errorType);
    return;
  }

  if (!res.body) {
    callbacks.onError("Connection failed. Please try again.", "network");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIdx: number;
      while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + 2);
        const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data:"));
        if (!dataLine) continue;
        const jsonStr = dataLine.slice(5).trim();
        if (!jsonStr) continue;

        let evt: { delta?: string; done?: boolean; error?: string; error_type?: AIChatErrorType };
        try {
          evt = JSON.parse(jsonStr);
        } catch {
          continue;
        }
        if (evt.error) {
          callbacks.onError(evt.error, evt.error_type ?? "server");
          return;
        }
        if (evt.delta) callbacks.onDelta(evt.delta);
        if (evt.done) {
          callbacks.onDone();
          return;
        }
      }
    }
    callbacks.onDone();
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    callbacks.onError("Connection failed. Please try again.", "network");
  }
}

export const api = {
  // Auth
  signup: (payload: { name: string; email: string; password: string }) =>
    post<AuthPayload>("/auth/signup", payload),
  login: (payload: { email: string; password: string; remember_me?: boolean }) =>
    post<AuthPayload>("/auth/login", payload),
  me: () => get<AuthUser>("/auth/me"),
  logout: () => post<null>("/auth/logout"),

  // Datasets
  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return postForm<DatasetSummary>("/upload-dataset", form);
  },
  createDataset: (payload: { name?: string; columns: string[]; rows: unknown[][] }) =>
    post<DatasetSummary>("/create-dataset", payload),
  previewDataset: (datasetId: string, limit = 200) =>
    get<DatasetPreview>(`/preview-dataset/${datasetId}?limit=${limit}`),
  datasetProfile: (datasetId: string) => get<DatasetProfile>(`/datasets/${datasetId}/profile`),
  listDatasets: () => get<DatasetListItem[]>("/datasets"),
  listDatasetModels: (datasetId: string) =>
    get<DatasetModelListItem[]>(`/datasets/${datasetId}/models`),
  deleteDataset: (datasetId: string) => del<{ deleted: boolean }>(`/datasets/${datasetId}`),

  // Dataset sources (Kaggle / Google Drive)
  datasetSourcesStatus: () => get<DatasetSourcesStatus>("/datasets/sources/status"),
  kaggleStatus: () => get<KaggleStatus>("/datasets/kaggle/status"),
  connectKaggle: (payload: { username: string; key: string }) =>
    post<KaggleStatus>("/datasets/kaggle/connect", payload),
  disconnectKaggle: () => post<{ connected: boolean }>("/datasets/kaggle/disconnect", {}),
  resolveKaggleDataset: (payload: { dataset_ref: string }) =>
    post<KaggleResolveResult>("/datasets/kaggle/resolve", payload),
  importKaggleDataset: (payload: { dataset_ref: string; file_name: string }) =>
    post<DatasetSummary>("/datasets/kaggle/import", payload),
  googleDriveAuthUrl: () => get<{ auth_url: string }>("/datasets/google-drive/auth-url"),
  listGoogleDriveFiles: (session: string) =>
    get<{ files: GoogleDriveFile[] }>(`/datasets/google-drive/files?session=${encodeURIComponent(session)}`),
  importGoogleDriveFile: (payload: { session: string; file_id: string; file_name: string }) =>
    post<DatasetSummary>("/datasets/google-drive/import", payload),

  // Preprocessing / features / split
  selectFeatures: (payload: { dataset_id: string; features: string[]; target: string }) =>
    post<SelectFeaturesResult>("/select-features", payload),
  preprocess: (payload: { dataset_id: string; config: PreprocessConfig }) =>
    post<PreprocessResult>("/preprocess", payload),
  splitPreview: (payload: { dataset_id: string; config: DataSplitConfig }) =>
    post<SplitPreview>("/split-data", payload),

  // Training / prediction
  trainModel: (payload: {
    dataset_id: string;
    model: string;
    features: string[];
    target: string;
    split: DataSplitConfig;
    hyperparameters?: Record<string, unknown>;
  }) => post<TrainModelResult>("/train-model", payload),
  getModelMetrics: (modelId: string) => get<ModelMetricsResult>(`/model-metrics/${modelId}`),
  listModels: () => get<ModelListItem[]>("/models"),
  listRegressionModels: () => get<RegressionModelEntry[]>("/regression-models"),
  deleteModel: (modelId: string) => del<{ deleted: boolean }>(`/models/${modelId}`),
  predict: (payload: { model_id: string; values: PredictionValues }) =>
    post<PredictionResult>("/predict", payload),

  // Classification — select-features/train-model are dedicated endpoints
  // (different target-validation rule, different pipeline/metrics engine);
  // model-metrics/predict reuse the exact same backend routes as regression
  // (already generic over any model doc's model_type), just typed
  // differently here for the classification wizard's callers.
  selectClassificationFeatures: (payload: { dataset_id: string; features: string[]; target: string }) =>
    post<ClassificationSelectFeaturesResult>("/classify/select-features", payload),
  listClassificationModels: () => get<ClassificationModelEntry[]>("/classify/classification-models"),
  trainClassificationModel: (payload: {
    dataset_id: string;
    model: string;
    features: string[];
    target: string;
    split: DataSplitConfig;
    hyperparameters?: Record<string, unknown>;
  }) => post<TrainClassificationModelResult>("/classify/train-model", payload),
  getClassificationModelMetrics: (modelId: string) =>
    get<ClassificationModelMetricsResult>(`/model-metrics/${modelId}`),
  predictClass: (payload: { model_id: string; values: PredictionValues }) =>
    post<ClassPredictionResult>("/predict", payload),

  // Clustering — unsupervised, so there is no target column and no
  // train/test split; select-features/train-model are dedicated endpoints
  // for that reason, and model-metrics is its own route (not the shared
  // regression/classification one) since the response shape is genuinely
  // different (cluster_profiles/cluster_sizes/visualizations, no target).
  listClusteringModels: () => get<ClusteringModelEntry[]>("/cluster/clustering-models"),
  selectClusteringFeatures: (payload: { dataset_id: string; features: string[] }) =>
    post<ClusteringSelectFeaturesResult>("/cluster/select-features", payload),
  trainClusteringModel: (payload: {
    dataset_id: string;
    model: string;
    features: string[];
    hyperparameters?: Record<string, unknown>;
  }) => post<TrainClusteringModelResult>("/cluster/train-model", payload),
  getClusteringModelMetrics: (modelId: string) =>
    get<ClusteringModelMetricsResult>(`/cluster/model-metrics/${modelId}`),

  // AI
  explainModel: (modelId: string) => post<AIExplainResult>("/ai/explain", { model_id: modelId }),
  chat: (payload: { message: string; provider?: string; model?: string }) =>
    post<ChatResult>("/chat", payload),

  // Reports / graphs
  getGraphs: (modelId: string) => get<GraphsResult>(`/graphs/${modelId}`),
  downloadReport: (modelId: string) => getBlob(`/report/${modelId}`),

  // Dashboard
  dashboardSummary: () => get<DashboardSummary>("/dashboard/summary"),

  // Hugging Face Models
  searchHfModels: (params: { query?: string; task?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params.query) qs.set("query", params.query);
    if (params.task) qs.set("task", params.task);
    if (params.limit) qs.set("limit", String(params.limit));
    return get<HFModelSummary[]>(`/hf/search?${qs.toString()}`);
  },
  getHfModel: (modelId: string) =>
    get<HFModelDetails>(`/hf/models/${encodeURIComponent(modelId)}`),
  checkHfCompatibility: (modelId: string, datasetId: string) =>
    post<HFCompatibilityResult>(`/hf/models/${encodeURIComponent(modelId)}/check-compatibility`, {
      dataset_id: datasetId,
    }),
  runHfModel: (modelId: string, payload: HFRunModelRequest) =>
    post<HFRunModelResult>(`/hf/models/${encodeURIComponent(modelId)}/run`, payload),
};

export type { ChartData };
