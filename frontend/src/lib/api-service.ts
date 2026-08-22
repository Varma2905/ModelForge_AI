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
  ConnectionInput,
  DashboardSummary,
  DataSourcePreview,
  DataSourceSummary,
  DataSplitConfig,
  DatasetListItem,
  DatasetModelListItem,
  DatasetPreview,
  DatasetSummary,
  GraphsResult,
  HFCompatibilityResult,
  HFModelDetails,
  HFModelSummary,
  HFRunModelRequest,
  HFRunModelResult,
  ImportDatasetResult,
  ImportTableRequest,
  ModelListItem,
  ModelMetricsResult,
  PredictionResult,
  PreprocessConfig,
  PreprocessResult,
  SplitPreview,
  TablesResult,
  TestConnectionResult,
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
  listDatasets: () => get<DatasetListItem[]>("/datasets"),
  listDatasetModels: (datasetId: string) =>
    get<DatasetModelListItem[]>(`/datasets/${datasetId}/models`),

  // Preprocessing / features / split
  selectFeatures: (payload: { dataset_id: string; features: string[]; target: string }) =>
    post<{ status: string; features: string[]; target: string }>("/select-features", payload),
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
  predict: (payload: { model_id: string; values: Record<string, number> }) =>
    post<PredictionResult>("/predict", payload),

  // AI
  explainModel: (modelId: string) => post<AIExplainResult>("/ai/explain", { model_id: modelId }),
  chat: (payload: { message: string; provider?: string; model?: string }) =>
    post<ChatResult>("/chat", payload),

  // Reports / graphs
  getGraphs: (modelId: string) => get<GraphsResult>(`/graphs/${modelId}`),
  downloadReport: (modelId: string) => getBlob(`/report/${modelId}`),

  // Dashboard
  dashboardSummary: () => get<DashboardSummary>("/dashboard/summary"),

  // Data Sources
  testConnection: (connection: ConnectionInput) =>
    post<TestConnectionResult>("/data-sources/test", { connection }),
  createDataSource: (payload: { name: string; connection: ConnectionInput }) =>
    post<DataSourceSummary>("/data-sources", payload),
  listDataSources: () => get<DataSourceSummary[]>("/data-sources"),
  getDataSource: (id: string) => get<DataSourceSummary>(`/data-sources/${id}`),
  testSavedDataSource: (id: string) => post<TestConnectionResult>(`/data-sources/${id}/test`),
  deleteDataSource: (id: string) => del<{ deleted: boolean }>(`/data-sources/${id}`),
  listDataSourceDatabases: (id: string) =>
    get<{ databases: string[] }>(`/data-sources/${id}/databases`),
  listDataSourceTables: (id: string, database: string) =>
    get<TablesResult>(`/data-sources/${id}/tables?database=${encodeURIComponent(database)}`),
  previewDataSourceTable: (id: string, database: string, table: string, limit = 100) =>
    get<DataSourcePreview>(
      `/data-sources/${id}/preview?database=${encodeURIComponent(database)}&table=${encodeURIComponent(table)}&limit=${limit}`,
    ),
  importDataSourceTable: (id: string, payload: ImportTableRequest) =>
    post<ImportDatasetResult>(`/data-sources/${id}/import`, payload),

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
