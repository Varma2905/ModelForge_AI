import type { ApiEnvelope, BaasAuthResult, BaasEndUser, BaasRecord, BaasRecordListResult } from "./types";

export class BaasApiError extends Error {
  code: number;
  constructor(code: number, message: string) {
    super(message);
    this.name = "BaasApiError";
    this.code = code;
  }
}

export type ClientOptions = {
  /** Base URL of your Regression Analysis Studio backend, e.g. "https://api.example.com" */
  projectUrl: string;
  /** Safe to embed in browser/client-side code. */
  publicKey: string;
  /**
   * Server-only. NEVER expose this in client-side/browser JavaScript — anyone
   * with it can write to your project's data. Only pass it when calling the
   * SDK from your own backend.
   */
  secretKey?: string;
};

function buildHeaders(opts: ClientOptions, needsSecret: boolean): Record<string, string> {
  if (needsSecret && !opts.secretKey) {
    throw new Error(
      "This operation requires a secret key. Never expose the secret key in client-side/browser code — " +
        "call this from your own server instead. See the SDK README for details.",
    );
  }
  const headers: Record<string, string> = { "X-Public-Key": opts.publicKey };
  if (opts.secretKey) headers["X-Secret-Key"] = opts.secretKey;
  return headers;
}

async function request<T>(
  opts: ClientOptions,
  path: string,
  init: (RequestInit & { needsSecret?: boolean }) = {},
): Promise<T> {
  const { needsSecret = false, ...rest } = init;
  // Runs before any network call — a missing required secret key fails
  // fast and locally rather than silently sending a request the server
  // will reject anyway.
  const authHeaders = buildHeaders(opts, needsSecret);

  const res = await fetch(`${opts.projectUrl}${path}`, {
    ...rest,
    headers: {
      ...(rest.body ? { "Content-Type": "application/json" } : {}),
      ...authHeaders,
      ...(rest.headers as Record<string, string> | undefined),
    },
  });

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new BaasApiError(res.status, `Request failed with status ${res.status}`);
  }

  if (body && typeof body === "object" && "success" in (body as Record<string, unknown>)) {
    const envelope = body as ApiEnvelope<T>;
    if (envelope.success) return envelope.data;
    throw new BaasApiError(envelope.error.code, envelope.error.message);
  }
  throw new BaasApiError(res.status, "Unexpected response shape from server.");
}

export function createClient(opts: ClientOptions) {
  return {
    data: {
      list: (table: string, limit = 50) =>
        request<BaasRecordListResult>(opts, `/baas/data/${encodeURIComponent(table)}?limit=${limit}`),
      get: (table: string, id: string) =>
        request<BaasRecord>(opts, `/baas/data/${encodeURIComponent(table)}/${encodeURIComponent(id)}`),
      insert: (table: string, record: Record<string, unknown>) =>
        request<BaasRecord>(opts, `/baas/data/${encodeURIComponent(table)}`, {
          method: "POST",
          body: JSON.stringify(record),
          needsSecret: true,
        }),
      update: (table: string, id: string, record: Record<string, unknown>) =>
        request<BaasRecord>(opts, `/baas/data/${encodeURIComponent(table)}/${encodeURIComponent(id)}`, {
          method: "PATCH",
          body: JSON.stringify(record),
          needsSecret: true,
        }),
      remove: (table: string, id: string) =>
        request<{ deleted: boolean }>(opts, `/baas/data/${encodeURIComponent(table)}/${encodeURIComponent(id)}`, {
          method: "DELETE",
          needsSecret: true,
        }),
    },
    auth: {
      signup: (email: string, password: string) =>
        request<BaasAuthResult>(opts, "/baas/end-users/signup", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        }),
      login: (email: string, password: string) =>
        request<BaasAuthResult>(opts, "/baas/end-users/login", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        }),
      me: (endUserToken: string) =>
        request<BaasEndUser>(opts, "/baas/end-users/me", {
          headers: { Authorization: `Bearer ${endUserToken}` },
        }),
    },
  };
}

export type BaasClient = ReturnType<typeof createClient>;
