// Types for the BaaS *public* API surface — a created project's own end
// users and data, never the Studio itself (no Studio-JWT-authed types here).

export type ApiEnvelope<T> =
  | { success: true; data: T; message?: string | null }
  | { success: false; error: { code: number; message: string } };

export type BaasRecord = {
  id: string;
  created_at: string;
  updated_at: string;
} & Record<string, unknown>;

export type BaasRecordListResult = {
  records: BaasRecord[];
  total_count: number;
};

export type BaasEndUser = {
  id: string;
  email: string;
  created_at: string;
};

export type BaasAuthResult = {
  user: BaasEndUser;
  token: string;
};
