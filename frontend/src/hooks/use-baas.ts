import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-service";
import type { BaasColumn, BaasCredentials } from "@/lib/api-types";

// --- Management (Studio JWT) ---
export function useBaasProjects() {
  return useQuery({ queryKey: ["baas-projects"], queryFn: api.listBaasProjects });
}

export function useCreateBaasProject() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => api.createBaasProject(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["baas-projects"] }),
  });
}

export function useBaasProject(projectId: string | null) {
  return useQuery({
    queryKey: ["baas-project", projectId],
    queryFn: () => api.getBaasProject(projectId as string),
    enabled: !!projectId,
  });
}

export function useDeleteBaasProject() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (projectId: string) => api.deleteBaasProject(projectId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["baas-projects"] }),
  });
}

export function useBaasKeys(projectId: string | null) {
  return useQuery({
    queryKey: ["baas-keys", projectId],
    queryFn: () => api.listBaasKeys(projectId as string),
    enabled: !!projectId,
  });
}

export function useCreateBaasKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ projectId, label }: { projectId: string; label?: string }) =>
      api.createBaasKey(projectId, label),
    onSuccess: (_data, vars) => queryClient.invalidateQueries({ queryKey: ["baas-keys", vars.projectId] }),
  });
}

export function useRevokeBaasKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ projectId, keyId }: { projectId: string; keyId: string }) =>
      api.revokeBaasKey(projectId, keyId),
    onSuccess: (_data, vars) => queryClient.invalidateQueries({ queryKey: ["baas-keys", vars.projectId] }),
  });
}

export function useProposeBaasSchema() {
  return useMutation({
    mutationFn: ({ projectId, description }: { projectId: string; description: string }) =>
      api.proposeBaasSchema(projectId, description),
  });
}

export function useBaasTables(projectId: string | null) {
  return useQuery({
    queryKey: ["baas-tables", projectId],
    queryFn: () => api.listBaasTables(projectId as string),
    enabled: !!projectId,
  });
}

export function useCreateBaasTable() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ projectId, name, columns }: { projectId: string; name: string; columns: BaasColumn[] }) =>
      api.createBaasTable(projectId, { name, columns }),
    onSuccess: (_data, vars) => queryClient.invalidateQueries({ queryKey: ["baas-tables", vars.projectId] }),
  });
}

// --- Data browser (public/secret key, not JWT) ---
export function useBaasRecords(creds: BaasCredentials | null, tableName: string | null, limit = 50) {
  return useQuery({
    queryKey: ["baas-records", creds?.publicKey, tableName, limit],
    queryFn: () => api.listBaasRecords(creds as BaasCredentials, tableName as string, limit),
    enabled: !!creds && !!tableName,
  });
}

export function useCreateBaasRecord() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      creds,
      tableName,
      data,
    }: {
      creds: BaasCredentials;
      tableName: string;
      data: Record<string, unknown>;
    }) => api.createBaasRecord(creds, tableName, data),
    onSuccess: (_data, vars) =>
      queryClient.invalidateQueries({ queryKey: ["baas-records", vars.creds.publicKey, vars.tableName] }),
  });
}

export function useUpdateBaasRecord() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      creds,
      tableName,
      recordId,
      data,
    }: {
      creds: BaasCredentials;
      tableName: string;
      recordId: string;
      data: Record<string, unknown>;
    }) => api.updateBaasRecord(creds, tableName, recordId, data),
    onSuccess: (_data, vars) =>
      queryClient.invalidateQueries({ queryKey: ["baas-records", vars.creds.publicKey, vars.tableName] }),
  });
}

export function useDeleteBaasRecord() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ creds, tableName, recordId }: { creds: BaasCredentials; tableName: string; recordId: string }) =>
      api.deleteBaasRecord(creds, tableName, recordId),
    onSuccess: (_data, vars) =>
      queryClient.invalidateQueries({ queryKey: ["baas-records", vars.creds.publicKey, vars.tableName] }),
  });
}
