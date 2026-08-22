import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-service";
import type { ConnectionInput, ImportTableRequest } from "@/lib/api-types";

export function useDataSourcesList() {
  return useQuery({ queryKey: ["data-sources"], queryFn: api.listDataSources });
}

export function useTestConnection() {
  return useMutation({ mutationFn: (connection: ConnectionInput) => api.testConnection(connection) });
}

export function useCreateDataSource() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: { name: string; connection: ConnectionInput }) => api.createDataSource(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["data-sources"] }),
  });
}

export function useTestSavedDataSource() {
  return useMutation({ mutationFn: (id: string) => api.testSavedDataSource(id) });
}

export function useDeleteDataSource() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteDataSource(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["data-sources"] }),
  });
}

export function useDataSourceDatabases(dataSourceId: string | null) {
  return useQuery({
    queryKey: ["data-source-databases", dataSourceId],
    queryFn: () => api.listDataSourceDatabases(dataSourceId as string),
    enabled: !!dataSourceId,
  });
}

export function useDataSourceTables(dataSourceId: string | null, database: string | null) {
  return useQuery({
    queryKey: ["data-source-tables", dataSourceId, database],
    queryFn: () => api.listDataSourceTables(dataSourceId as string, database as string),
    enabled: !!dataSourceId && !!database,
  });
}

export function useDataSourcePreview(
  dataSourceId: string | null,
  database: string | null,
  table: string | null,
  limit = 100,
) {
  return useQuery({
    queryKey: ["data-source-preview", dataSourceId, database, table, limit],
    queryFn: () => api.previewDataSourceTable(dataSourceId as string, database as string, table as string, limit),
    enabled: !!dataSourceId && !!database && !!table,
  });
}

export function useImportDataSourceTable() {
  return useMutation({
    mutationFn: ({ dataSourceId, payload }: { dataSourceId: string; payload: ImportTableRequest }) =>
      api.importDataSourceTable(dataSourceId, payload),
  });
}
