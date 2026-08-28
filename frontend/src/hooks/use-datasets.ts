import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-service";

export function useUploadDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => api.uploadDataset(file),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function useCreateDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: { name?: string; columns: string[]; rows: unknown[][] }) =>
      api.createDataset(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function usePreviewDataset(datasetId: string | null, limit = 200) {
  return useQuery({
    queryKey: ["dataset-preview", datasetId, limit],
    queryFn: () => api.previewDataset(datasetId as string, limit),
    enabled: !!datasetId,
  });
}

export function useDatasetProfile(datasetId: string | null) {
  return useQuery({
    queryKey: ["dataset-profile", datasetId],
    queryFn: () => api.datasetProfile(datasetId as string),
    enabled: !!datasetId,
  });
}

export function useDatasetsList() {
  return useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
}

export function useDatasetModels(datasetId: string | null) {
  return useQuery({
    queryKey: ["dataset-models", datasetId],
    queryFn: () => api.listDatasetModels(datasetId as string),
    enabled: !!datasetId,
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (datasetId: string) => api.deleteDataset(datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });
}

// --- Dataset sources (Kaggle / Google Drive) ---

export function useDatasetSourcesStatus() {
  return useQuery({ queryKey: ["dataset-sources-status"], queryFn: api.datasetSourcesStatus });
}

export function useKaggleStatus() {
  return useQuery({ queryKey: ["kaggle-status"], queryFn: api.kaggleStatus });
}

export function useConnectKaggle() {
  return useMutation({
    mutationFn: (payload: { username: string; key: string }) => api.connectKaggle(payload),
  });
}

export function useDisconnectKaggle() {
  return useMutation({ mutationFn: () => api.disconnectKaggle() });
}

export function useResolveKaggleDataset() {
  return useMutation({
    mutationFn: (payload: { dataset_ref: string }) => api.resolveKaggleDataset(payload),
  });
}

export function useImportKaggleDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: { dataset_ref: string; file_name: string }) => api.importKaggleDataset(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function useGoogleDriveAuthUrl() {
  return useMutation({ mutationFn: () => api.googleDriveAuthUrl() });
}

export function useGoogleDriveFiles(session: string | null) {
  return useQuery({
    queryKey: ["gdrive-files", session],
    queryFn: () => api.listGoogleDriveFiles(session as string),
    enabled: !!session,
  });
}

export function useImportGoogleDriveFile() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: { session: string; file_id: string; file_name: string }) =>
      api.importGoogleDriveFile(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["datasets"] }),
  });
}
