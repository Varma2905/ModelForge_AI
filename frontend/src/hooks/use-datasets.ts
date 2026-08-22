import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-service";

export function useUploadDataset() {
  return useMutation({ mutationFn: (file: File) => api.uploadDataset(file) });
}

export function useCreateDataset() {
  return useMutation({
    mutationFn: (payload: { name?: string; columns: string[]; rows: unknown[][] }) =>
      api.createDataset(payload),
  });
}

export function usePreviewDataset(datasetId: string | null, limit = 200) {
  return useQuery({
    queryKey: ["dataset-preview", datasetId, limit],
    queryFn: () => api.previewDataset(datasetId as string, limit),
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
