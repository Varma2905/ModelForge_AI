import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-service";
import type { DataSplitConfig } from "@/lib/api-types";

export function useSplitPreview() {
  return useMutation({
    mutationFn: (payload: { dataset_id: string; config: DataSplitConfig }) =>
      api.splitPreview(payload),
  });
}

export function useTrainModel() {
  return useMutation({
    mutationFn: (payload: {
      dataset_id: string;
      model: string;
      features: string[];
      target: string;
      split: DataSplitConfig;
      hyperparameters?: Record<string, unknown>;
    }) => api.trainModel(payload),
  });
}

export function useModelMetrics(modelId: string | null) {
  return useQuery({
    queryKey: ["model-metrics", modelId],
    queryFn: () => api.getModelMetrics(modelId as string),
    enabled: !!modelId,
  });
}

export function useModelsList() {
  return useQuery({ queryKey: ["models"], queryFn: api.listModels });
}
