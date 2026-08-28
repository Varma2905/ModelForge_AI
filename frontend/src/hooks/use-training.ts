import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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

export function useRegressionModels() {
  // Registry-backed catalog (see backend regression_registry.py) — every
  // entry's `status` is computed live server-side (package installed? a
  // real pipeline exists?), so the model-selection page never has to guess
  // or hard-code which models are actually trainable right now.
  return useQuery({ queryKey: ["regression-models"], queryFn: api.listRegressionModels });
}

export function useModelMetrics(modelId: string | null) {
  return useQuery({
    queryKey: ["model-metrics", modelId],
    queryFn: () => api.getModelMetrics(modelId as string),
    enabled: !!modelId,
  });
}

export function useModelsList() {
  // Returns BOTH regression and classification models — the backend's
  // /models endpoint queries the shared "models" collection with no
  // model_type filter, so no separate classification list/merge is needed
  // here (each item's model_type field, defaulting to "regression" when
  // absent, is what a caller uses to tell them apart).
  return useQuery({ queryKey: ["models"], queryFn: api.listModels });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) => api.deleteModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

// --- Classification ---

export function useSelectClassificationFeatures() {
  return useMutation({
    mutationFn: (payload: { dataset_id: string; features: string[]; target: string }) =>
      api.selectClassificationFeatures(payload),
  });
}

export function useClassificationModels() {
  // Registry-backed catalog (see backend classification_registry.py) — same
  // pattern as useRegressionModels: status is computed live server-side.
  return useQuery({ queryKey: ["classification-models"], queryFn: api.listClassificationModels });
}

export function useTrainClassificationModel() {
  return useMutation({
    mutationFn: (payload: {
      dataset_id: string;
      model: string;
      features: string[];
      target: string;
      split: DataSplitConfig;
      hyperparameters?: Record<string, unknown>;
    }) => api.trainClassificationModel(payload),
  });
}

export function useClassificationModelMetrics(modelId: string | null) {
  return useQuery({
    // Same cache key family as regression's model-metrics — both hit the
    // identical /model-metrics/{id} endpoint, just typed differently here.
    queryKey: ["model-metrics", modelId],
    queryFn: () => api.getClassificationModelMetrics(modelId as string),
    enabled: !!modelId,
  });
}

// --- Clustering ---

export function useClusteringModels() {
  // Registry-backed catalog (see backend clustering_registry.py).
  return useQuery({ queryKey: ["clustering-models"], queryFn: api.listClusteringModels });
}

export function useSelectClusteringFeatures() {
  return useMutation({
    mutationFn: (payload: { dataset_id: string; features: string[] }) =>
      api.selectClusteringFeatures(payload),
  });
}

export function useTrainClusteringModel() {
  return useMutation({
    mutationFn: (payload: {
      dataset_id: string;
      model: string;
      features: string[];
      hyperparameters?: Record<string, unknown>;
    }) => api.trainClusteringModel(payload),
  });
}

export function useClusteringModelMetrics(modelId: string | null) {
  return useQuery({
    // A dedicated endpoint, not the shared regression/classification one —
    // the clustering response shape (cluster_profiles/visualizations, no
    // target) is different enough to warrant its own query-key family too.
    queryKey: ["clustering-model-metrics", modelId],
    queryFn: () => api.getClusteringModelMetrics(modelId as string),
    enabled: !!modelId,
  });
}
