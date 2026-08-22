import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-service";

export function useHfSearch(query: string, task?: string, limit = 20) {
  return useQuery({
    queryKey: ["hf-search", query, task, limit],
    queryFn: () => api.searchHfModels({ query, task, limit }),
    enabled: query.trim().length > 0,
  });
}

export function useHfModelDetails(modelId: string | null) {
  return useQuery({
    queryKey: ["hf-model", modelId],
    queryFn: () => api.getHfModel(modelId as string),
    enabled: !!modelId,
  });
}

export function useHfCompatibilityCheck() {
  return useMutation({
    mutationFn: ({ modelId, datasetId }: { modelId: string; datasetId: string }) =>
      api.checkHfCompatibility(modelId, datasetId),
  });
}
