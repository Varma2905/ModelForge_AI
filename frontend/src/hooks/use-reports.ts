import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-service";

export function useAiExplanation(modelId: string | null) {
  return useQuery({
    queryKey: ["ai-explain", modelId],
    queryFn: () => api.explainModel(modelId as string),
    enabled: !!modelId,
    staleTime: Infinity, // The AI pipeline is expensive to re-run; treat a result as immutable once fetched.
  });
}

export function useDownloadReport() {
  return useMutation({
    mutationFn: (modelId: string) => api.downloadReport(modelId),
  });
}
