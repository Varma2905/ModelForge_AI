import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api-service";
import type { PreprocessConfig } from "@/lib/api-types";

export function usePreprocess() {
  return useMutation({
    mutationFn: (payload: { dataset_id: string; config: PreprocessConfig }) =>
      api.preprocess(payload),
  });
}

export function useSelectFeatures() {
  return useMutation({
    mutationFn: (payload: { dataset_id: string; features: string[]; target: string }) =>
      api.selectFeatures(payload),
  });
}
