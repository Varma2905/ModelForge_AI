import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api-service";
import type { PredictionValues } from "@/lib/api-types";

export function usePredict() {
  return useMutation({
    mutationFn: (payload: { model_id: string; values: PredictionValues }) =>
      api.predict(payload),
  });
}

export function usePredictClass() {
  return useMutation({
    mutationFn: (payload: { model_id: string; values: PredictionValues }) =>
      api.predictClass(payload),
  });
}
