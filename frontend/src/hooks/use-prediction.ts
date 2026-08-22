import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api-service";

export function usePredict() {
  return useMutation({
    mutationFn: (payload: { model_id: string; values: Record<string, number> }) =>
      api.predict(payload),
  });
}
