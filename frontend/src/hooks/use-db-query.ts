import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api-service";
import type { NLQueryRequest } from "@/lib/api-types";

export function useAskDatabaseQuestion() {
  return useMutation({
    mutationFn: (payload: NLQueryRequest) => api.askDatabaseQuestion(payload),
  });
}
