import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api-service";
import type { DBQueryExportRequest, NLQueryRequest } from "@/lib/api-types";

export function useAskDatabaseQuestion() {
  return useMutation({
    mutationFn: (payload: NLQueryRequest) => api.askDatabaseQuestion(payload),
  });
}

export function useExportDbQueryReport() {
  return useMutation({
    mutationFn: (payload: DBQueryExportRequest) => api.exportDbQueryReport(payload),
  });
}
