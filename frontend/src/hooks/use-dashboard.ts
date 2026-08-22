import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-service";

export function useDashboardSummary() {
  return useQuery({ queryKey: ["dashboard-summary"], queryFn: api.dashboardSummary });
}
