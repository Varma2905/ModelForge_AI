import { createFileRoute, Link } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useModelsList } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { PlusCircle } from "lucide-react";

export const Route = createFileRoute("/history")({
  beforeLoad: requireAuth,
  component: HistoryPage,
});

function HistoryPage() {
  const { data, isLoading, isError } = useModelsList();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dataset History</h1>
          <p className="text-sm text-muted-foreground">
            Past regression analyses and their results.
          </p>
        </div>
        <Button asChild>
          <Link to="/new/upload">New Analysis</Link>
        </Button>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>All Analyses</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : isError ? (
            <p className="text-sm text-destructive">Couldn't load your analysis history.</p>
          ) : !data || data.length === 0 ? (
            <div className="text-center py-10">
              <p className="text-sm font-medium">No analyses yet</p>
              <p className="text-xs text-muted-foreground mt-1">
                Start your first regression analysis by uploading a dataset.
              </p>
              <Button asChild size="sm" className="mt-4">
                <Link to="/new/upload">
                  <PlusCircle className="h-3.5 w-3.5 mr-1.5" /> New Analysis
                </Link>
              </Button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2">Dataset</th>
                    <th>Model</th>
                    <th>R²</th>
                    <th>Date</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((m) => (
                    <tr key={m.model_id} className="border-b">
                      <td className="py-3 font-medium">{m.dataset_name}</td>
                      <td>{m.model}</td>
                      <td>
                        <Badge variant="secondary">
                          {typeof m.metrics.R2 === "number" ? m.metrics.R2.toFixed(3) : "—"}
                        </Badge>
                      </td>
                      <td className="text-muted-foreground">
                        {m.created_at ? new Date(m.created_at).toLocaleDateString() : "—"}
                      </td>
                      <td className="text-right">
                        <Button asChild variant="ghost" size="sm">
                          <Link to="/models/$modelId" params={{ modelId: m.model_id }}>
                            View
                          </Link>
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
