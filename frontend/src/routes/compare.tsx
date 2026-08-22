import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useDatasetsList, useDatasetModels } from "@/hooks/use-datasets";
import { requireAuth } from "@/lib/require-auth";
import { GitCompare, PlusCircle, Trophy } from "lucide-react";

export const Route = createFileRoute("/compare")({
  beforeLoad: requireAuth,
  component: ComparePage,
});

function ComparePage() {
  const datasetsQuery = useDatasetsList();
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedDatasetId && datasetsQuery.data && datasetsQuery.data.length > 0) {
      setSelectedDatasetId(datasetsQuery.data[0].dataset_id);
    }
  }, [datasetsQuery.data, selectedDatasetId]);

  const modelsQuery = useDatasetModels(selectedDatasetId);

  const bestModelId =
    modelsQuery.data && modelsQuery.data.length > 0
      ? modelsQuery.data.reduce(
          (best, m) => (m.metrics.R2 > best.metrics.R2 ? m : best),
          modelsQuery.data[0],
        ).model_id
      : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <GitCompare className="h-5 w-5" /> Compare Models
        </h1>
        <p className="text-sm text-muted-foreground">
          See how different algorithms performed on the same dataset.
        </p>
      </div>

      {datasetsQuery.isLoading ? (
        <Skeleton className="h-10 w-64" />
      ) : !datasetsQuery.data || datasetsQuery.data.length === 0 ? (
        <Card>
          <CardContent className="p-10 text-center">
            <p className="text-sm font-medium">No datasets yet</p>
            <p className="text-xs text-muted-foreground mt-1">
              Upload a dataset and train a couple of models to compare them here.
            </p>
            <Button asChild size="sm" className="mt-4">
              <Link to="/new/upload">
                <PlusCircle className="h-3.5 w-3.5 mr-1.5" /> New Analysis
              </Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          <Select value={selectedDatasetId ?? ""} onValueChange={setSelectedDatasetId}>
            <SelectTrigger className="w-72">
              <SelectValue placeholder="Select a dataset" />
            </SelectTrigger>
            <SelectContent>
              {datasetsQuery.data.map((d) => (
                <SelectItem key={d.dataset_id} value={d.dataset_id}>
                  {d.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Card>
            <CardHeader>
              <CardTitle>Model Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              {modelsQuery.isLoading ? (
                <Skeleton className="h-32 w-full" />
              ) : !modelsQuery.data || modelsQuery.data.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No models trained on this dataset yet. Train one via New Analysis.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-muted-foreground">
                        <th className="pb-2">Model</th>
                        <th>R²</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>MSE</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelsQuery.data.map((m) => {
                        const isBest = m.model_id === bestModelId;
                        return (
                          <tr
                            key={m.model_id}
                            className={`border-b ${isBest ? "bg-emerald-50 dark:bg-emerald-950/20" : ""}`}
                          >
                            <td className="py-3 font-medium">
                              <span className="flex items-center gap-2">
                                {isBest && <Trophy className="h-3.5 w-3.5 text-emerald-500" />}
                                {m.model}
                              </span>
                            </td>
                            <td>
                              <Badge variant={isBest ? "default" : "secondary"}>
                                {typeof m.metrics.R2 === "number" ? m.metrics.R2.toFixed(3) : "—"}
                              </Badge>
                            </td>
                            <td>
                              {typeof m.metrics.MAE === "number" ? m.metrics.MAE.toFixed(3) : "—"}
                            </td>
                            <td>
                              {typeof m.metrics.RMSE === "number" ? m.metrics.RMSE.toFixed(3) : "—"}
                            </td>
                            <td>
                              {typeof m.metrics.MSE === "number" ? m.metrics.MSE.toFixed(0) : "—"}
                            </td>
                            <td className="text-right">
                              <Button asChild variant="ghost" size="sm">
                                <Link to="/models/$modelId" params={{ modelId: m.model_id }}>
                                  View
                                </Link>
                              </Button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
