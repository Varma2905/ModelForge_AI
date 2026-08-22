import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EmptyState } from "@/components/empty-state";
import { useDatasetsList, useDatasetModels } from "@/hooks/use-datasets";
import { requireAuth } from "@/lib/require-auth";
import { GitCompare, MessageCircleQuestion, Trophy } from "lucide-react";

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
  const datasetName = datasetsQuery.data?.find((d) => d.dataset_id === selectedDatasetId)?.name;

  const bestModelId =
    modelsQuery.data && modelsQuery.data.length > 0
      ? modelsQuery.data.reduce(
          (best, m) => (m.metrics.R2 > best.metrics.R2 ? m : best),
          modelsQuery.data[0],
        ).model_id
      : null;

  const comparePrompt = useMemo(() => {
    if (!modelsQuery.data || modelsQuery.data.length === 0 || !datasetName) return null;
    const lines = modelsQuery.data
      .map((m) => `${m.model}: R² ${typeof m.metrics.R2 === "number" ? m.metrics.R2.toFixed(3) : "—"}`)
      .join(", ");
    return `Compare these models trained on ${datasetName} and tell me which one I should use and why: ${lines}.`;
  }, [modelsQuery.data, datasetName]);

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
          <CardContent className="p-10">
            <EmptyState
              icon={GitCompare}
              title="No datasets yet"
              description="Upload a dataset and train a couple of models to compare them here."
              action={{ label: "New Analysis", to: "/new/upload" }}
            />
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
            <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
              <CardTitle>Model Comparison</CardTitle>
              {comparePrompt && (
                <Button asChild size="sm" variant="outline">
                  <Link to="/assistant" search={{ prefill: comparePrompt }}>
                    <MessageCircleQuestion className="h-3.5 w-3.5 mr-1.5" /> Ask AI to Compare
                  </Link>
                </Button>
              )}
            </CardHeader>
            <CardContent>
              {modelsQuery.isLoading ? (
                <Skeleton className="h-32 w-full" />
              ) : !modelsQuery.data || modelsQuery.data.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No models trained on this dataset yet. Train one via New Analysis.
                </p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Model</TableHead>
                      <TableHead>R²</TableHead>
                      <TableHead>MAE</TableHead>
                      <TableHead>RMSE</TableHead>
                      <TableHead>MSE</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {modelsQuery.data.map((m) => {
                      const isBest = m.model_id === bestModelId;
                      return (
                        <TableRow key={m.model_id} className={isBest ? "bg-success/10" : undefined}>
                          <TableCell className="font-medium">
                            <span className="flex items-center gap-2">
                              {isBest && <Trophy className="h-3.5 w-3.5 text-success" />}
                              {m.model}
                              {m.source === "huggingface" && (
                                <Badge variant="outline" className="text-xs font-normal">
                                  Hugging Face
                                </Badge>
                              )}
                            </span>
                          </TableCell>
                          <TableCell>
                            <Badge variant={isBest ? "default" : "secondary"}>
                              {typeof m.metrics.R2 === "number" ? m.metrics.R2.toFixed(3) : "—"}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {typeof m.metrics.MAE === "number" ? m.metrics.MAE.toFixed(3) : "—"}
                          </TableCell>
                          <TableCell>
                            {typeof m.metrics.RMSE === "number" ? m.metrics.RMSE.toFixed(3) : "—"}
                          </TableCell>
                          <TableCell>
                            {typeof m.metrics.MSE === "number" ? m.metrics.MSE.toFixed(0) : "—"}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button asChild variant="ghost" size="sm">
                              <Link to="/models/$modelId" params={{ modelId: m.model_id }}>
                                View
                              </Link>
                            </Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
