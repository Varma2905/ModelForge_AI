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
import { ScrollReveal } from "@/components/scroll-reveal";
import { useDatasetsList, useDatasetModels } from "@/hooks/use-datasets";
import { requireAuth } from "@/lib/require-auth";
import { GitCompare, MessageCircleQuestion, Trophy } from "lucide-react";
import type { ClassificationMetrics, ClusteringMetrics, DatasetModelListItem, ModelMetrics } from "@/lib/api-types";

export const Route = createFileRoute("/compare")({
  beforeLoad: requireAuth,
  component: ComparePage,
});

// A DatasetModelListItem's `metrics` field is declared as the regression
// ModelMetrics shape, but the actual JSON for a classification/clustering
// model doc carries that task's own metric keys instead.
type MixedMetrics = ModelMetrics & Partial<ClassificationMetrics> & Partial<ClusteringMetrics>;

function isClassification(m: DatasetModelListItem): boolean {
  return m.model_type === "classification";
}

function isClustering(m: DatasetModelListItem): boolean {
  return m.model_type === "clustering";
}

function RegressionTable({ models }: { models: DatasetModelListItem[] }) {
  const best = models.reduce(
    (best, m) => ((m.metrics as MixedMetrics).R2 > (best.metrics as MixedMetrics).R2 ? m : best),
    models[0],
  );
  const bestModelId = best.model_id;
  const bestR2 = (best.metrics as MixedMetrics).R2;

  return (
    <>
      {typeof bestR2 === "number" && (
        <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5">
          <Trophy className="h-3.5 w-3.5 text-success shrink-0" />
          <span>
            <span className="font-medium text-foreground">{best.model}</span> achieved the highest R² (
            {bestR2.toFixed(3)}) among the trained models — R² is higher-is-better, so this is currently the
            best-explaining fit for this dataset.
          </span>
        </p>
      )}
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
        {models.map((m) => {
          const isBest = m.model_id === bestModelId;
          const metrics = m.metrics as MixedMetrics;
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
                  {typeof metrics.R2 === "number" ? metrics.R2.toFixed(3) : "—"}
                </Badge>
              </TableCell>
              <TableCell>{typeof metrics.MAE === "number" ? metrics.MAE.toFixed(3) : "—"}</TableCell>
              <TableCell>{typeof metrics.RMSE === "number" ? metrics.RMSE.toFixed(3) : "—"}</TableCell>
              <TableCell>{typeof metrics.MSE === "number" ? metrics.MSE.toFixed(0) : "—"}</TableCell>
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
    </>
  );
}

// The comparison's default primary metric — F1 for a binary model, F1 Macro
// for multiclass (falls back to plain F1 for a model trained before "F1
// Macro" existed). Determined per-model from its own class count, since a
// dataset's trained models aren't guaranteed to all share the same target.
function primaryMetric(m: DatasetModelListItem): { label: string; value: number | null } {
  const metrics = m.metrics as MixedMetrics;
  const isBinary = (metrics.Classes?.length ?? 0) === 2;
  if (isBinary) return { label: "F1 Score", value: typeof metrics.F1 === "number" ? metrics.F1 : null };
  const macro = metrics["F1 Macro"];
  return {
    label: "F1 Macro",
    value: typeof macro === "number" ? macro : typeof metrics.F1 === "number" ? metrics.F1 : null,
  };
}

function ClassificationTable({ models }: { models: DatasetModelListItem[] }) {
  const best = models.reduce((best, m) => {
    const v = primaryMetric(m).value;
    const bestV = primaryMetric(best).value;
    return v !== null && (bestV === null || v > bestV) ? m : best;
  }, models[0]);
  const bestModelId = best.model_id;
  const bestPrimary = primaryMetric(best);

  return (
    <>
      {bestPrimary.value !== null && (
        <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5">
          <Trophy className="h-3.5 w-3.5 text-success shrink-0" />
          <span>
            <span className="font-medium text-foreground">{best.model}</span> achieved the highest{" "}
            {bestPrimary.label} ({bestPrimary.value.toFixed(3)}) among the trained models.
          </span>
        </p>
      )}
      <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Accuracy</TableHead>
          <TableHead>Precision</TableHead>
          <TableHead>Recall</TableHead>
          <TableHead>F1</TableHead>
          <TableHead>ROC-AUC</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.map((m) => {
          const isBest = m.model_id === bestModelId;
          const metrics = m.metrics as MixedMetrics;
          return (
            <TableRow key={m.model_id} className={isBest ? "bg-success/10" : undefined}>
              <TableCell className="font-medium">
                <span className="flex items-center gap-2">
                  {isBest && <Trophy className="h-3.5 w-3.5 text-success" />}
                  {m.model}
                </span>
              </TableCell>
              <TableCell>
                <Badge variant={isBest ? "default" : "secondary"}>
                  {typeof metrics.Accuracy === "number" ? `${(metrics.Accuracy * 100).toFixed(1)}%` : "—"}
                </Badge>
              </TableCell>
              <TableCell>{typeof metrics.Precision === "number" ? metrics.Precision.toFixed(3) : "—"}</TableCell>
              <TableCell>{typeof metrics.Recall === "number" ? metrics.Recall.toFixed(3) : "—"}</TableCell>
              <TableCell>{typeof metrics.F1 === "number" ? metrics.F1.toFixed(3) : "—"}</TableCell>
              <TableCell>{typeof metrics.ROC_AUC === "number" ? metrics.ROC_AUC.toFixed(3) : "—"}</TableCell>
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
    </>
  );
}

function ClusteringTable({ models }: { models: DatasetModelListItem[] }) {
  const withSilhouette = models.filter((m) => typeof (m.metrics as MixedMetrics).Silhouette === "number");
  const best =
    withSilhouette.length > 0
      ? withSilhouette.reduce((best, m) =>
          (m.metrics as MixedMetrics).Silhouette! > (best.metrics as MixedMetrics).Silhouette! ? m : best,
        )
      : null;
  const bestModelId = best?.model_id;
  const bestSilhouette = best ? (best.metrics as MixedMetrics).Silhouette! : null;

  return (
    <>
      {best && bestSilhouette !== null && (
        <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5">
          <Trophy className="h-3.5 w-3.5 text-success shrink-0" />
          <span>
            <span className="font-medium text-foreground">{best.model}</span> achieved the highest Silhouette
            Score ({bestSilhouette.toFixed(3)}) among the trained models — higher indicates better-separated
            clusters.
          </span>
        </p>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Algorithm</TableHead>
            <TableHead>Clusters</TableHead>
            <TableHead>Silhouette</TableHead>
            <TableHead>Davies-Bouldin</TableHead>
            <TableHead>Calinski-Harabasz</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((m) => {
            const isBest = m.model_id === bestModelId;
            const metrics = m.metrics as MixedMetrics;
            return (
              <TableRow key={m.model_id} className={isBest ? "bg-success/10" : undefined}>
                <TableCell className="font-medium">
                  <span className="flex items-center gap-2">
                    {isBest && <Trophy className="h-3.5 w-3.5 text-success" />}
                    {m.model}
                  </span>
                </TableCell>
                <TableCell>{typeof metrics.ClusterCount === "number" ? metrics.ClusterCount : "—"}</TableCell>
                <TableCell>
                  <Badge variant={isBest ? "default" : "secondary"}>
                    {typeof metrics.Silhouette === "number" ? metrics.Silhouette.toFixed(3) : "—"}
                  </Badge>
                </TableCell>
                <TableCell>{typeof metrics.DaviesBouldin === "number" ? metrics.DaviesBouldin.toFixed(3) : "—"}</TableCell>
                <TableCell>{typeof metrics.CalinskiHarabasz === "number" ? metrics.CalinskiHarabasz.toFixed(1) : "—"}</TableCell>
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
    </>
  );
}

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

  const regressionModels = modelsQuery.data?.filter((m) => !isClassification(m) && !isClustering(m)) ?? [];
  const classificationModels = modelsQuery.data?.filter(isClassification) ?? [];
  const clusteringModels = modelsQuery.data?.filter(isClustering) ?? [];
  const multipleSections =
    [regressionModels.length > 0, classificationModels.length > 0, clusteringModels.length > 0].filter(Boolean)
      .length > 1;

  const comparePrompt = useMemo(() => {
    if (!modelsQuery.data || modelsQuery.data.length === 0 || !datasetName) return null;
    const lines = modelsQuery.data
      .map((m) => {
        const metrics = m.metrics as MixedMetrics;
        return isClassification(m)
          ? `${m.model}: Accuracy ${typeof metrics.Accuracy === "number" ? metrics.Accuracy.toFixed(3) : "—"}`
          : isClustering(m)
            ? `${m.model}: Silhouette ${typeof metrics.Silhouette === "number" ? metrics.Silhouette.toFixed(3) : "—"}`
            : `${m.model}: R² ${typeof metrics.R2 === "number" ? metrics.R2.toFixed(3) : "—"}`;
      })
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

          <ScrollReveal>
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
            <CardContent className="space-y-6">
              {modelsQuery.isLoading ? (
                <Skeleton className="h-32 w-full" />
              ) : !modelsQuery.data || modelsQuery.data.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No models trained on this dataset yet. Train one via New Analysis.
                </p>
              ) : (
                <>
                  {regressionModels.length > 0 && (
                    <div>
                      {multipleSections && (
                        <h3 className="text-sm font-semibold mb-2">Regression Models</h3>
                      )}
                      <RegressionTable models={regressionModels} />
                    </div>
                  )}
                  {classificationModels.length > 0 && (
                    <div>
                      {multipleSections && (
                        <h3 className="text-sm font-semibold mb-2">Classification Models</h3>
                      )}
                      <ClassificationTable models={classificationModels} />
                    </div>
                  )}
                  {clusteringModels.length > 0 && (
                    <div>
                      {multipleSections && (
                        <h3 className="text-sm font-semibold mb-2">Clustering Models</h3>
                      )}
                      <ClusteringTable models={clusteringModels} />
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
          </ScrollReveal>
        </>
      )}
    </div>
  );
}
