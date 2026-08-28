import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { ScrollReveal } from "@/components/scroll-reveal";
import { useDatasetsList, useDeleteDataset } from "@/hooks/use-datasets";
import { useAnalysis } from "@/lib/analysis-store";
import { useClassificationAnalysis } from "@/lib/classification-analysis-store";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import type { DatasetListItem, DatasetSourceType } from "@/lib/api-types";
import { Database, Shapes, Trash2, TrendingUp } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/data-sources")({
  beforeLoad: requireAuth,
  component: DataSourcesPage,
});

const SOURCE_LABELS: Record<DatasetSourceType, string> = {
  local: "Local Computer",
  google_drive: "Google Drive",
  kaggle: "Kaggle",
  manual: "Created manually",
};

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unitIdx = 0;
  while (value >= 1024 && unitIdx < units.length - 1) {
    value /= 1024;
    unitIdx += 1;
  }
  return `${value.toFixed(1)} ${units[unitIdx]}`;
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function DataSourcesPage() {
  const { data, isLoading, isError } = useDatasetsList();
  const deleteMutation = useDeleteDataset();
  const { update: updateRegression } = useAnalysis();
  const { update: updateClassification } = useClassificationAnalysis();
  const navigate = useNavigate();

  const startRegression = (item: DatasetListItem) => {
    updateRegression({ datasetId: item.dataset_id });
    navigate({ to: "/new/upload" });
  };

  const startClassification = (item: DatasetListItem) => {
    updateClassification({ datasetId: item.dataset_id });
    navigate({ to: "/classify/upload" });
  };

  const handleDelete = async (item: DatasetListItem) => {
    try {
      await deleteMutation.mutateAsync(item.dataset_id);
      toast.success(`Deleted ${item.name}`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to delete dataset");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Data Sources</h1>
        <p className="text-sm text-muted-foreground">
          Every dataset you've uploaded, imported, or created — start a new analysis from any of
          them, or clean up ones you no longer need.
        </p>
      </div>

      <ScrollReveal>
        <Card>
          <CardHeader>
            <CardTitle>Your Datasets</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : isError ? (
              <p className="text-sm text-destructive">Couldn't load your datasets.</p>
            ) : !data || data.length === 0 ? (
              <EmptyState
                icon={Database}
                title="No datasets yet"
                description="Upload a dataset to start your first regression or classification analysis."
                action={{ label: "Upload Dataset", to: "/new/upload" }}
              />
            ) : (
              <div className="space-y-2">
                {data.map((item) => (
                  <div
                    key={item.dataset_id}
                    className="flex flex-col sm:flex-row sm:items-center gap-3 rounded-lg border p-3"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <p className="text-sm font-medium truncate">{item.name}</p>
                        {item.source && (
                          <Badge variant="secondary" className="text-[10px]">
                            {SOURCE_LABELS[item.source]}
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {item.rows.toLocaleString()} rows · {item.columns} cols ·{" "}
                        {formatBytes(item.size_bytes)} · {formatDate(item.created_at)}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <Button size="sm" variant="outline" onClick={() => startRegression(item)}>
                        <TrendingUp className="h-3.5 w-3.5 mr-1.5" />
                        Regression
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => startClassification(item)}>
                        <Shapes className="h-3.5 w-3.5 mr-1.5" />
                        Classification
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8 text-destructive hover:text-destructive"
                        aria-label={`Delete ${item.name}`}
                        disabled={deleteMutation.isPending}
                        onClick={() => handleDelete(item)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </ScrollReveal>

      <p className="text-xs text-muted-foreground">
        Need to import from Kaggle or Google Drive?{" "}
        <Link to="/new/upload" className="underline underline-offset-2">
          Start a new analysis
        </Link>{" "}
        — every import source lives there.
      </p>
    </div>
  );
}
