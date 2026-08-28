import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Check, Cloud, Database, Monitor, Sparkles } from "lucide-react";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { GradientIcon } from "@/components/gradient-icon";
import { ScrollReveal } from "@/components/scroll-reveal";
import { LocalFilePanel } from "@/components/dataset-sources/local-file-panel";
import { ManualEditorPanel } from "@/components/dataset-sources/manual-editor-panel";
import { KaggleImportPanel } from "@/components/dataset-sources/kaggle-import-panel";
import { GoogleDrivePanel } from "@/components/dataset-sources/google-drive-panel";
import { RecentUploadsPanel } from "@/components/dataset-sources/recent-uploads-panel";
import { UploadInfoCards } from "@/components/dataset-sources/upload-info-cards";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { usePreviewDataset, useDatasetSourcesStatus } from "@/hooks/use-datasets";
import { applyNewClusteringDataset } from "@/lib/apply-new-clustering-dataset";
import { requireAuth } from "@/lib/require-auth";
import type { DatasetListItem, DatasetSourceType, DatasetSummary } from "@/lib/api-types";

export const Route = createFileRoute("/cluster/upload")({
  beforeLoad: requireAuth,
  component: UploadPage,
});

type SourceId = "local" | "google_drive" | "kaggle" | "manual";

const SOURCE_LABELS: Record<DatasetSourceType, string> = {
  local: "Local Computer",
  google_drive: "Google Drive",
  kaggle: "Kaggle",
  manual: "Created manually",
};

function UploadPage() {
  const { state, update } = useClusteringAnalysis();
  const navigate = useNavigate();
  const [activeSource, setActiveSource] = useState<SourceId | null>("local");
  const [datasetSource, setDatasetSource] = useState<DatasetSourceType>("local");

  const sourcesStatus = useDatasetSourcesStatus();
  const preview = usePreviewDataset(state.datasetId);

  const handleSelectRecent = (item: DatasetListItem) => {
    setDatasetSource(item.source ?? "local");
    update({ datasetId: item.dataset_id });
  };

  const handleNewDataset = (result: DatasetSummary) => {
    setDatasetSource(result.source ?? activeSource ?? "local");
    applyNewClusteringDataset(update, {
      datasetId: result.dataset_id,
      name: result.dataset_name,
      totalRows: result.rows,
      missingValues: result.missing_values,
      dataTypes: result.data_types,
    });
  };

  const cards: {
    id: SourceId;
    icon: typeof Monitor;
    gradient: string;
    title: string;
    description: string;
    formats?: string;
  }[] = [
    {
      id: "local",
      icon: Monitor,
      gradient: "from-indigo-500 to-blue-600",
      title: "Local Computer",
      description: "Browse files from your PC",
      formats: "CSV, XLSX, JSON, Parquet",
    },
    {
      id: "google_drive",
      icon: Cloud,
      gradient: "from-cyan-500 to-blue-600",
      title: "Google Drive",
      description: "Select a dataset from Drive",
    },
    {
      id: "kaggle",
      icon: Database,
      gradient: "from-emerald-500 to-teal-600",
      title: "Kaggle",
      description: "Fetch a public Kaggle dataset",
    },
    {
      id: "manual",
      icon: Sparkles,
      gradient: "from-amber-500 to-orange-600",
      title: "Create Manually",
      description: "Build a dataset from scratch",
    },
  ];

  const ds = preview.data;

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Add Dataset for Clustering</h1>
          <p className="text-muted-foreground text-sm">
            Choose where your dataset comes from or upload your files.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {cards.map((c, i) => {
            const selected = activeSource === c.id;
            return (
              <ScrollReveal key={c.id} delay={i * 60}>
                <Card
                  variant="glass"
                  className={`card-interactive cursor-pointer ${selected ? "border-primary ring-2 ring-primary/30" : ""}`}
                  onClick={() => setActiveSource(c.id)}
                >
                  <CardContent className="p-5 space-y-3">
                    <div className="flex items-start justify-between">
                      <GradientIcon icon={c.icon} className={`bg-gradient-to-br ${c.gradient}`} />
                      {selected && <Check className="h-4 w-4 text-primary" />}
                    </div>
                    <div>
                      <h3 className="font-semibold">{c.title}</h3>
                      <p className="text-sm text-muted-foreground">{c.description}</p>
                      {c.formats && <p className="text-xs text-muted-foreground mt-1">{c.formats}</p>}
                    </div>
                  </CardContent>
                </Card>
              </ScrollReveal>
            );
          })}
        </div>

        <div className="grid gap-4 lg:grid-cols-5">
          <div className="lg:col-span-3">
            {activeSource === "local" && <LocalFilePanel onUploaded={handleNewDataset} />}
            {activeSource === "manual" && <ManualEditorPanel onCreated={handleNewDataset} />}
            {activeSource === "kaggle" && <KaggleImportPanel onImported={handleNewDataset} />}
            {activeSource === "google_drive" && (
              <GoogleDrivePanel
                configured={sourcesStatus.data?.google_drive_configured ?? false}
                onImported={handleNewDataset}
              />
            )}
          </div>
          <div className="lg:col-span-2">
            <RecentUploadsPanel onSelect={handleSelectRecent} />
          </div>
        </div>

        <UploadInfoCards taskType="clustering" />

        {state.datasetId && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle>
                {ds ? "Dataset Ready" : "Loading dataset…"} — {ds?.name ?? state.dataset?.name}
              </CardTitle>
              {ds && <SourceBadge source={ds.source ?? datasetSource} />}
            </CardHeader>
            <CardContent className="space-y-4">
              {preview.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-40 w-full" />
                </div>
              ) : preview.isError ? (
                <p className="text-sm text-destructive">
                  Failed to load dataset preview. Try re-uploading.
                </p>
              ) : ds ? (
                <>
                  <div className="grid gap-3 sm:grid-cols-4">
                    <Stat label="Rows" value={ds.total_rows} />
                    <Stat label="Columns" value={ds.total_columns} />
                    <Stat label="Missing" value={ds.missing_values} />
                    <Stat
                      label="Numeric"
                      value={
                        Object.values(ds.data_types).filter(
                          (t) => t.includes("int") || t.includes("float"),
                        ).length
                      }
                    />
                  </div>
                  <div className="overflow-x-auto rounded-lg border max-h-80">
                    <table className="w-full text-sm">
                      <thead className="bg-muted sticky top-0">
                        <tr>
                          {ds.columns.map((c) => (
                            <th key={c} className="text-left p-2 font-medium">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {ds.rows.slice(0, 20).map((r, i) => (
                          <tr key={i} className="border-t">
                            {r.map((v, j) => (
                              <td key={j} className="p-2">
                                {v === null ? (
                                  <span className="text-muted-foreground">—</span>
                                ) : (
                                  String(v)
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {ds.total_rows > ds.rows.length && (
                    <p className="text-xs text-muted-foreground">
                      Previewing {Math.min(20, ds.rows.length)} of {ds.total_rows.toLocaleString()} rows.
                      The full dataset is used for clustering.
                    </p>
                  )}
                </>
              ) : null}
              <div className="flex justify-end gap-2">
                <Button asChild variant="outline">
                  <Link to="/">Cancel</Link>
                </Button>
                <Button
                  disabled={!ds}
                  onClick={() => {
                    if (ds)
                      update({
                        dataset: {
                          name: ds.name,
                          columns: ds.columns,
                          rows: ds.rows,
                          totalRows: ds.total_rows,
                        },
                      });
                    navigate({ to: "/cluster/variables" });
                  }}
                >
                  Continue to Feature Selection
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function SourceBadge({ source }: { source: DatasetSourceType }) {
  return <Badge variant="secondary">{SOURCE_LABELS[source]}</Badge>;
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}
