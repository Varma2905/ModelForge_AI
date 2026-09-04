import { createFileRoute, Link } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { useDownloadReport } from "@/hooks/use-reports";
import { requireAuth } from "@/lib/require-auth";
import { downloadBlob } from "@/lib/download-blob";
import { Download, FileText, Loader2 } from "lucide-react";
import { toast } from "sonner";
import type { ReactNode } from "react";

export const Route = createFileRoute("/cluster/report")({
  beforeLoad: requireAuth,
  component: ReportPage,
});

function ReportPage() {
  const { state } = useClusteringAnalysis();
  const downloadReport = useDownloadReport();

  const download = async () => {
    if (!state.modelId) return;
    try {
      const blob = await downloadReport.mutateAsync(state.modelId);
      downloadBlob(blob, `clustering-report-${state.modelId}.pdf`);
      toast.success("PDF report downloaded");
    } catch (err) {
      console.error("Report generation failed:", err);
      toast.error("Report generation failed. Please try again.");
    }
  };

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps steps={clusteringSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Run clustering first to generate a report.</p>
            <Button asChild className="mt-4">
              <Link to="/cluster/train">Go to clustering</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between flex-wrap gap-2">
            <span className="flex items-center gap-2">
              <FileText className="h-5 w-5" /> Report Preview
            </span>
            <Button onClick={download} disabled={downloadReport.isPending}>
              {downloadReport.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Download className="h-4 w-4 mr-2" />
              )}
              {downloadReport.isPending ? "Building report…" : "Download PDF Report"}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <Sec title="Dataset Information">
            <p>Name: {state.dataset?.name ?? "—"}</p>
            <p>
              Rows: {state.dataset?.totalRows ?? state.dataset?.rows.length ?? 0} · Columns:{" "}
              {state.dataset?.columns.length ?? 0}
            </p>
          </Sec>
          <Sec title="Selected Features">
            <div className="flex flex-wrap gap-1">
              {state.features.map((f) => (
                <Badge key={f} variant="secondary">
                  {f}
                </Badge>
              ))}
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Clustering is unsupervised — no target column is used.
            </p>
          </Sec>
          <Sec title="Algorithm">
            <p>{state.model ?? "—"}</p>
            {Object.keys(state.hyperparameters).length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {Object.entries(state.hyperparameters).map(([k, v]) => (
                  <Badge key={k} variant="outline" className="text-xs font-normal">
                    {k}: {String(v)}
                  </Badge>
                ))}
              </div>
            )}
          </Sec>
          <Sec title="Clustering Metrics">
            <div className="flex flex-wrap gap-2">
              {Object.entries(state.results).map(([k, v]) => (
                <Badge key={k} variant="outline">
                  {k}: {typeof v === "number" ? v.toFixed(4) : v === null ? "N/A" : String(v)}
                </Badge>
              ))}
            </div>
          </Sec>
          <p className="text-xs text-muted-foreground">
            The downloaded PDF includes the dataset overview, feature summary, exploratory
            visualizations, and the AI-generated cluster interpretation — generated fresh from
            your clustering run.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function Sec({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h3 className="font-semibold text-sm mb-2 uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      <div className="text-sm">{children}</div>
    </div>
  );
}
