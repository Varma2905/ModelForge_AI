import { createFileRoute, Link } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { useDownloadReport } from "@/hooks/use-reports";
import { requireAuth } from "@/lib/require-auth";
import { downloadBlob } from "@/lib/download-blob";
import { Download, FileText, Loader2 } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/new/report")({
  beforeLoad: requireAuth,
  component: ReportPage,
});

function ReportPage() {
  const { state } = useAnalysis();
  const downloadReport = useDownloadReport();

  const download = async () => {
    if (!state.modelId) return;
    try {
      const blob = await downloadReport.mutateAsync(state.modelId);
      downloadBlob(blob, `regression-report-${state.modelId}.pdf`);
      toast.success("PDF report downloaded");
    } catch (err) {
      // The detailed error stays in the console for debugging — the user
      // never sees a raw backend exception message.
      console.error("Report generation failed:", err);
      toast.error("Report generation failed. Please try again.");
    }
  };

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first to generate a report.</p>
            <Button asChild className="mt-4">
              <Link to="/new/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <WizardSteps />
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
            <p className="mt-2">
              Target: <Badge>{state.target ?? "—"}</Badge>
            </p>
          </Sec>
          <Sec title="Preprocessing">
            <p>
              Missing: {state.preprocessing.missing} · Dedupe: {String(state.preprocessing.dedupe)}{" "}
              · Outlier: {state.preprocessing.outlier} · Scaling: {state.preprocessing.scaling}
            </p>
          </Sec>
          <Sec title="Split Ratio">
            <p>
              Train {state.split.train}% · Test{" "}
              {100 - state.split.train - (state.split.useVal ? state.split.val : 0)}%
              {state.split.useVal ? ` · Val ${state.split.val}%` : ""}
            </p>
            {state.trainedRowCounts && (
              <p className="text-muted-foreground">
                {state.trainedRowCounts.total} total rows · {state.trainedRowCounts.train} training
                rows · {state.trainedRowCounts.test} testing rows
                {state.trainedRowCounts.val > 0 ? ` · ${state.trainedRowCounts.val} validation rows` : ""}
              </p>
            )}
          </Sec>
          <Sec title="Algorithm">
            <p>{state.model ?? "—"}</p>
          </Sec>
          <Sec title="Evaluation Metrics">
            <div className="flex flex-wrap gap-2">
              {Object.entries(state.results).map(([k, v]) => (
                <Badge key={k} variant="outline">
                  {k}: {typeof v === "number" ? v.toFixed(4) : String(v)}
                </Badge>
              ))}
            </div>
          </Sec>
          <Sec title="Prediction">
            <p>
              {state.prediction !== null
                ? `Sample predicted value: ${state.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })}`
                : "No prediction made yet"}
            </p>
          </Sec>
          <p className="text-xs text-muted-foreground">
            The downloaded PDF includes the full statistical analysis, diagnostic charts, and the
            AI-generated explanation report — generated fresh from your trained model.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function Sec({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="font-semibold text-sm mb-2 uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      <div className="text-sm">{children}</div>
    </div>
  );
}
