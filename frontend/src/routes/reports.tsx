import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FileText, Download, Eye, Loader2, Trash2 } from "lucide-react";
import { useModelsList, useDeleteModel } from "@/hooks/use-training";
import { useDownloadReport } from "@/hooks/use-reports";
import { GradientIcon } from "@/components/gradient-icon";
import { EmptyState } from "@/components/empty-state";
import { ScrollReveal } from "@/components/scroll-reveal";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { downloadBlob } from "@/lib/download-blob";
import { toast } from "sonner";
import type { ModelListItem } from "@/lib/api-types";

export const Route = createFileRoute("/reports")({
  beforeLoad: requireAuth,
  component: ReportsPage,
});

function ReportCard({ model }: { model: ModelListItem }) {
  const downloadMutation = useDownloadReport();
  const previewMutation = useDownloadReport();
  const deleteMutation = useDeleteModel();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);

  // Object URLs are only valid for this tab's lifetime — release it once the
  // dialog is dismissed so the blob doesn't linger in memory.
  useEffect(() => {
    if (!previewOpen && previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewOpen, previewUrl]);

  const handleDownload = async () => {
    try {
      const blob = await downloadMutation.mutateAsync(model.model_id);
      downloadBlob(blob, `regression-report-${model.model_id}.pdf`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to generate PDF report");
    }
  };

  const handlePreview = async () => {
    try {
      // Same download endpoint as the button below — rendered inline instead
      // of triggering a save, not a separate/fake preview capability.
      const blob = await previewMutation.mutateAsync(model.model_id);
      setPreviewUrl(URL.createObjectURL(blob));
      setPreviewOpen(true);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to generate PDF report");
    }
  };

  const handleDelete = async () => {
    if (!window.confirm(`Delete the report for "${model.dataset_name} — ${model.model}"? This can't be undone.`)) {
      return;
    }
    try {
      await deleteMutation.mutateAsync(model.model_id);
      toast.success("Report deleted");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to delete report");
    }
  };

  return (
    <Card variant="glass" className="card-interactive">
      <CardContent className="p-5">
        <div className="flex items-start gap-3">
          <GradientIcon icon={FileText} />
          <div className="flex-1 min-w-0">
            <div className="font-medium truncate">
              {model.dataset_name} — {model.model}
            </div>
            <div className="text-xs text-muted-foreground">
              {model.created_at ? new Date(model.created_at).toLocaleDateString() : "—"}
            </div>
          </div>
        </div>
        <div className="flex gap-2 mt-4">
          <Button
            size="sm"
            variant="outline"
            className="flex-1"
            onClick={handlePreview}
            disabled={previewMutation.isPending}
          >
            {previewMutation.isPending ? (
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            ) : (
              <Eye className="h-3 w-3 mr-1" />
            )}
            Preview
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="flex-1"
            onClick={handleDownload}
            disabled={downloadMutation.isPending}
          >
            {downloadMutation.isPending ? (
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            ) : (
              <Download className="h-3 w-3 mr-1" />
            )}
            Download
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="text-destructive hover:text-destructive"
            aria-label={`Delete report for ${model.dataset_name} — ${model.model}`}
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Trash2 className="h-3 w-3" />
            )}
          </Button>
        </div>
      </CardContent>

      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-4xl h-[85vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>
              {model.dataset_name} — {model.model}
            </DialogTitle>
          </DialogHeader>
          {previewUrl && (
            <iframe src={previewUrl} title="Report preview" className="flex-1 w-full rounded-md border" />
          )}
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function ReportsPage() {
  const { data, isLoading, isError } = useModelsList();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Reports</h1>
        <p className="text-sm text-muted-foreground">
          Generate a PDF report for any trained model — real charts, statistics, and AI
          explanations.
        </p>
      </div>

      {isLoading ? (
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full rounded-xl" />
          ))}
        </div>
      ) : isError ? (
        <Card>
          <CardContent className="p-6 text-sm text-destructive">
            Couldn't load your reports.
          </CardContent>
        </Card>
      ) : !data || data.length === 0 ? (
        <Card>
          <CardContent className="p-10">
            <EmptyState
              icon={FileText}
              title="No reports yet"
              description="Train a model to generate your first PDF report."
              action={{ label: "New Analysis", to: "/new/upload" }}
            />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {data.map((m, i) => (
            <ScrollReveal key={m.model_id} delay={(i % 6) * 50}>
              <ReportCard model={m} />
            </ScrollReveal>
          ))}
        </div>
      )}
    </div>
  );
}
