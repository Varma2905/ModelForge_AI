import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Loader2, PlayCircle, XCircle } from "lucide-react";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { useTrainClusteringModel } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { toast } from "sonner";

export const Route = createFileRoute("/cluster/train")({
  beforeLoad: requireAuth,
  component: TrainPage,
});

function TrainPage() {
  const { state, update } = useClusteringAnalysis();
  const navigate = useNavigate();
  const trainModel = useTrainClusteringModel();

  const canTrain = !!state.datasetId && !!state.model && state.features.length > 0;

  if (!canTrain) {
    return (
      <div>
        <WizardSteps steps={clusteringSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Please complete the earlier steps first.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const runTraining = () => {
    const start = Date.now();
    trainModel.mutate(
      {
        dataset_id: state.datasetId!,
        model: state.model!,
        features: state.features,
        hyperparameters: state.hyperparameters,
      },
      {
        onSuccess: (result) => {
          update({
            trained: true,
            trainingTimeMs: Date.now() - start,
            modelId: result.model_id,
            results: result.metrics,
          });
        },
        onError: (err) => {
          toast.error(err instanceof ApiError ? err.message : "Clustering failed. Please try again.");
        },
      },
    );
  };

  const isIdle = trainModel.isIdle && !state.trained;
  const isPending = trainModel.isPending;
  const isError = trainModel.isError;
  const done = state.trained;

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {isError ? (
              <XCircle className="h-5 w-5 text-destructive" />
            ) : done ? (
              <CheckCircle2 className="h-5 w-5 text-emerald-500" />
            ) : isPending ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <PlayCircle className="h-5 w-5 text-muted-foreground" />
            )}
            {isError
              ? "Clustering Failed"
              : done
                ? "Clustering Completed Successfully"
                : isPending
                  ? "Running Clustering…"
                  : "Ready to Cluster"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-sm">
            <Info
              label="Dataset size"
              value={`${state.dataset?.totalRows ?? state.dataset?.rows.length ?? 0} rows`}
            />
            <Info label="Features" value={state.features.join(", ") || "—"} />
            <Info label="Algorithm" value={state.model ?? "—"} />
            <Info
              label="Time"
              value={done ? `${(state.trainingTimeMs / 1000).toFixed(1)}s` : "…"}
            />
          </div>

          {isIdle && (
            <div className="rounded-lg border p-6 text-center">
              <p className="text-sm text-muted-foreground mb-4">
                Ready to run {state.model} on your selected features. No target column needed —
                clustering discovers groups on its own.
              </p>
              <Button onClick={runTraining}>
                <PlayCircle className="h-4 w-4 mr-2" />
                Start Clustering
              </Button>
            </div>
          )}

          {isPending && (
            <div className="space-y-2">
              <Skeleton className="h-4 w-1/3" />
              <Skeleton className="h-16 w-full" />
            </div>
          )}

          {isError && (
            <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
              {trainModel.error instanceof ApiError
                ? trainModel.error.message
                : "Something went wrong while running clustering."}
              <div className="mt-3">
                <Button size="sm" variant="outline" onClick={runTraining}>
                  Try again
                </Button>
              </div>
            </div>
          )}

          {done && (
            <div className="rounded-lg border bg-emerald-50 dark:bg-emerald-950/30 p-4">
              <div className="flex flex-wrap gap-2">
                {Object.entries(state.results).map(([k, v]) => (
                  <Badge key={k} variant="secondary">
                    {k}: {typeof v === "number" ? v.toFixed(4) : v === null ? "N/A" : String(v)}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-end">
            <Button disabled={!done} onClick={() => navigate({ to: "/cluster/analysis" })}>
              Continue to Analysis
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-medium truncate">{value}</div>
    </div>
  );
}
