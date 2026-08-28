import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Loader2, PlayCircle, XCircle } from "lucide-react";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis, toBackendSplitConfig } from "@/lib/analysis-store";
import { useTrainModel } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { toast } from "sonner";

export const Route = createFileRoute("/new/train")({
  beforeLoad: requireAuth,
  component: TrainPage,
});

function TrainPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const trainModel = useTrainModel();

  const canTrain = !!state.datasetId && !!state.target && !!state.model;

  if (!canTrain) {
    return (
      <div>
        <WizardSteps />
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
    const datasetId = state.preprocessedDatasetId ?? state.datasetId!;

    // Triggered from a real click handler (not a mount effect) — this is
    // React Query's own recommended pattern for mutations, and deliberately
    // so here: an effect-fired mutate() on first mount of this route proved
    // unreliable in this app (the request completes correctly, confirmed
    // server-side every time, but the resulting state update could silently
    // fail to reach this component — a known rough edge of firing mutations
    // from effects rather than event handlers). A click handler runs in a
    // plain synchronous event context with no such ambiguity.
    trainModel.mutate(
      {
        dataset_id: datasetId,
        model: state.model!,
        features: state.features,
        target: state.target!,
        split: toBackendSplitConfig(state.split),
      },
      {
        onSuccess: (result) => {
          update({
            trained: true,
            trainingTimeMs: Date.now() - start,
            modelId: result.model_id,
            results: result.metrics,
            trainedRowCounts: {
              total: result.total_rows,
              train: result.train_rows,
              test: result.test_rows,
              val: result.val_rows,
            },
          });
        },
        onError: (err) => {
          toast.error(err instanceof ApiError ? err.message : "Training failed. Please try again.");
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
      <WizardSteps />
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
              ? "Training Failed"
              : done
                ? "Model Trained Successfully"
                : isPending
                  ? "Training Regression Model…"
                  : "Ready to Train"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-sm">
            <Info
              label="Dataset size"
              value={`${state.trainedRowCounts?.total ?? state.dataset?.totalRows ?? state.dataset?.rows.length ?? 0} rows`}
            />
            <Info label="Features" value={state.features.join(", ") || "—"} />
            <Info label="Algorithm" value={state.model ?? "—"} />
            <Info
              label="Time"
              value={done ? `${(state.trainingTimeMs / 1000).toFixed(1)}s` : "…"}
            />
            {done && state.trainedRowCounts && (
              <>
                <Info label="Training rows" value={`${state.trainedRowCounts.train} rows`} />
                <Info label="Testing rows" value={`${state.trainedRowCounts.test} rows`} />
              </>
            )}
          </div>

          {isIdle && (
            <div className="rounded-lg border p-6 text-center">
              <p className="text-sm text-muted-foreground mb-4">
                Ready to fit {state.model} on your configured dataset.
              </p>
              <Button onClick={runTraining}>
                <PlayCircle className="h-4 w-4 mr-2" />
                Start Training
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
                : "Something went wrong while training the model."}
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
                    {k}: {typeof v === "number" ? v.toFixed(4) : "—"}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-end">
            <Button disabled={!done} onClick={() => navigate({ to: "/new/predict" })}>
              Continue to Prediction
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
