import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Loader2, PlayCircle, XCircle } from "lucide-react";
import { WizardSteps, classificationSteps } from "@/components/wizard-steps";
import { useClassificationAnalysis, toBackendSplitConfig } from "@/lib/classification-analysis-store";
import { useTrainClassificationModel } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { toast } from "sonner";

export const Route = createFileRoute("/classify/train")({
  beforeLoad: requireAuth,
  component: TrainPage,
});

// These keys carry structured (array/object) data, not a single scalar
// value — they get their own dedicated visualizations elsewhere (Visualize
// step's confusion matrix, Explain step's per-class table), so they're
// excluded from this page's flat metric-badge summary.
const NON_SCALAR_METRIC_KEYS = new Set(["ConfusionMatrix", "Classes", "ClassificationReport"]);

function TrainPage() {
  const { state, update } = useClassificationAnalysis();
  const navigate = useNavigate();
  const trainModel = useTrainClassificationModel();

  const canTrain = !!state.datasetId && !!state.target && !!state.model;

  if (!canTrain) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
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
            classes: result.metrics.Classes ?? null,
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
      <WizardSteps steps={classificationSteps} />
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
                  ? "Training Classification Model…"
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
                {Object.entries(state.results)
                  .filter(([k]) => !NON_SCALAR_METRIC_KEYS.has(k))
                  .map(([k, v]) => (
                    <Badge key={k} variant="secondary">
                      {k}: {typeof v === "number" ? (k === "Accuracy" || k === "Precision" || k === "Recall" || k === "F1" ? v.toFixed(3) : v.toFixed(4)) : "—"}
                    </Badge>
                  ))}
              </div>
            </div>
          )}

          <div className="flex justify-end">
            <Button disabled={!done} onClick={() => navigate({ to: "/classify/predict" })}>
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
