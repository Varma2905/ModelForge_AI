import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/metric-card";
import { WizardSteps, classificationSteps } from "@/components/wizard-steps";
import { useClassificationAnalysis } from "@/lib/classification-analysis-store";
import { useClassificationModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/classify/metrics")({
  beforeLoad: requireAuth,
  component: MetricsPage,
});

function fmtPct(v: number | undefined | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmt(v: number | undefined | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-medium text-lg mt-0.5">{value}</div>
    </div>
  );
}

function MetricsPage() {
  const { state } = useClassificationAnalysis();
  const navigate = useNavigate();
  const metricsQuery = useClassificationModelMetrics(state.modelId);

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first to see evaluation metrics.</p>
            <Button asChild className="mt-4">
              <Link to="/classify/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const metrics = metricsQuery.data?.metrics;
  const isBinary = (metrics?.Classes?.length ?? 0) === 2;

  const metricCards = [
    { label: "Accuracy", value: fmtPct(metrics?.Accuracy), gradient: "from-emerald-500 to-teal-500" },
    { label: "Precision", value: fmtPct(metrics?.Precision), gradient: "from-purple-500 to-violet-500" },
    { label: "Recall", value: fmtPct(metrics?.Recall), gradient: "from-indigo-500 to-blue-500" },
    { label: "F1-Score", value: fmt(metrics?.F1), gradient: "from-amber-500 to-orange-500" },
    { label: "ROC-AUC", value: fmt(metrics?.ROC_AUC), gradient: "from-fuchsia-500 to-pink-500" },
  ];

  return (
    <div>
      <WizardSteps steps={classificationSteps} />

      <div className="mb-6 flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-lg font-semibold">Evaluation Metrics</h2>
          <p className="text-sm text-muted-foreground">
            Calculated by the Python ML backend from {state.model ?? "your model"}'s predictions on the held-out test set.
          </p>
        </div>
        {metrics?.Classes && (
          <Badge variant="outline">
            {isBinary ? "Binary Classification" : "Multiclass Classification"} · {metrics.Classes.length} classes
          </Badge>
        )}
      </div>

      {metricsQuery.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5 mb-6">
            {metricCards.map((m) => (
              <MetricCard key={m.label} label={m.label} value={m.value} gradient={m.gradient} />
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Additional Classification Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <StatTile label="Balanced Accuracy" value={fmtPct(metrics?.BalancedAccuracy)} />
                <StatTile label="Matthews Correlation Coefficient" value={fmt(metrics?.MCC)} />
                <StatTile label="Log Loss" value={fmt(metrics?.LogLoss)} />
                {isBinary ? (
                  <StatTile label="Specificity" value={fmtPct(metrics?.Specificity)} />
                ) : (
                  <>
                    <StatTile label="Precision Macro" value={fmt(metrics?.["Precision Macro"])} />
                    <StatTile label="Recall Macro" value={fmt(metrics?.["Recall Macro"])} />
                    <StatTile label="F1 Macro" value={fmt(metrics?.["F1 Macro"])} />
                    <StatTile label="Precision Weighted" value={fmt(metrics?.["Precision Weighted"])} />
                    <StatTile label="Recall Weighted" value={fmt(metrics?.["Recall Weighted"])} />
                    <StatTile label="F1 Weighted" value={fmt(metrics?.["F1 Weighted"])} />
                  </>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                A metric shows "—" when it's mathematically undefined for this model (e.g. Log Loss requires
                predicted probabilities).
              </p>
            </CardContent>
          </Card>
        </>
      )}

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/classify/visualize" })}>Continue to Visualizations</Button>
      </div>
    </div>
  );
}
