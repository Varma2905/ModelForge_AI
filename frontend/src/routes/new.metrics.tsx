import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/metric-card";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { useModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/new/metrics")({
  beforeLoad: requireAuth,
  component: MetricsPage,
});

function fmt(v: number | undefined | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return Math.abs(v) >= 1000
    ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : v.toFixed(3);
}

function fmtPct(v: number | undefined | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
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
  const { state } = useAnalysis();
  const navigate = useNavigate();
  const metricsQuery = useModelMetrics(state.modelId);

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first to see evaluation metrics.</p>
            <Button asChild className="mt-4">
              <Link to="/new/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const metrics = metricsQuery.data?.metrics ?? state.results;

  return (
    <div>
      <WizardSteps />

      <div className="mb-6">
        <h2 className="text-lg font-semibold">Evaluation Metrics</h2>
        <p className="text-sm text-muted-foreground">
          Calculated by the Python ML backend from {state.model ?? "your model"}'s predictions on the held-out test set.
        </p>
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
            <MetricCard label="R² Score" value={fmt(metrics.R2)} gradient="from-emerald-500 to-teal-500" />
            <MetricCard
              label="Adjusted R²"
              value={fmt(metrics["Adjusted R2"])}
              gradient="from-purple-500 to-violet-500"
            />
            <MetricCard label="RMSE" value={fmt(metrics.RMSE)} gradient="from-indigo-500 to-blue-500" />
            <MetricCard label="MAE" value={fmt(metrics.MAE)} gradient="from-amber-500 to-orange-500" />
            <MetricCard label="MSE" value={fmt(metrics.MSE)} gradient="from-fuchsia-500 to-pink-500" />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Additional Error &amp; Statistical Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <StatTile label="MAPE" value={fmtPct(metrics.MAPE)} />
                <StatTile label="MSLE" value={fmt(metrics.MSLE)} />
                <StatTile label="RMSLE" value={fmt(metrics.RMSLE)} />
                <StatTile label="Median Absolute Error" value={fmt(metrics["Median Absolute Error"])} />
                <StatTile label="Max Error" value={fmt(metrics["Max Error"])} />
                <StatTile label="Explained Variance" value={fmt(metrics["Explained Variance"])} />
                <StatTile label="Pearson Correlation" value={fmt(metrics["Pearson Correlation"])} />
                <StatTile label="Spearman Correlation" value={fmt(metrics["Spearman Correlation"])} />
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                A metric shows "—" when it's mathematically undefined for this model's actual/predicted values
                (e.g. MSLE requires non-negative values; correlation requires a non-constant prediction).
              </p>
            </CardContent>
          </Card>
        </>
      )}

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/new/visualize" })}>Continue to Visualizations</Button>
      </div>
    </div>
  );
}
