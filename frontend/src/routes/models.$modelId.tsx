import { createFileRoute, Link } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/metric-card";
import { ChartCard } from "@/components/chart-card";
import { CorrelationMatrix } from "@/components/correlation-matrix";
import { ConfusionMatrix } from "@/components/confusion-matrix";
import { useModelMetrics } from "@/hooks/use-training";
import { useDownloadReport } from "@/hooks/use-reports";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { useChartTheme } from "@/lib/chart-theme";
import { downloadBlob } from "@/lib/download-blob";
import { ArrowLeft, Download, Loader2 } from "lucide-react";
import { toast } from "sonner";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
  Legend,
  BarChart,
  Bar,
} from "recharts";
import type { ClassificationChartData, ClassificationMetrics } from "@/lib/api-types";

export const Route = createFileRoute("/models/$modelId")({
  beforeLoad: requireAuth,
  component: ModelDetailPage,
});

function fmt(v: number | undefined) {
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

function ModelDetailPage() {
  const { modelId } = Route.useParams();
  const t = useChartTheme();
  // getModelMetrics()/GET /model-metrics/{id} is generic over any model
  // doc's model_type — the declared ModelMetricsResult return type is
  // regression-shaped, but a classification model doc's actual JSON
  // response carries a differently-shaped metrics/chart_data (confusion
  // matrix instead of residuals, etc.), checked and cast below via
  // model_type rather than trusted from the static type alone.
  const metricsQuery = useModelMetrics(modelId);
  const downloadReport = useDownloadReport();

  const download = async () => {
    try {
      const blob = await downloadReport.mutateAsync(modelId);
      downloadBlob(blob, `model-report-${modelId}.pdf`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to generate PDF report");
    }
  };

  if (metricsQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (metricsQuery.isError || !metricsQuery.data) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <p>This model couldn't be found.</p>
          <Button asChild className="mt-4">
            <Link to="/history">Back to history</Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  const model = metricsQuery.data;
  const isClassification = model.model_type === "classification";
  const axisProps = { stroke: t.mutedForeground, tick: t.axisTick };

  const header = (
    <div className="flex items-center justify-between flex-wrap gap-3">
      <div>
        <Button asChild variant="ghost" size="sm" className="mb-2 -ml-2">
          <Link to="/history">
            <ArrowLeft className="h-4 w-4 mr-1" /> Back to saved models
          </Link>
        </Button>
        <h1 className="text-2xl font-bold">{model.model}</h1>
        <p className="text-sm text-muted-foreground">
          Target: {model.target} · Features: {model.features.join(", ")}
        </p>
      </div>
      <Button onClick={download} disabled={downloadReport.isPending}>
        {downloadReport.isPending ? (
          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
        ) : (
          <Download className="h-4 w-4 mr-2" />
        )}
        Download PDF Report
      </Button>
    </div>
  );

  if (isClassification) {
    // Cast, not re-fetched — same JSON payload, just a differently-shaped
    // metrics/chart_data than the regression-typed ModelMetricsResult.
    const classificationMetrics = model.metrics as unknown as ClassificationMetrics;
    const chartData = model.chart_data as unknown as ClassificationChartData;
    const importance = chartData.feature_importance ?? [];

    return (
      <div className="space-y-6">
        {header}

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <MetricCard label="Accuracy" value={fmtPct(classificationMetrics.Accuracy)} gradient="from-emerald-500 to-teal-500" />
          <MetricCard label="Precision" value={fmtPct(classificationMetrics.Precision)} gradient="from-purple-500 to-violet-500" />
          <MetricCard label="Recall" value={fmtPct(classificationMetrics.Recall)} gradient="from-indigo-500 to-blue-500" />
          <MetricCard label="F1-Score" value={classificationMetrics.F1?.toFixed(3) ?? "—"} gradient="from-amber-500 to-orange-500" />
          <MetricCard label="ROC-AUC" value={classificationMetrics.ROC_AUC?.toFixed(3) ?? "—"} gradient="from-fuchsia-500 to-pink-500" />
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Confusion Matrix</CardTitle>
            </CardHeader>
            <CardContent>
              <ConfusionMatrix data={chartData.confusion_matrix} />
            </CardContent>
          </Card>

          {importance.length > 0 ? (
            <ChartCard title="Feature Importance">
              <BarChart data={importance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
                <XAxis type="number" {...axisProps} />
                <YAxis type="category" dataKey="feature" {...axisProps} width={100} />
                <Tooltip
                  contentStyle={t.tooltipStyle}
                  labelStyle={t.tooltipLabelStyle}
                  itemStyle={t.tooltipItemStyle}
                  cursor={{ fill: t.mutedForeground, opacity: 0.1 }}
                />
                <Bar dataKey="value" fill={t.series.primary} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ChartCard>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Feature Importance</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">No importance data available.</p>
              </CardContent>
            </Card>
          )}
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Correlation Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <CorrelationMatrix data={chartData.correlation_matrix} />
          </CardContent>
        </Card>
      </div>
    );
  }

  const chartData = model.chart_data;
  const actualVsPredicted = chartData.actual.map((a, i) => ({
    i,
    actual: a,
    predicted: chartData.predicted[i],
  }));
  const residualPoints = chartData.residuals.map((r, i) => ({ i, residual: r }));

  return (
    <div className="space-y-6">
      {header}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <MetricCard
          label="R² Score"
          value={fmt(model.metrics.R2)}
          gradient="from-emerald-500 to-teal-500"
        />
        <MetricCard
          label="Adjusted R²"
          value={fmt(model.metrics["Adjusted R2"])}
          gradient="from-purple-500 to-violet-500"
        />
        <MetricCard
          label="RMSE"
          value={fmt(model.metrics.RMSE)}
          gradient="from-indigo-500 to-blue-500"
        />
        <MetricCard
          label="MAE"
          value={fmt(model.metrics.MAE)}
          gradient="from-amber-500 to-orange-500"
        />
        <MetricCard
          label="MSE"
          value={fmt(model.metrics.MSE)}
          gradient="from-fuchsia-500 to-pink-500"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Additional Error &amp; Statistical Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="MAPE" value={fmtPct(model.metrics.MAPE)} />
            <StatTile label="MSLE" value={fmt(model.metrics.MSLE)} />
            <StatTile label="RMSLE" value={fmt(model.metrics.RMSLE)} />
            <StatTile label="Median Absolute Error" value={fmt(model.metrics["Median Absolute Error"])} />
            <StatTile label="Max Error" value={fmt(model.metrics["Max Error"])} />
            <StatTile label="Explained Variance" value={fmt(model.metrics["Explained Variance"])} />
            <StatTile label="Pearson Correlation" value={fmt(model.metrics["Pearson Correlation"])} />
            <StatTile label="Spearman Correlation" value={fmt(model.metrics["Spearman Correlation"])} />
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            A metric shows "—" when it's mathematically undefined for this model's actual/predicted values
            (e.g. MSLE requires non-negative values; correlation requires a non-constant prediction).
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Actual vs Predicted (test set)">
          <LineChart data={actualVsPredicted}>
            <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
            <XAxis dataKey="i" {...axisProps} />
            <YAxis {...axisProps} />
            <Tooltip
              contentStyle={t.tooltipStyle}
              labelStyle={t.tooltipLabelStyle}
              itemStyle={t.tooltipItemStyle}
            />
            <Legend wrapperStyle={t.legendStyle} />
            <Line
              type="monotone"
              dataKey="actual"
              stroke={t.series.success}
              dot={false}
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke={t.series.primary}
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ChartCard>
        <ChartCard title="Residual Plot (test set)">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
            <XAxis type="number" dataKey="i" {...axisProps} />
            <YAxis type="number" dataKey="residual" {...axisProps} />
            <Tooltip
              contentStyle={t.tooltipStyle}
              labelStyle={t.tooltipLabelStyle}
              itemStyle={t.tooltipItemStyle}
            />
            <Scatter data={residualPoints} fill={t.series.danger} />
          </ScatterChart>
        </ChartCard>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Correlation Matrix</CardTitle>
        </CardHeader>
        <CardContent>
          <CorrelationMatrix data={chartData.correlation_matrix} />
        </CardContent>
      </Card>
    </div>
  );
}
