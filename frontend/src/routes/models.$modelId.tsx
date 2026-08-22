import { createFileRoute, Link } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/metric-card";
import { ChartCard } from "@/components/chart-card";
import { CorrelationMatrix } from "@/components/correlation-matrix";
import { useModelMetrics } from "@/hooks/use-training";
import { useDownloadReport } from "@/hooks/use-reports";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { useChartTheme } from "@/lib/chart-theme";
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
} from "recharts";

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

function ModelDetailPage() {
  const { modelId } = Route.useParams();
  const t = useChartTheme();
  const metricsQuery = useModelMetrics(modelId);
  const downloadReport = useDownloadReport();

  const download = async () => {
    try {
      const blob = await downloadReport.mutateAsync(modelId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `regression-report-${modelId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
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
  const chartData = model.chart_data;
  const actualVsPredicted = chartData.actual.map((a, i) => ({
    i,
    actual: a,
    predicted: chartData.predicted[i],
  }));
  const residualPoints = chartData.residuals.map((r, i) => ({ i, residual: r }));
  const axisProps = { stroke: t.mutedForeground, tick: t.axisTick };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <Button asChild variant="ghost" size="sm" className="mb-2 -ml-2">
            <Link to="/history">
              <ArrowLeft className="h-4 w-4 mr-1" /> Back to history
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
