import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { WizardSteps } from "@/components/wizard-steps";
import { ChartCard } from "@/components/chart-card";
import { MetricCard } from "@/components/metric-card";
import { CorrelationMatrix } from "@/components/correlation-matrix";
import { useAnalysis } from "@/lib/analysis-store";
import { useModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { useChartTheme } from "@/lib/chart-theme";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  LineChart,
  Line,
  Legend,
} from "recharts";

export const Route = createFileRoute("/new/visualize")({
  beforeLoad: requireAuth,
  component: VisualizePage,
});

function VisualizePage() {
  const { state } = useAnalysis();
  const navigate = useNavigate();
  const t = useChartTheme();
  const metricsQuery = useModelMetrics(state.modelId);

  const rows = state.dataset?.rows ?? [];
  const cols = state.dataset?.columns ?? [];
  const targetIdx = cols.indexOf(state.target ?? "");
  const firstFeatureIdx = cols.indexOf(state.features[0] ?? cols[0]);

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first to see results.</p>
            <Button asChild className="mt-4">
              <Link to="/new/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const scatterData = rows.map((r) => ({
    x: Number(r[firstFeatureIdx]) || 0,
    y: Number(r[targetIdx]) || 0,
  }));

  const histogram = buildHistogram(
    rows.map((r) => Number(r[targetIdx]) || 0),
    6,
  );

  const metrics = metricsQuery.data?.metrics ?? state.results;
  const chartData = metricsQuery.data?.chart_data;

  const metricCards = [
    { label: "R² Score", value: fmt(metrics.R2), gradient: "from-emerald-500 to-teal-500" },
    {
      label: "Adjusted R²",
      value: fmt(metrics["Adjusted R2"]),
      gradient: "from-purple-500 to-violet-500",
    },
    { label: "RMSE", value: fmt(metrics.RMSE), gradient: "from-indigo-500 to-blue-500" },
    { label: "MAE", value: fmt(metrics.MAE), gradient: "from-amber-500 to-orange-500" },
    { label: "MSE", value: fmt(metrics.MSE), gradient: "from-fuchsia-500 to-pink-500" },
  ];

  const axisProps = {
    stroke: t.mutedForeground,
    tick: t.axisTick,
  };

  const actualVsPredicted =
    chartData?.actual.map((a, i) => ({ i, actual: a, predicted: chartData.predicted[i] })) ?? [];
  const residualPoints = chartData?.residuals.map((r, i) => ({ i, residual: r })) ?? [];
  const corr = chartData?.correlation_matrix;

  return (
    <div>
      <WizardSteps />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5 mb-6">
        {metricCards.map((m) => (
          <MetricCard key={m.label} label={m.label} value={m.value} gradient={m.gradient} />
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title={`Distribution — ${state.target ?? "target"}`}>
          <BarChart data={histogram}>
            <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
            <XAxis dataKey="bin" {...axisProps} />
            <YAxis {...axisProps} />
            <Tooltip
              contentStyle={t.tooltipStyle}
              labelStyle={t.tooltipLabelStyle}
              itemStyle={t.tooltipItemStyle}
              cursor={{ fill: t.mutedForeground, opacity: 0.1 }}
            />
            <Bar dataKey="count" fill={t.series.primary} />
          </BarChart>
        </ChartCard>

        <ChartCard title={`Scatter — ${cols[firstFeatureIdx]} vs ${state.target ?? "target"}`}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
            <XAxis type="number" dataKey="x" name={cols[firstFeatureIdx]} {...axisProps} />
            <YAxis type="number" dataKey="y" name={state.target ?? ""} {...axisProps} />
            <Tooltip
              contentStyle={t.tooltipStyle}
              labelStyle={t.tooltipLabelStyle}
              itemStyle={t.tooltipItemStyle}
              cursor={{ strokeDasharray: "3 3", stroke: t.mutedForeground }}
            />
            <Scatter data={scatterData} fill={t.series.primary} />
          </ScatterChart>
        </ChartCard>

        {metricsQuery.isLoading ? (
          <>
            <Skeleton className="h-[308px] w-full rounded-xl" />
            <Skeleton className="h-[308px] w-full rounded-xl" />
          </>
        ) : (
          <>
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
          </>
        )}
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="text-base">Correlation Matrix</CardTitle>
        </CardHeader>
        <CardContent>
          {metricsQuery.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <CorrelationMatrix data={corr} />
          )}
        </CardContent>
      </Card>

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/new/explain" })}>Continue to AI Insights</Button>
      </div>
    </div>
  );
}

function fmt(v: number | undefined) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return Math.abs(v) >= 1000
    ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : v.toFixed(3);
}

function buildHistogram(values: number[], bins: number) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) / bins || 1;
  const arr = Array.from({ length: bins }, (_, i) => ({
    bin: `${Math.round(min + i * width)}`,
    count: 0,
  }));
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.floor((v - min) / width));
    arr[idx].count++;
  }
  return arr;
}
