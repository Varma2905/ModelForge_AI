import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { WizardSteps, classificationSteps } from "@/components/wizard-steps";
import { ChartCard } from "@/components/chart-card";
import { MetricCard } from "@/components/metric-card";
import { CorrelationMatrix } from "@/components/correlation-matrix";
import { ConfusionMatrix } from "@/components/confusion-matrix";
import { useClassificationAnalysis } from "@/lib/classification-analysis-store";
import { useClassificationModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { useChartTheme } from "@/lib/chart-theme";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

export const Route = createFileRoute("/classify/visualize")({
  beforeLoad: requireAuth,
  component: VisualizePage,
});

function fmtPct(v: number | undefined | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

// Small "this plot isn't available for this model/dataset" placeholder —
// used identically across every visualization below so a missing field
// (unsupported model, non-binary target, no predict_proba, ...) always
// renders the same calm message instead of an empty chart or a crash.
function Unavailable({ message }: { message: string }) {
  return <p className="text-sm text-muted-foreground">{message}</p>;
}

function VisualizePage() {
  const { state } = useClassificationAnalysis();
  const navigate = useNavigate();
  const t = useChartTheme();
  const metricsQuery = useClassificationModelMetrics(state.modelId);

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first to see results.</p>
            <Button asChild className="mt-4">
              <Link to="/classify/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (metricsQuery.isError) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
        <Card>
          <CardContent className="p-8 text-center text-sm text-muted-foreground">
            Unable to generate visualization.
          </CardContent>
        </Card>
      </div>
    );
  }

  const chartData = metricsQuery.data?.chart_data;
  // The dedicated visualization payload (see backend classify_routes.py) —
  // every field on it is independently nullable, so each card below guards
  // itself rather than assuming the whole object (or any one field) exists.
  const viz = metricsQuery.data?.visualizations;

  const classDistribution = viz?.class_distribution;
  const classDistributionData =
    classDistribution && classDistribution.labels.length > 0
      ? classDistribution.labels.map((label, i) => ({ label, count: classDistribution.values[i] ?? 0 }))
      : [];

  const nativeImportance = viz?.feature_importance;
  const nativeImportanceData =
    nativeImportance && nativeImportance.features.length > 0
      ? nativeImportance.features.map((feature, i) => ({ feature, value: nativeImportance.importance[i] ?? 0 }))
      : [];

  const roc = viz?.roc_curve;
  const rocData = roc && roc.fpr.length > 0 ? roc.fpr.map((fpr, i) => ({ fpr, tpr: roc.tpr[i] ?? 0 })) : [];

  const prCurve = viz?.precision_recall_curve;
  const prCurveData =
    prCurve && prCurve.recall.length > 0
      ? prCurve.recall.map((recall, i) => ({ recall, precision: prCurve.precision[i] ?? 0 }))
      : [];

  const metricCards = [
    { label: "Accuracy", value: fmtPct(metricsQuery.data?.metrics.Accuracy), gradient: "from-emerald-500 to-teal-500" },
    { label: "Precision", value: fmtPct(metricsQuery.data?.metrics.Precision), gradient: "from-purple-500 to-violet-500" },
    { label: "Recall", value: fmtPct(metricsQuery.data?.metrics.Recall), gradient: "from-indigo-500 to-blue-500" },
    { label: "F1-Score", value: metricsQuery.data?.metrics.F1?.toFixed(3) ?? "—", gradient: "from-amber-500 to-orange-500" },
    { label: "ROC-AUC", value: metricsQuery.data?.metrics.ROC_AUC?.toFixed(3) ?? "—", gradient: "from-fuchsia-500 to-pink-500" },
  ];

  const axisProps = { stroke: t.mutedForeground, tick: t.axisTick };

  return (
    <div>
      <WizardSteps steps={classificationSteps} />

      <div className="mb-6">
        <h2 className="text-lg font-semibold">Classification Visualizations</h2>
        <p className="text-sm text-muted-foreground">
          {metricsQuery.isLoading ? "Preparing visualizations…" : `Model: ${metricsQuery.data?.model ?? "—"}`}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5 mb-6">
        {metricCards.map((m) => (
          <MetricCard key={m.label} label={m.label} value={m.value} gradient={m.gradient} />
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* 1. Class Distribution */}
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : classDistributionData.length > 0 ? (
          <ChartCard title="Class Distribution">
            <BarChart data={classDistributionData}>
              <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
              <XAxis dataKey="label" {...axisProps} />
              <YAxis allowDecimals={false} {...axisProps} />
              <Tooltip
                contentStyle={t.tooltipStyle}
                labelStyle={t.tooltipLabelStyle}
                itemStyle={t.tooltipItemStyle}
                cursor={{ fill: t.mutedForeground, opacity: 0.1 }}
              />
              <Bar dataKey="count" fill={t.series.accent} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ChartCard>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Class Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <Unavailable message="Class distribution is not available for this model." />
            </CardContent>
          </Card>
        )}

        {/* 2. Confusion Matrix */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Confusion Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            {metricsQuery.isLoading ? (
              <Skeleton className="h-48 w-full" />
            ) : (
              <ConfusionMatrix data={chartData?.confusion_matrix} />
            )}
          </CardContent>
        </Card>

        {/* 3. Feature Importance (native to the trained model — see backend
            extract_feature_importance) */}
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : nativeImportanceData.length > 0 ? (
          <ChartCard title="Feature Importance">
            <BarChart data={nativeImportanceData} layout="vertical">
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
              <Unavailable message="Feature importance is not available for this model." />
            </CardContent>
          </Card>
        )}

        {/* 4. ROC Curve (binary only) */}
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : rocData.length > 0 ? (
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">ROC Curve</CardTitle>
              {typeof roc?.auc === "number" && <Badge variant="secondary">AUC {roc.auc.toFixed(3)}</Badge>}
            </CardHeader>
            <CardContent>
              <div style={{ width: "100%", height: 260 }}>
                <ResponsiveContainer>
                  <LineChart data={rocData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
                    <XAxis
                      dataKey="fpr"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v: number) => v.toFixed(1)}
                      label={{ value: "False Positive Rate", position: "insideBottom", offset: -2, fill: t.mutedForeground, fontSize: 11 }}
                      {...axisProps}
                    />
                    <YAxis
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v: number) => v.toFixed(1)}
                      label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: t.mutedForeground, fontSize: 11 }}
                      {...axisProps}
                    />
                    <Tooltip
                      contentStyle={t.tooltipStyle}
                      labelStyle={t.tooltipLabelStyle}
                      itemStyle={t.tooltipItemStyle}
                      formatter={(value: number) => value.toFixed(3)}
                      labelFormatter={(v: number) => `FPR ${v.toFixed(3)}`}
                    />
                    <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={t.mutedForeground} strokeDasharray="4 4" />
                    <Line type="monotone" dataKey="tpr" stroke={t.series.primary} dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">ROC Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <Unavailable message="ROC curve is not available for this model." />
            </CardContent>
          </Card>
        )}
      </div>

      {/* 5. Precision-Recall Curve (binary only) */}
      <div className="mt-6">
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : prCurveData.length > 0 ? (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Precision-Recall Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <div style={{ width: "100%", height: 260 }}>
                <ResponsiveContainer>
                  <LineChart data={prCurveData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
                    <XAxis
                      dataKey="recall"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v: number) => v.toFixed(1)}
                      label={{ value: "Recall", position: "insideBottom", offset: -2, fill: t.mutedForeground, fontSize: 11 }}
                      {...axisProps}
                    />
                    <YAxis
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v: number) => v.toFixed(1)}
                      label={{ value: "Precision", angle: -90, position: "insideLeft", fill: t.mutedForeground, fontSize: 11 }}
                      {...axisProps}
                    />
                    <Tooltip
                      contentStyle={t.tooltipStyle}
                      labelStyle={t.tooltipLabelStyle}
                      itemStyle={t.tooltipItemStyle}
                      formatter={(value: number) => value.toFixed(3)}
                      labelFormatter={(v: number) => `Recall ${v.toFixed(3)}`}
                    />
                    <Line type="monotone" dataKey="precision" stroke={t.series.success} dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Precision-Recall Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <Unavailable message="Precision-Recall curve is not available for this model." />
            </CardContent>
          </Card>
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
            <CorrelationMatrix data={chartData?.correlation_matrix} />
          )}
        </CardContent>
      </Card>

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/classify/explain" })}>Continue to AI Insights</Button>
      </div>
    </div>
  );
}
