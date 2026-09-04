import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ChartCard } from "@/components/chart-card";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { useClusteringModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { useChartTheme } from "@/lib/chart-theme";
import {
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

export const Route = createFileRoute("/cluster/visualize")({
  beforeLoad: requireAuth,
  component: VisualizePage,
});

function Unavailable({ message }: { message: string }) {
  return <p className="text-sm text-muted-foreground">{message}</p>;
}

// Distinct, theme-agnostic per-cluster palette — "Noise" always gets a
// dedicated muted red regardless of how many real clusters exist, so
// DBSCAN/OPTICS noise points read as visually distinct from every cluster.
const CLUSTER_COLORS = ["#6366f1", "#22d3ee", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#fb923c", "#2dd4bf"];
const NOISE_COLOR = "#ef4444";

function colorForLabel(label: string, index: number) {
  if (label === "Noise") return NOISE_COLOR;
  return CLUSTER_COLORS[index % CLUSTER_COLORS.length];
}

function Dendrogram({ icoord, dcoord }: { icoord: number[][]; dcoord: number[][] }) {
  const t = useChartTheme();
  const width = 600;
  const height = 260;
  const padding = 20;

  const allX = icoord.flat();
  const allY = dcoord.flat();
  const maxX = Math.max(...allX, 1);
  const maxY = Math.max(...allY, 1);

  const scaleX = (x: number) => padding + (x / maxX) * (width - padding * 2);
  const scaleY = (y: number) => height - padding - (y / maxY) * (height - padding * 2);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
      {icoord.map((xs, i) => {
        const ys = dcoord[i];
        const points = xs.map((x, j) => `${scaleX(x)},${scaleY(ys[j])}`).join(" ");
        return (
          <polyline
            key={i}
            points={points}
            fill="none"
            stroke={t.series.primary}
            strokeWidth={1.5}
            opacity={0.85}
          />
        );
      })}
    </svg>
  );
}

function VisualizePage() {
  const { state } = useClusteringAnalysis();
  const navigate = useNavigate();
  const t = useChartTheme();
  const metricsQuery = useClusteringModelMetrics(state.modelId);

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps steps={clusteringSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Run clustering first to see visualizations.</p>
            <Button asChild className="mt-4">
              <Link to="/cluster/train">Go to clustering</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (metricsQuery.isError) {
    return (
      <div>
        <WizardSteps steps={clusteringSteps} />
        <Card>
          <CardContent className="p-8 text-center text-sm text-muted-foreground">
            Unable to generate visualization.
          </CardContent>
        </Card>
      </div>
    );
  }

  const viz = metricsQuery.data?.visualizations;
  const axisProps = { stroke: t.mutedForeground, tick: t.axisTick };

  // PCA scatter — one <Scatter> series per cluster label so each gets a
  // distinct, legend-visible color (including "Noise", always red).
  const pca = viz?.pca;
  const pcaByLabel: Record<string, { x: number; y: number }[]> = {};
  const labelOrder: string[] = [];
  if (pca) {
    pca.x.forEach((x, i) => {
      const label = pca.labels[i];
      if (!pcaByLabel[label]) {
        pcaByLabel[label] = [];
        labelOrder.push(label);
      }
      pcaByLabel[label].push({ x, y: pca.y[i] });
    });
    // Sort so "Noise" renders last (drawn on top is fine — it's visually distinct anyway)
    labelOrder.sort((a, b) => (a === "Noise" ? 1 : b === "Noise" ? -1 : a.localeCompare(b, undefined, { numeric: true })));
  }

  // Silhouette by cluster — per-cluster AVERAGE of the backend's per-sample
  // silhouette values (a full one-bar-per-sample plot doesn't translate
  // cleanly to a bar chart at typical dataset sizes; the average still
  // shows which clusters are well-separated vs. poorly-separated).
  const silhouettePlot = viz?.silhouette_plot;
  const silhouetteByClusterData = silhouettePlot
    ? Object.entries(silhouettePlot.by_cluster).map(([cluster, values]) => ({
        cluster: `Cluster ${cluster}`,
        avg: values.reduce((a, b) => a + b, 0) / (values.length || 1),
      }))
    : [];

  const elbow = viz?.elbow;
  const elbowData = elbow ? elbow.k_values.map((k, i) => ({ k, value: elbow.values[i] })) : [];

  const dendrogram = viz?.dendrogram;

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />
      {metricsQuery.isLoading && (
        <p className="text-sm text-muted-foreground mb-3">Preparing visualizations…</p>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* PCA 2D Cluster Plot */}
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : pca && labelOrder.length > 0 ? (
          <ChartCard title="PCA 2D Cluster Plot">
            <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
              <XAxis type="number" dataKey="x" name="PC1" {...axisProps} />
              <YAxis type="number" dataKey="y" name="PC2" {...axisProps} />
              <Tooltip
                contentStyle={t.tooltipStyle}
                labelStyle={t.tooltipLabelStyle}
                itemStyle={t.tooltipItemStyle}
                cursor={{ strokeDasharray: "3 3", stroke: t.mutedForeground }}
              />
              <Legend wrapperStyle={t.legendStyle} />
              {labelOrder.map((label, i) => (
                <Scatter
                  key={label}
                  name={label === "Noise" ? "Noise" : `Cluster ${label}`}
                  data={pcaByLabel[label]}
                  fill={colorForLabel(label, i)}
                />
              ))}
            </ScatterChart>
          </ChartCard>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">PCA 2D Cluster Plot</CardTitle>
            </CardHeader>
            <CardContent>
              <Unavailable message="PCA plot not available — at least 2 usable feature dimensions are required." />
            </CardContent>
          </Card>
        )}

        {/* Silhouette Plot (by cluster average) */}
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : silhouetteByClusterData.length > 0 ? (
          <ChartCard title="Silhouette Score by Cluster">
            <BarChart data={silhouetteByClusterData}>
              <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
              <XAxis dataKey="cluster" {...axisProps} />
              <YAxis domain={[-1, 1]} {...axisProps} />
              <Tooltip
                contentStyle={t.tooltipStyle}
                labelStyle={t.tooltipLabelStyle}
                itemStyle={t.tooltipItemStyle}
                formatter={(v: number) => v.toFixed(3)}
                cursor={{ fill: t.mutedForeground, opacity: 0.1 }}
              />
              <Bar dataKey="avg" fill={t.series.success} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ChartCard>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Silhouette Plot</CardTitle>
            </CardHeader>
            <CardContent>
              <Unavailable message="Silhouette plot not available for this configuration (requires at least 2 real clusters)." />
            </CardContent>
          </Card>
        )}

        {/* Elbow Plot */}
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : elbowData.length > 0 ? (
          <ChartCard title={`Elbow Plot (${elbow!.metric_label})`}>
            <LineChart data={elbowData}>
              <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
              <XAxis dataKey="k" label={{ value: "k (clusters)", position: "insideBottom", offset: -2, fill: t.mutedForeground, fontSize: 11 }} {...axisProps} />
              <YAxis {...axisProps} />
              <Tooltip
                contentStyle={t.tooltipStyle}
                labelStyle={t.tooltipLabelStyle}
                itemStyle={t.tooltipItemStyle}
              />
              <Line type="monotone" dataKey="value" stroke={t.series.primary} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ChartCard>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Elbow Plot</CardTitle>
            </CardHeader>
            <CardContent>
              <Unavailable message="Elbow plot is only applicable to centroid-based algorithms (K-Means, K-Medoids, GMM)." />
            </CardContent>
          </Card>
        )}

        {/* Dendrogram */}
        {metricsQuery.isLoading ? (
          <Skeleton className="h-[308px] w-full rounded-xl" />
        ) : dendrogram ? (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Dendrogram</CardTitle>
              {dendrogram.sampled && (
                <p className="text-xs text-muted-foreground">
                  Sampled {dendrogram.sample_size.toLocaleString()} of {dendrogram.total_size.toLocaleString()} rows for rendering.
                </p>
              )}
            </CardHeader>
            <CardContent>
              <Dendrogram icoord={dendrogram.icoord} dcoord={dendrogram.dcoord} />
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Dendrogram</CardTitle>
            </CardHeader>
            <CardContent>
              <Unavailable message="Dendrogram is only applicable to Agglomerative (hierarchical) Clustering." />
            </CardContent>
          </Card>
        )}
      </div>

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/cluster/explain" })}>Continue to AI Insights</Button>
      </div>
    </div>
  );
}
