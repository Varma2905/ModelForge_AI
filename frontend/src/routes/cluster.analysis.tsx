import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/metric-card";
import { ChartCard } from "@/components/chart-card";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { useClusteringModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { useChartTheme } from "@/lib/chart-theme";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

export const Route = createFileRoute("/cluster/analysis")({
  beforeLoad: requireAuth,
  component: AnalysisPage,
});

function fmt(v: number | undefined | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}

function AnalysisPage() {
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
            <p>Run clustering first to see the analysis.</p>
            <Button asChild className="mt-4">
              <Link to="/cluster/train">Go to clustering</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const data = metricsQuery.data;
  const metrics = data?.metrics;
  const hasNoise = metrics?.NoisePoints !== null && metrics?.NoisePoints !== undefined;

  const clusterSizes = data?.cluster_sizes ?? {};
  const distributionData = Object.entries(clusterSizes).map(([cluster, count]) => ({
    cluster: cluster === "-1" ? "Noise" : `Cluster ${cluster}`,
    count,
  }));

  const axisProps = { stroke: t.mutedForeground, tick: t.axisTick };

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />

      <div className="mb-6">
        <h2 className="text-lg font-semibold">Clustering Analysis</h2>
        <p className="text-sm text-muted-foreground">
          {data?.model ?? "Model"} found {metrics?.ClusterCount ?? "—"} clusters across{" "}
          {data?.total_rows?.toLocaleString() ?? "—"} samples.
        </p>
      </div>

      {metricsQuery.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <>
          {/* Cluster Quality */}
          <div className={`grid gap-4 sm:grid-cols-2 ${hasNoise ? "lg:grid-cols-5" : "lg:grid-cols-4"} mb-6`}>
            <MetricCard label="Silhouette Score" value={fmt(metrics?.Silhouette)} gradient="from-emerald-500 to-teal-500" />
            <MetricCard label="Calinski-Harabasz" value={metrics?.CalinskiHarabasz != null ? metrics.CalinskiHarabasz.toFixed(1) : "—"} gradient="from-purple-500 to-violet-500" />
            <MetricCard label="Davies-Bouldin" value={fmt(metrics?.DaviesBouldin)} gradient="from-indigo-500 to-blue-500" />
            <MetricCard label="Cluster Count" value={metrics?.ClusterCount ?? "—"} gradient="from-amber-500 to-orange-500" />
            {hasNoise && (
              <MetricCard label="Noise Points" value={metrics?.NoisePoints ?? "—"} gradient="from-fuchsia-500 to-pink-500" />
            )}
          </div>

          {/* Cluster Distribution */}
          <div className="mb-6">
            {distributionData.length > 0 ? (
              <ChartCard title="Cluster Distribution">
                <BarChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
                  <XAxis dataKey="cluster" {...axisProps} />
                  <YAxis allowDecimals={false} {...axisProps} />
                  <Tooltip
                    contentStyle={t.tooltipStyle}
                    labelStyle={t.tooltipLabelStyle}
                    itemStyle={t.tooltipItemStyle}
                    cursor={{ fill: t.mutedForeground, opacity: 0.1 }}
                  />
                  <Bar dataKey="count" fill={t.series.primary} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartCard>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Cluster Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">No cluster distribution available.</p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Cluster Profiles */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Cluster Profiles</CardTitle>
              <p className="text-xs text-muted-foreground">
                Feature means and the characteristics that most distinguish each cluster from the dataset average.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              {(data?.cluster_profiles ?? []).length === 0 ? (
                <p className="text-sm text-muted-foreground">No cluster profiles available.</p>
              ) : (
                data!.cluster_profiles.map((profile) => (
                  <div key={profile.cluster} className="rounded-lg border p-4 space-y-3">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <h4 className="font-semibold">
                        {profile.cluster === "Noise" ? "Noise" : `Cluster ${profile.cluster}`}
                      </h4>
                      <Badge variant="secondary">{profile.samples.toLocaleString()} samples</Badge>
                    </div>

                    {Object.keys(profile.feature_means).length > 0 && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b text-left text-xs text-muted-foreground">
                              <th className="py-1.5 pr-4 font-medium">Feature</th>
                              <th className="py-1.5 pr-4 font-medium">Mean</th>
                              <th className="py-1.5 font-medium">Median</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(profile.feature_means).map(([feat, mean]) => (
                              <tr key={feat} className="border-b last:border-0">
                                <td className="py-1.5 pr-4 font-medium">{feat}</td>
                                <td className="py-1.5 pr-4 tabular-nums">{fmt(mean)}</td>
                                <td className="py-1.5 tabular-nums">{fmt(profile.feature_medians[feat])}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {Object.keys(profile.categorical_modes).length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(profile.categorical_modes).map(([feat, mode]) => (
                          <Badge key={feat} variant="outline" className="text-xs font-normal">
                            {feat}: {mode ?? "—"}
                          </Badge>
                        ))}
                      </div>
                    )}

                    {profile.important_characteristics.length > 0 && (
                      <div>
                        <p className="text-xs font-medium text-muted-foreground mb-1">Important Characteristics</p>
                        <ul className="text-sm space-y-0.5">
                          {profile.important_characteristics.map((c, i) => (
                            <li key={i}>• {c}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </>
      )}

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/cluster/visualize" })}>Continue to Visualizations</Button>
      </div>
    </div>
  );
}
