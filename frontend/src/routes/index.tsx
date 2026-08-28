import { createFileRoute, Link } from "@tanstack/react-router";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollReveal } from "@/components/scroll-reveal";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Database,
  Brain,
  FileText,
  TrendingUp,
  ArrowRight,
  Sparkles,
  PlusCircle,
  Upload,
  MessageCircleQuestion,
  FolderOpen,
  LineChart as LineChartIcon,
  Shapes,
  Target,
  Layers,
  Lightbulb,
  Plus,
  Network,
} from "lucide-react";
import { useDashboardSummary } from "@/hooks/use-dashboard";
import { useChartTheme } from "@/lib/chart-theme";
import { requireAuth } from "@/lib/require-auth";
import { useAuth } from "@/lib/auth-context";
import { useNewAnalysisModal } from "@/components/new-analysis-modal";

export const Route = createFileRoute("/")({
  beforeLoad: requireAuth,
  component: Dashboard,
});

function PerformanceTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { value: number; name: string; color: string }[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-popover/90 backdrop-blur-md px-3 py-2 shadow-lg">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="text-sm font-semibold flex items-center gap-2" style={{ color: p.color }}>
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span>
            {p.name}: {p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

function MiniSparkline({
  data,
  color,
  gradientId,
}: {
  data: number[];
  color: string;
  gradientId: string;
}) {
  const chartData = data.map((val, index) => ({ index, val }));
  return (
    <div className="h-8 w-full mt-2 -mb-2 overflow-hidden opacity-85">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.25} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="val"
            stroke={color}
            strokeWidth={1.5}
            fill={`url(#${gradientId})`}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function StatCard({
  label,
  value,
  subtext,
  icon: Icon,
  gradient,
  sparklineData,
  sparklineColor,
  sparklineGradId,
  trend,
  trendColorClass = "text-emerald-500",
}: {
  label: string;
  value: string | number;
  subtext?: string;
  icon: any;
  gradient: string;
  sparklineData: number[];
  sparklineColor: string;
  sparklineGradId: string;
  trend?: string | null;
  trendColorClass?: string;
}) {
  return (
    <Card className="relative overflow-hidden border-border/60 bg-card/45 backdrop-blur-md transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:border-primary/20">
      <CardContent className="p-4 flex flex-col justify-between h-full min-h-[120px]">
        <div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              {label}
            </span>
            <div
              className={`h-7 w-7 rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center text-white shadow-sm flex-shrink-0`}
            >
              <Icon className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-1">
            <div className="text-2xl font-bold tracking-tight">{value}</div>
            {subtext && <div className="text-[11px] text-muted-foreground mt-0.5 truncate">{subtext}</div>}
            {trend && <div className={`text-[10px] font-medium mt-0.5 ${trendColorClass}`}>{trend}</div>}
          </div>
        </div>
        {sparklineData && sparklineData.length > 0 && (
          <MiniSparkline data={sparklineData} color={sparklineColor} gradientId={sparklineGradId} />
        )}
      </CardContent>
    </Card>
  );
}

function Dashboard() {
  const { data, isLoading, isError } = useDashboardSummary();
  const { user } = useAuth();
  const chartTheme = useChartTheme();
  const { setOpen: setNewAnalysisOpen } = useNewAnalysisModal();

  const totalModels = data?.total_analyses ?? 0;
  const classificationCount = data?.total_classification_analyses ?? 0;
  const clusteringCount = data?.total_clustering_analyses ?? 0;
  const regressionCount = totalModels - classificationCount - clusteringCount;

  const distributionData = [
    { name: "Regression", value: regressionCount, color: "#6366f1" },
    { name: "Classification", value: classificationCount, color: "#14b8a6" },
    { name: "Clustering", value: clusteringCount, color: "#f59e0b" },
  ];

  const recent = data?.recent_analyses ?? [];

  return (
    <div className="space-y-6 pb-20 relative">
      {/* Header / Greeting */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
            Welcome back, {user?.name || "User"}! 👋
          </h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            Here's what's happening with your machine learning workspace.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button
            onClick={() => setNewAnalysisOpen(true)}
            className="bg-violet-600 hover:bg-violet-700 text-white shadow-md shadow-violet-600/10 flex items-center gap-1.5 font-medium"
          >
            <Plus className="h-4 w-4" /> New Analysis
          </Button>
        </div>
      </div>

      {/* KPI 6-card row */}
      {isLoading ? (
        <div className="grid gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="h-28 border-border bg-card/50 backdrop-blur-sm animate-pulse" />
          ))}
        </div>
      ) : isError ? (
        <Card>
          <CardContent className="p-6 text-sm text-destructive">
            Couldn't load dashboard stats. Please reload.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
          <ScrollReveal delay={0}>
            <StatCard
              label="Total Datasets"
              value={data.total_datasets}
              icon={Database}
              gradient="from-violet-500 to-purple-600"
              sparklineData={data.datasets_sparkline}
              sparklineColor="#a855f7"
              sparklineGradId="dsGrad"
              trend={
                data.trend_datasets_pct !== null
                  ? `${data.trend_datasets_pct >= 0 ? "↑" : "↓"} ${Math.abs(data.trend_datasets_pct)}% vs last month`
                  : null
              }
              trendColorClass={data.trend_datasets_pct && data.trend_datasets_pct >= 0 ? "text-emerald-500" : "text-rose-500"}
            />
          </ScrollReveal>

          <ScrollReveal delay={40}>
            <StatCard
              label="Regression Analyses"
              value={regressionCount}
              icon={Brain}
              gradient="from-indigo-500 to-blue-600"
              sparklineData={data.regression_sparkline}
              sparklineColor="#4f46e5"
              sparklineGradId="regGrad"
              trend={
                data.trend_analyses_pct !== null
                  ? `${data.trend_analyses_pct >= 0 ? "↑" : "↓"} ${Math.abs(data.trend_analyses_pct)}% vs last month`
                  : null
              }
              trendColorClass={data.trend_analyses_pct && data.trend_analyses_pct >= 0 ? "text-emerald-500" : "text-rose-500"}
            />
          </ScrollReveal>

          <ScrollReveal delay={80}>
            <StatCard
              label="Classification Analyses"
              value={classificationCount}
              icon={Shapes}
              gradient="from-teal-500 to-emerald-600"
              sparklineData={data.classification_sparkline}
              sparklineColor="#0d9488"
              sparklineGradId="clfGrad"
              trend={
                data.avg_accuracy_delta !== null
                  ? `${data.avg_accuracy_delta >= 0 ? "↑" : "↓"} ${(Math.abs(data.avg_accuracy_delta) * 100).toFixed(0)}% accuracy delta`
                  : null
              }
              trendColorClass={data.avg_accuracy_delta && data.avg_accuracy_delta >= 0 ? "text-emerald-500" : "text-rose-500"}
            />
          </ScrollReveal>

          <ScrollReveal delay={120}>
            <StatCard
              label="Reports Generated"
              value={data.total_reports}
              icon={FileText}
              gradient="from-amber-500 to-orange-600"
              sparklineData={data.reports_sparkline}
              sparklineColor="#f59e0b"
              sparklineGradId="repGrad"
              trend={
                data.trend_reports_pct !== null
                  ? `${data.trend_reports_pct >= 0 ? "↑" : "↓"} ${Math.abs(data.trend_reports_pct)}% vs last month`
                  : null
              }
              trendColorClass={data.trend_reports_pct && data.trend_reports_pct >= 0 ? "text-emerald-500" : "text-rose-500"}
            />
          </ScrollReveal>

          <ScrollReveal delay={160}>
            <StatCard
              label="Best R² Score"
              value={data.best_model ? data.best_model.r2.toFixed(3) : "—"}
              subtext={data.best_model ? data.best_model.model : "Linear Regression"}
              icon={TrendingUp}
              gradient="from-fuchsia-500 to-purple-600"
              sparklineData={data.r2_sparkline}
              sparklineColor="#d946ef"
              sparklineGradId="r2Grad"
            />
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <StatCard
              label="Best Accuracy"
              value={data.best_accuracy_model ? `${(data.best_accuracy_model.accuracy * 100).toFixed(1)}%` : "—"}
              subtext={data.best_accuracy_model ? data.best_accuracy_model.model : "Random Forest"}
              icon={Target}
              gradient="from-pink-500 to-rose-600"
              sparklineData={data.accuracy_sparkline}
              sparklineColor="#ec4899"
              sparklineGradId="accGrad"
            />
          </ScrollReveal>
        </div>
      )}

      {/* Middle row: Trend Chart + Donut Distribution + Recent list */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Trend line chart */}
        <ScrollReveal>
          <Card className="border-border/60 bg-card/45 backdrop-blur-md">
            <CardHeader className="pb-2">
              <CardTitle className="text-base font-semibold">Analysis Trend</CardTitle>
              <CardDescription className="text-xs">Regression vs Classification vs Clustering over time</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="h-[200px] w-full bg-muted/20 animate-pulse rounded-lg" />
              ) : (
                <div style={{ width: "100%", height: 200 }}>
                  <ResponsiveContainer>
                    <LineChart data={data?.trend_data} margin={{ top: 10, right: 5, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 9, fill: chartTheme.axisTick.fill }}
                        axisLine={{ stroke: chartTheme.border }}
                        tickLine={false}
                        interval={6}
                      />
                      <YAxis
                        tick={{ fontSize: 9, fill: chartTheme.axisTick.fill }}
                        axisLine={{ stroke: chartTheme.border }}
                        tickLine={false}
                        allowDecimals={false}
                      />
                      <Tooltip content={<PerformanceTooltip />} />
                      <Line
                        type="monotone"
                        dataKey="regression"
                        name="Regression"
                        stroke="#6366f1"
                        strokeWidth={2}
                        dot={{ r: 2 }}
                        activeDot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="classification"
                        name="Classification"
                        stroke="#14b8a6"
                        strokeWidth={2}
                        dot={{ r: 2 }}
                        activeDot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="clustering"
                        name="Clustering"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        dot={{ r: 2 }}
                        activeDot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </ScrollReveal>

        {/* Donut distribution */}
        <ScrollReveal delay={80}>
          <Card className="border-border/60 bg-card/45 backdrop-blur-md">
            <CardHeader className="pb-2">
              <CardTitle className="text-base font-semibold">Analysis Distribution</CardTitle>
              <CardDescription className="text-xs">Proportion of trained models by type</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="h-[200px] w-full bg-muted/20 animate-pulse rounded-lg" />
              ) : (
                <div className="flex flex-col justify-between">
                  <div className="relative flex items-center justify-center" style={{ height: 160 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={distributionData}
                          cx="50%"
                          cy="50%"
                          innerRadius={55}
                          outerRadius={75}
                          paddingAngle={3}
                          dataKey="value"
                        >
                          {distributionData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute flex flex-col items-center justify-center text-center">
                      <span className="text-2xl font-extrabold tracking-tight">{totalModels}</span>
                      <span className="text-[9px] uppercase tracking-wider text-muted-foreground font-semibold">
                        Total
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-center gap-5 mt-2 text-xs">
                    <div className="flex items-center gap-1.5">
                      <div className="h-2 w-2 rounded-full bg-[#6366f1]" />
                      <span className="text-muted-foreground text-[11px]">Regression</span>
                      <span className="font-semibold text-[11px] text-foreground">
                        {totalModels > 0 ? ((regressionCount / totalModels) * 100).toFixed(0) : 0}% ({regressionCount})
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="h-2 w-2 rounded-full bg-[#14b8a6]" />
                      <span className="text-muted-foreground text-[11px]">Classification</span>
                      <span className="font-semibold text-[11px] text-foreground">
                        {totalModels > 0 ? ((classificationCount / totalModels) * 100).toFixed(0) : 0}% ({classificationCount})
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="h-2 w-2 rounded-full bg-[#f59e0b]" />
                      <span className="text-muted-foreground text-[11px]">Clustering</span>
                      <span className="font-semibold text-[11px] text-foreground">
                        {totalModels > 0 ? ((clusteringCount / totalModels) * 100).toFixed(0) : 0}% ({clusteringCount})
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </ScrollReveal>

        {/* Recent analyses list */}
        <ScrollReveal delay={160}>
          <Card className="border-border/60 bg-card/45 backdrop-blur-md">
            <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
              <div>
                <CardTitle className="text-base font-semibold">Recent Analyses</CardTitle>
                <CardDescription className="text-xs">Your latest model runs</CardDescription>
              </div>
              <Button asChild variant="link" size="sm" className="h-auto p-0 text-violet-500 hover:text-violet-600 text-xs">
                <Link to="/compare">View all</Link>
              </Button>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="h-10 bg-muted/20 animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : recent.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-6 text-center text-muted-foreground">
                  <Brain className="h-8 w-8 opacity-40 mb-2" />
                  <p className="text-xs">No analyses completed yet</p>
                </div>
              ) : (
                <div className="space-y-2.5 max-h-[200px] overflow-y-auto pr-1">
                  {recent.map((a) => (
                    <Link
                      key={a.model_id}
                      to="/models/$modelId"
                      params={{ modelId: a.model_id }}
                      className="flex items-center gap-3 rounded-lg p-2 hover:bg-muted/40 transition-colors"
                    >
                      <div className="h-7 w-7 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                        {a.model_type === "classification" ? (
                          <Shapes className="h-3.5 w-3.5 text-teal-500" />
                        ) : a.model_type === "clustering" ? (
                          <Network className="h-3.5 w-3.5 text-amber-500" />
                        ) : (
                          <LineChartIcon className="h-3.5 w-3.5 text-violet-500" />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-semibold truncate text-foreground">
                          {a.dataset_name}
                        </div>
                        <div className="text-[10px] text-muted-foreground truncate">
                          {a.model} · {formatDistanceToNow(new Date(a.created_at), { addSuffix: true })}
                        </div>
                      </div>
                      <div className="flex-shrink-0">
                        {a.model_type === "classification" && a.accuracy !== null ? (
                          <Badge variant="outline" className="text-[10px] border-teal-500/30 text-teal-500">
                            Acc {(a.accuracy * 100).toFixed(1)}%
                          </Badge>
                        ) : a.model_type === "clustering" && a.silhouette !== null && a.silhouette !== undefined ? (
                          <Badge variant="outline" className="text-[10px] border-amber-500/30 text-amber-500">
                            Silhouette {a.silhouette.toFixed(3)}
                          </Badge>
                        ) : (
                          a.r2 !== null && (
                            <Badge variant="secondary" className="text-[10px] bg-violet-500/10 text-violet-500 border-none">
                              R² {a.r2.toFixed(3)}
                            </Badge>
                          )
                        )}
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </ScrollReveal>
      </div>

      {/* Grid: Top Regression Table + Top Classification Table + Quick Actions */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Regression table */}
        <ScrollReveal className="lg:col-span-1">
          <Card className="border-border/60 bg-card/45 backdrop-blur-md h-full">
            <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle className="text-sm font-semibold">Top Regression Models</CardTitle>
                <CardDescription className="text-xs">Highest R² score models</CardDescription>
              </div>
              <Button asChild variant="link" size="sm" className="h-auto p-0 text-violet-500 text-xs">
                <Link to="/compare">View all</Link>
              </Button>
            </CardHeader>
            <CardContent className="px-3">
              {isLoading ? (
                <div className="h-[200px] w-full bg-muted/20 animate-pulse rounded-lg" />
              ) : !data.top_regression_models || data.top_regression_models.length === 0 ? (
                <div className="text-center text-xs text-muted-foreground py-8">
                  No regression models trained yet
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table className="min-w-[280px]">
                    <TableHeader>
                      <TableRow className="hover:bg-transparent border-border/40">
                        <TableHead className="text-[10px] py-2 px-1">Model</TableHead>
                        <TableHead className="text-[10px] py-2 px-1">Dataset</TableHead>
                        <TableHead className="text-[10px] py-2 px-1 text-right">R² Score</TableHead>
                        <TableHead className="text-[10px] py-2 px-1 text-right">RMSE</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.top_regression_models.map((m, idx) => (
                        <TableRow key={m.model_id} className="hover:bg-muted/20 border-border/20">
                          <TableCell className="text-[11px] font-medium py-2 px-1 truncate max-w-[90px]">
                            <Link to="/models/$modelId" params={{ modelId: m.model_id }} className="hover:underline">
                              {idx + 1}. {m.model}
                            </Link>
                          </TableCell>
                          <TableCell className="text-[11px] text-muted-foreground py-2 px-1 truncate max-w-[80px]">
                            {m.dataset_name}
                          </TableCell>
                          <TableCell className="text-[11px] text-right font-semibold py-2 px-1">
                            {m.r2 !== null ? m.r2.toFixed(3) : "—"}
                          </TableCell>
                          <TableCell className="text-[11px] text-right text-muted-foreground py-2 px-1">
                            {m.rmse !== null ? m.rmse.toFixed(2) : "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </ScrollReveal>

        {/* Classification table */}
        <ScrollReveal delay={80} className="lg:col-span-1">
          <Card className="border-border/60 bg-card/45 backdrop-blur-md h-full">
            <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle className="text-sm font-semibold">Top Classification Models</CardTitle>
                <CardDescription className="text-xs">Highest accuracy models</CardDescription>
              </div>
              <Button asChild variant="link" size="sm" className="h-auto p-0 text-teal-500 text-xs">
                <Link to="/compare">View all</Link>
              </Button>
            </CardHeader>
            <CardContent className="px-3">
              {isLoading ? (
                <div className="h-[200px] w-full bg-muted/20 animate-pulse rounded-lg" />
              ) : !data.top_classification_models || data.top_classification_models.length === 0 ? (
                <div className="text-center text-xs text-muted-foreground py-8">
                  No classification models trained yet
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table className="min-w-[280px]">
                    <TableHeader>
                      <TableRow className="hover:bg-transparent border-border/40">
                        <TableHead className="text-[10px] py-2 px-1">Model</TableHead>
                        <TableHead className="text-[10px] py-2 px-1">Dataset</TableHead>
                        <TableHead className="text-[10px] py-2 px-1 text-right">Accuracy</TableHead>
                        <TableHead className="text-[10px] py-2 px-1 text-right">F1 Score</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.top_classification_models.map((m, idx) => (
                        <TableRow key={m.model_id} className="hover:bg-muted/20 border-border/20">
                          <TableCell className="text-[11px] font-medium py-2 px-1 truncate max-w-[90px]">
                            <Link to="/models/$modelId" params={{ modelId: m.model_id }} className="hover:underline">
                              {idx + 1}. {m.model}
                            </Link>
                          </TableCell>
                          <TableCell className="text-[11px] text-muted-foreground py-2 px-1 truncate max-w-[80px]">
                            {m.dataset_name}
                          </TableCell>
                          <TableCell className="text-[11px] text-right font-semibold py-2 px-1 text-teal-500">
                            {m.accuracy !== null ? `${(m.accuracy * 100).toFixed(1)}%` : "—"}
                          </TableCell>
                          <TableCell className="text-[11px] text-right text-muted-foreground py-2 px-1">
                            {m.f1_score !== null ? m.f1_score.toFixed(3) : "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </ScrollReveal>

        {/* Clustering table */}
        <ScrollReveal delay={160} className="lg:col-span-1">
          <Card className="border-border/60 bg-card/45 backdrop-blur-md h-full">
            <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle className="text-sm font-semibold">Top Clustering Models</CardTitle>
                <CardDescription className="text-xs">Highest silhouette score models</CardDescription>
              </div>
              <Button asChild variant="link" size="sm" className="h-auto p-0 text-amber-500 text-xs">
                <Link to="/compare">View all</Link>
              </Button>
            </CardHeader>
            <CardContent className="px-3">
              {isLoading ? (
                <div className="h-[200px] w-full bg-muted/20 animate-pulse rounded-lg" />
              ) : !data.top_clustering_models || data.top_clustering_models.length === 0 ? (
                <div className="text-center text-xs text-muted-foreground py-8">
                  No clustering models yet.{" "}
                  <Link to="/cluster/upload" className="text-amber-500 hover:underline">
                    Start your first clustering analysis →
                  </Link>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table className="min-w-[280px]">
                    <TableHeader>
                      <TableRow className="hover:bg-transparent border-border/40">
                        <TableHead className="text-[10px] py-2 px-1">Model</TableHead>
                        <TableHead className="text-[10px] py-2 px-1">Dataset</TableHead>
                        <TableHead className="text-[10px] py-2 px-1 text-right">Silhouette</TableHead>
                        <TableHead className="text-[10px] py-2 px-1 text-right">Davies-Bouldin</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.top_clustering_models.map((m, idx) => (
                        <TableRow key={m.model_id} className="hover:bg-muted/20 border-border/20">
                          <TableCell className="text-[11px] font-medium py-2 px-1 truncate max-w-[90px]">
                            <Link to="/models/$modelId" params={{ modelId: m.model_id }} className="hover:underline">
                              {idx + 1}. {m.model}
                            </Link>
                          </TableCell>
                          <TableCell className="text-[11px] text-muted-foreground py-2 px-1 truncate max-w-[80px]">
                            {m.dataset_name}
                          </TableCell>
                          <TableCell className="text-[11px] text-right font-semibold py-2 px-1 text-amber-500">
                            {m.silhouette !== null ? m.silhouette.toFixed(3) : "—"}
                          </TableCell>
                          <TableCell className="text-[11px] text-right text-muted-foreground py-2 px-1">
                            {m.davies_bouldin !== null ? m.davies_bouldin.toFixed(3) : "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </ScrollReveal>
      </div>

      {/* Quick Actions */}
      <ScrollReveal>
        <Card className="border-border/60 bg-card/45 backdrop-blur-md">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">Quick Actions</CardTitle>
            <CardDescription className="text-xs">Common workflows and tools</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7 gap-3">
              <QuickAction
                to="/new/upload"
                icon={LineChartIcon}
                title="Regression Analysis"
                subtitle="Build regression model"
                gradient="from-violet-500 to-purple-600"
              />
              <QuickAction
                to="/classify/upload"
                icon={Shapes}
                title="Classification Analysis"
                subtitle="Build classification model"
                gradient="from-cyan-500 to-blue-600"
              />
              <QuickAction
                to="/cluster/upload"
                icon={Network}
                title="Clustering Analysis"
                subtitle="Discover natural groups"
                gradient="from-amber-500 to-orange-600"
              />
              <QuickAction
                to="/new/upload"
                icon={Upload}
                title="Upload Dataset"
                subtitle="Import your data"
                gradient="from-blue-500 to-indigo-600"
              />
              <QuickAction
                to="/compare"
                icon={Layers}
                title="Compare Models"
                subtitle="Compare performance"
                gradient="from-purple-500 to-pink-600"
              />
              <QuickAction
                to="/assistant"
                icon={MessageCircleQuestion}
                title="AI Assistant"
                subtitle="Get AI help"
                gradient="from-rose-500 to-orange-600"
              />
              <QuickAction
                to="/reports"
                icon={FolderOpen}
                title="Generate Report"
                subtitle="Download insights"
                gradient="from-amber-500 to-orange-600"
              />
            </div>
          </CardContent>
        </Card>
      </ScrollReveal>

      {/* AI Insights block */}
      <ScrollReveal>
        <div className="border border-border/60 bg-card/25 backdrop-blur-md rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="h-4 w-4 text-violet-500 animate-pulse" />
            <span className="text-sm font-semibold">AI Insights</span>
            <span className="text-xs text-muted-foreground font-normal">
              · Smart observations from your recent analyses
            </span>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {/* Insight 1: Best Regression Model */}
            <div className="relative overflow-hidden bg-violet-950/10 border border-violet-500/20 hover:border-violet-500/40 rounded-lg p-3 transition-colors">
              <div className="absolute right-2 bottom-0 text-violet-500/10 scale-150">
                <TrendingUp className="h-10 w-10" />
              </div>
              <div className="flex items-start gap-2.5">
                <div className="mt-0.5 p-1 rounded bg-violet-500/10 text-violet-500">
                  <Sparkles className="h-3 w-3" />
                </div>
                <div className="text-xs leading-relaxed">
                  {data?.best_model ? (
                    <span>
                      Your <span className="font-semibold text-violet-400">{data.best_model.model}</span> model on{" "}
                      <span className="font-semibold text-violet-400">{data.best_model.dataset_name}</span> has the
                      highest R² score of <span className="font-semibold text-violet-400">{data.best_model.r2.toFixed(3)}</span>.
                    </span>
                  ) : (
                    <span>Train a regression model to unlock smart performance insights.</span>
                  )}
                </div>
              </div>
            </div>

            {/* Insight 2: Best Classification Model */}
            <div className="relative overflow-hidden bg-teal-950/10 border border-teal-500/20 hover:border-teal-500/40 rounded-lg p-3 transition-colors">
              <div className="absolute right-2 bottom-0 text-teal-500/10 scale-150">
                <Target className="h-10 w-10" />
              </div>
              <div className="flex items-start gap-2.5">
                <div className="mt-0.5 p-1 rounded bg-teal-500/10 text-teal-500">
                  <Sparkles className="h-3 w-3" />
                </div>
                <div className="text-xs leading-relaxed">
                  {data?.best_accuracy_model ? (
                    <span>
                      <span className="font-semibold text-teal-400">{data.best_accuracy_model.model}</span> on{" "}
                      <span className="font-semibold text-teal-400">{data.best_accuracy_model.dataset_name}</span> achieved the
                      best accuracy of <span className="font-semibold text-teal-400">{(data.best_accuracy_model.accuracy * 100).toFixed(1)}%</span>.
                    </span>
                  ) : (
                    <span>Train a classification model to see key accuracy highlights.</span>
                  )}
                </div>
              </div>
            </div>

            {/* Insight 3: Datasets count achievement */}
            <div className="relative overflow-hidden bg-blue-950/10 border border-blue-500/20 hover:border-blue-500/40 rounded-lg p-3 transition-colors">
              <div className="absolute right-2 bottom-0 text-blue-500/10 scale-150">
                <Database className="h-10 w-10" />
              </div>
              <div className="flex items-start gap-2.5">
                <div className="mt-0.5 p-1 rounded bg-blue-500/10 text-blue-500">
                  <Database className="h-3 w-3" />
                </div>
                <div className="text-xs leading-relaxed">
                  {data?.total_datasets ? (
                    <span>
                      You have uploaded <span className="font-semibold text-blue-400">{data.total_datasets} different datasets</span> in your workspace. Keep up the good work!
                    </span>
                  ) : (
                    <span>Import your first dataset to start training custom models.</span>
                  )}
                </div>
              </div>
            </div>

            {/* Insight 4: Actionable Tip */}
            <div className="relative overflow-hidden bg-amber-950/10 border border-amber-500/20 hover:border-amber-500/40 rounded-lg p-3 transition-colors">
              <div className="absolute right-2 bottom-0 text-amber-500/10 scale-150">
                <Lightbulb className="h-10 w-10" />
              </div>
              <div className="flex items-start gap-2.5">
                <div className="mt-0.5 p-1 rounded bg-amber-500/10 text-amber-500">
                  <Lightbulb className="h-3 w-3" />
                </div>
                <div className="text-xs leading-relaxed">
                  <span>
                    Try feature engineering or outlier removal on your datasets before training to boost model R² and Accuracy scores.
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </ScrollReveal>

      {/* Floating "Ask AI Assistant" button */}
      <div className="fixed bottom-6 right-6 z-40">
        <Button
          asChild
          size="lg"
          className="rounded-full bg-violet-600 hover:bg-violet-700 text-white shadow-xl shadow-violet-500/30 flex items-center gap-2 border-violet-500/20"
        >
          <Link to="/assistant">
            <MessageCircleQuestion className="h-4.5 w-4.5" /> Ask AI Assistant
          </Link>
        </Button>
      </div>
    </div>
  );
}

function QuickAction({
  to,
  icon: Icon,
  title,
  subtitle,
  gradient,
}: {
  to: string;
  icon: any;
  title: string;
  subtitle: string;
  gradient: string;
}) {
  return (
    <Link
      to={to}
      className="flex flex-col gap-2 rounded-lg border border-border p-3 hover:bg-muted/40 transition-all hover:border-primary/20"
    >
      <div className={`h-8 w-8 rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center text-white`}>
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <div className="text-[11px] font-semibold leading-tight text-foreground">{title}</div>
        <div className="text-[10px] text-muted-foreground leading-tight mt-0.5">{subtitle}</div>
      </div>
    </Link>
  );
}
