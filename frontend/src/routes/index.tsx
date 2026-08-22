import { createFileRoute, Link } from "@tanstack/react-router";
import { formatDistanceToNow } from "date-fns";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/metric-card";
import { GradientIcon } from "@/components/gradient-icon";
import { EmptyState } from "@/components/empty-state";
import {
  Database,
  Brain,
  FileText,
  TrendingUp,
  ArrowRight,
  Sparkles,
  PlusCircle,
  MessageCircleQuestion,
  Plug,
  LineChart as LineChartIcon,
} from "lucide-react";
import { useDashboardSummary } from "@/hooks/use-dashboard";
import { useChartTheme } from "@/lib/chart-theme";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/")({
  beforeLoad: requireAuth,
  component: Dashboard,
});

function Dashboard() {
  const { data, isLoading, isError } = useDashboardSummary();
  const chartTheme = useChartTheme();

  const recent = data?.recent_analyses ?? [];
  // Chronological (oldest first) so the sparkline/chart reads left-to-right.
  const recentChrono = [...recent].reverse();

  const stats = [
    {
      label: "Datasets",
      value: data?.total_datasets ?? 0,
      icon: Database,
      gradient: "from-violet-500 to-purple-600",
    },
    {
      label: "Analyses Trained",
      value: data?.total_analyses ?? 0,
      icon: Brain,
      gradient: "from-cyan-500 to-blue-600",
    },
    {
      label: "Reports Generated",
      value: data?.total_reports ?? 0,
      icon: FileText,
      gradient: "from-emerald-500 to-teal-600",
    },
    {
      label: "Avg R² Score",
      value: data?.avg_r2 !== null && data?.avg_r2 !== undefined ? data.avg_r2.toFixed(3) : "—",
      icon: TrendingUp,
      gradient: "from-amber-500 to-orange-600",
      sparklineData: recentChrono
        .map((a) => a.r2)
        .filter((r2): r2 is number => r2 !== null),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="gradient-animated relative overflow-hidden rounded-2xl bg-[image:var(--gradient-brand)] p-8 text-white shadow-xl shadow-primary/20">
        <div className="pointer-events-none absolute -top-24 -right-24 h-72 w-72 rounded-full bg-white/10 blur-3xl animate-pulse" />
        <div className="pointer-events-none absolute -bottom-16 -left-16 h-56 w-56 rounded-full bg-white/10 blur-3xl" />
        <div className="relative">
          <Badge className="bg-white/20 hover:bg-white/20 text-white border-0 mb-3">
            <Sparkles className="h-3 w-3 mr-1" /> Agentic AI Powered
          </Badge>
          <h1 className="text-3xl md:text-4xl font-bold">AI Regression Studio</h1>
          <p className="mt-2 max-w-2xl text-white/85">
            Automate the entire regression workflow — upload data, preprocess, train models, visualize
            results, and get plain-English AI explanations plus a downloadable PDF report.
          </p>
          <Button asChild size="lg" className="mt-5 bg-white text-violet-700 hover:bg-white/90">
            <Link to="/new/upload">
              Start Analysis <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      {/* KPI cards */}
      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full rounded-xl" />
          ))}
        </div>
      ) : isError ? (
        <Card>
          <CardContent className="p-6 text-sm text-destructive">
            Couldn't load your dashboard stats. Try refreshing the page.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((s) => (
            <MetricCard
              key={s.label}
              label={s.label}
              value={s.value}
              icon={s.icon}
              gradient={s.gradient}
              sparklineData={"sparklineData" in s ? s.sparklineData : undefined}
            />
          ))}
        </div>
      )}

      {/* Performance chart + quick actions */}
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">Recent Analyses — R² by Model</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-56 w-full" />
            ) : recentChrono.length === 0 ? (
              <EmptyState
                icon={LineChartIcon}
                title="No analyses yet"
                description="Train your first model to see performance here."
                action={{ label: "New Analysis", to: "/new/upload" }}
              />
            ) : (
              <div style={{ width: "100%", height: 240 }}>
                <ResponsiveContainer>
                  <BarChart data={recentChrono}>
                    <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
                    <XAxis
                      dataKey="model"
                      tick={chartTheme.axisTick}
                      axisLine={{ stroke: chartTheme.border }}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tick={chartTheme.axisTick}
                      axisLine={{ stroke: chartTheme.border }}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={chartTheme.tooltipStyle}
                      labelStyle={chartTheme.tooltipLabelStyle}
                      itemStyle={chartTheme.tooltipItemStyle}
                    />
                    <Bar dataKey="r2" fill={chartTheme.series.primary} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/new/upload">
                <PlusCircle className="h-4 w-4 mr-2" /> New Analysis
              </Link>
            </Button>
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/data-sources">
                <Plug className="h-4 w-4 mr-2" /> Connect a Database
              </Link>
            </Button>
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/assistant">
                <MessageCircleQuestion className="h-4 w-4 mr-2" /> Ask AI Assistant
              </Link>
            </Button>
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/reports">
                <FileText className="h-4 w-4 mr-2" /> Browse Reports
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Recent analyses list + AI insights */}
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Recent Analyses</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : recent.length === 0 ? (
              <EmptyState
                icon={Brain}
                title="No analyses yet"
                description="Start your first regression analysis by uploading a dataset."
                action={{ label: "New Analysis", to: "/new/upload" }}
              />
            ) : (
              <div className="space-y-1">
                {recent.map((a) => (
                  <Link
                    key={a.model_id}
                    to="/models/$modelId"
                    params={{ modelId: a.model_id }}
                    className="flex items-center gap-3 rounded-lg p-2 -mx-2 hover:bg-accent transition-colors"
                  >
                    <GradientIcon icon={Brain} size="sm" />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium truncate">{a.dataset_name}</div>
                      <div className="text-xs text-muted-foreground truncate">
                        {a.model} · {formatDistanceToNow(new Date(a.created_at), { addSuffix: true })}
                      </div>
                    </div>
                    {a.r2 !== null && <Badge variant="secondary">R² {a.r2.toFixed(3)}</Badge>}
                  </Link>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="gradient-border">
          <Card variant="glass">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" /> AI Insights
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {isLoading ? (
                <Skeleton className="h-16 w-full" />
              ) : data?.best_model ? (
                <>
                  <p className="text-sm text-muted-foreground">
                    Your best performing model is{" "}
                    <span className="font-medium text-foreground">{data.best_model.model}</span> with an
                    R² score of{" "}
                    <span className="font-medium text-foreground">{data.best_model.r2.toFixed(3)}</span>{" "}
                    on <span className="font-medium text-foreground">{data.best_model.dataset_name}</span>.
                  </p>
                  <Button asChild variant="gradient" size="sm" className="w-full">
                    <Link
                      to="/assistant"
                      search={{
                        prefill: `Explain why ${data.best_model.model} scored R² ${data.best_model.r2.toFixed(3)} on ${data.best_model.dataset_name}, and how I could improve it further.`,
                      }}
                    >
                      Ask AI about this model <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
                    </Link>
                  </Button>
                </>
              ) : (
                <>
                  <p className="text-sm text-muted-foreground">
                    Train your first model to unlock AI-generated insights about its performance.
                  </p>
                  <Button asChild variant="gradient" size="sm" className="w-full">
                    <Link to="/new/upload">
                      Start Analysis <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
                    </Link>
                  </Button>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
