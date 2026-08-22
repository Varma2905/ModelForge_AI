import { createFileRoute, Link } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/metric-card";
import {
  Database,
  Brain,
  FileText,
  TrendingUp,
  ArrowRight,
  Sparkles,
  PlusCircle,
} from "lucide-react";
import { useDashboardSummary } from "@/hooks/use-dashboard";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/")({
  beforeLoad: requireAuth,
  component: Dashboard,
});

function Dashboard() {
  const { data, isLoading, isError } = useDashboardSummary();

  const stats = [
    {
      label: "Datasets",
      value: data?.total_datasets ?? 0,
      icon: Database,
      gradient: "from-indigo-500 to-blue-500",
    },
    {
      label: "Analyses Trained",
      value: data?.total_analyses ?? 0,
      icon: Brain,
      gradient: "from-fuchsia-500 to-pink-500",
    },
    {
      label: "Reports Generated",
      value: data?.total_reports ?? 0,
      icon: FileText,
      gradient: "from-emerald-500 to-teal-500",
    },
    {
      label: "Avg R² Score",
      value: data?.avg_r2 !== null && data?.avg_r2 !== undefined ? data.avg_r2.toFixed(3) : "—",
      icon: TrendingUp,
      gradient: "from-amber-500 to-orange-500",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="rounded-2xl bg-gradient-to-br from-indigo-600 via-purple-600 to-fuchsia-600 p-8 text-white shadow-lg">
        <Badge className="bg-white/20 hover:bg-white/20 text-white border-0 mb-3">
          <Sparkles className="h-3 w-3 mr-1" /> Agentic AI Powered
        </Badge>
        <h1 className="text-3xl md:text-4xl font-bold">AI Regression Studio</h1>
        <p className="mt-2 max-w-2xl text-white/85">
          Automate the entire regression workflow — upload data, preprocess, train models, visualize
          results, and get plain-English AI explanations plus a downloadable PDF report.
        </p>
        <Button asChild size="lg" className="mt-5 bg-white text-indigo-700 hover:bg-white/90">
          <Link to="/new/upload">
            Start Analysis <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>

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
            />
          ))}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Best Performing Model</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-16 w-full" />
            ) : data?.best_model ? (
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{data.best_model.dataset_name}</div>
                  <div className="text-xs text-muted-foreground">{data.best_model.model}</div>
                </div>
                <Badge variant="secondary">R² {data.best_model.r2.toFixed(3)}</Badge>
              </div>
            ) : (
              <EmptyState />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Quick Start</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/new/upload">
                <PlusCircle className="h-4 w-4 mr-2" /> New Analysis
              </Link>
            </Button>
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/assistant">Ask AI assistant</Link>
            </Button>
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/reports">Browse reports</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="text-center py-6">
      <p className="text-sm font-medium">No analyses yet</p>
      <p className="text-xs text-muted-foreground mt-1">
        Start your first regression analysis by uploading a dataset.
      </p>
      <Button asChild size="sm" className="mt-4">
        <Link to="/new/upload">
          <PlusCircle className="h-3.5 w-3.5 mr-1.5" /> New Analysis
        </Link>
      </Button>
    </div>
  );
}
