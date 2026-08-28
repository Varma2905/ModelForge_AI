import { TrendingUp, Target, Network, type LucideIcon } from "lucide-react";
import type { ModelListItem, ModelType, ModelMetrics, ClassificationMetrics, ClusteringMetrics } from "@/lib/api-types";

export type ReportType = "regression" | "classification" | "clustering";

// A ModelListItem's `metrics` field is declared as the regression shape, but
// the actual JSON for a classification/clustering report carries that task's
// own keys instead — same pattern as compare.tsx's MixedMetrics.
type AnyMetrics = ModelMetrics & Partial<ClassificationMetrics> & Partial<ClusteringMetrics>;

export function reportType(item: { model_type?: ModelType }): ReportType {
  return item.model_type === "classification" || item.model_type === "clustering"
    ? item.model_type
    : "regression";
}

export const REPORT_TYPE_META: Record<
  ReportType,
  { label: string; icon: LucideIcon; accentBorder: string; text: string; badgeBg: string }
> = {
  regression: {
    label: "Regression",
    icon: TrendingUp,
    accentBorder: "border-l-violet-500",
    text: "text-violet-400",
    badgeBg: "bg-violet-500/10",
  },
  classification: {
    label: "Classification",
    icon: Target,
    accentBorder: "border-l-teal-500",
    text: "text-teal-400",
    badgeBg: "bg-teal-500/10",
  },
  clustering: {
    label: "Clustering",
    icon: Network,
    accentBorder: "border-l-amber-500",
    text: "text-amber-400",
    badgeBg: "bg-amber-500/10",
  },
};

function titleCaseDatasetName(name: string): string {
  const stripped = name.replace(/\.(csv|xlsx?|json|parquet)$/i, "");
  return stripped
    .replace(/[_-]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ") || name;
}

// Derives a friendly report title from real dataset_name + model_type —
// never a hard-coded/stored title, since the app has no such field.
export function reportTitle(item: { dataset_name: string; model_type?: ModelType }): string {
  const type = reportType(item);
  return `${titleCaseDatasetName(item.dataset_name)} ${REPORT_TYPE_META[type].label} Report`;
}

export function reportPrimaryMetric(item: ModelListItem): { label: string; value: string } | null {
  const type = reportType(item);
  const m = item.metrics as AnyMetrics;
  if (type === "regression") {
    return typeof m.R2 === "number" ? { label: "R²", value: m.R2.toFixed(3) } : null;
  }
  if (type === "classification") {
    return typeof m.Accuracy === "number" ? { label: "Accuracy", value: `${(m.Accuracy * 100).toFixed(1)}%` } : null;
  }
  return typeof m.Silhouette === "number" ? { label: "Silhouette", value: m.Silhouette.toFixed(3) } : null;
}

// "Only display metrics that actually exist in that report" — every row is
// individually gated on the field being a real number, never hard-coded.
export function reportDetailRows(item: ModelListItem): { label: string; value: string }[] {
  const type = reportType(item);
  const m = item.metrics as AnyMetrics;
  const rows: { label: string; value: string }[] = [{ label: "Type", value: REPORT_TYPE_META[type].label }];

  if (type === "regression") {
    rows.push({ label: "Model", value: item.model }, { label: "Dataset", value: item.dataset_name });
    if (typeof m.R2 === "number") rows.push({ label: "R²", value: m.R2.toFixed(3) });
    if (typeof m.RMSE === "number") rows.push({ label: "RMSE", value: m.RMSE.toFixed(2) });
  } else if (type === "classification") {
    rows.push({ label: "Model", value: item.model }, { label: "Dataset", value: item.dataset_name });
    if (typeof m.Accuracy === "number") rows.push({ label: "Accuracy", value: `${(m.Accuracy * 100).toFixed(1)}%` });
    if (typeof m.F1 === "number") rows.push({ label: "F1 Score", value: m.F1.toFixed(3) });
  } else {
    rows.push({ label: "Algorithm", value: item.model }, { label: "Dataset", value: item.dataset_name });
    if (typeof m.ClusterCount === "number") rows.push({ label: "Clusters", value: String(m.ClusterCount) });
    if (typeof m.Silhouette === "number") rows.push({ label: "Silhouette", value: m.Silhouette.toFixed(3) });
    if (typeof m.DaviesBouldin === "number") rows.push({ label: "Davies-Bouldin", value: m.DaviesBouldin.toFixed(3) });
  }
  return rows;
}

export const SUGGESTIONS_BY_TYPE: Record<ReportType, string[]> = {
  regression: [
    "Why is my R² score low?",
    "How can I improve this model?",
    "Which features are most important?",
    "Is my model overfitting?",
    "How do I interpret the statistical results?",
    "What do the p-values indicate?",
    "Should I try another regression model?",
  ],
  classification: [
    "Why is my accuracy low?",
    "How should I interpret the confusion matrix?",
    "Which class is performing poorly?",
    "Is the model overfitting?",
    "How can I improve precision and recall?",
    "Which classification model should I try?",
  ],
  clustering: [
    "How many clusters were identified?",
    "Are the clusters well separated?",
    "What does the Silhouette Score mean?",
    "What are the characteristics of each cluster?",
    "Which features distinguish the clusters?",
    "Is this clustering result reliable?",
  ],
};

export const PLACEHOLDER_BY_TYPE: Record<ReportType, string> = {
  regression: "Ask me anything about this regression report…",
  classification: "Ask me anything about this classification report…",
  clustering: "Ask me anything about this clustering report…",
};

// Real-time mode — no report selected, so no report-specific grounding is
// sent. General machine learning questions, not tied to any one report.
export const GENERAL_SUGGESTIONS = [
  "What is overfitting?",
  "What's the difference between regression and classification?",
  "How do I choose the right ML algorithm?",
  "What does R² actually measure?",
  "What is a confusion matrix?",
];

export const GENERAL_PLACEHOLDER = "Ask a general ML question, or select a report for report-specific analysis…";

export type ReportSort = "recent" | "oldest" | "name" | "type";

export function sortReports(items: ModelListItem[], sort: ReportSort): ModelListItem[] {
  const copy = [...items];
  switch (sort) {
    case "oldest":
      return copy.sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));
    case "name":
      return copy.sort((a, b) => reportTitle(a).localeCompare(reportTitle(b)));
    case "type":
      return copy.sort((a, b) => reportType(a).localeCompare(reportType(b)));
    case "recent":
    default:
      return copy.sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
  }
}

export function matchesReportSearch(item: ModelListItem, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const haystack = [reportTitle(item), item.dataset_name, item.model, reportType(item)]
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}
