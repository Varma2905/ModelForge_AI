import { useMemo, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FileText, Search, Check, Eye } from "lucide-react";
import type { ModelListItem } from "@/lib/api-types";
import {
  REPORT_TYPE_META,
  type ReportType,
  type ReportSort,
  reportType,
  reportTitle,
  reportPrimaryMetric,
  reportDetailRows,
  sortReports,
  matchesReportSearch,
} from "@/lib/report-display";
import { useModelMetrics, useClusteringModelMetrics } from "@/hooks/use-training";

const FILTERS: { id: "all" | ReportType; label: string }[] = [
  { id: "all", label: "All" },
  { id: "regression", label: "Regression" },
  { id: "classification", label: "Classification" },
  { id: "clustering", label: "Clustering" },
];

// Threshold below which search/filter/sort controls stay hidden — with only
// a couple of reports they're just clutter, not something worth wiring UI
// state for.
const MANY_REPORTS_THRESHOLD = 3;

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-right">{value}</span>
    </div>
  );
}

function ReportContextModal({
  item,
  open,
  onOpenChange,
}: {
  item: ModelListItem;
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const type = reportType(item);
  const isClustering = type === "clustering";
  const modelQuery = useModelMetrics(!isClustering && open ? item.model_id : null);
  const clusterQuery = useClusteringModelMetrics(isClustering && open ? item.model_id : null);
  const isLoading = isClustering ? clusterQuery.isLoading : modelQuery.isLoading;
  const isError = isClustering ? clusterQuery.isError : modelQuery.isError;
  const data = isClustering ? clusterQuery.data : modelQuery.data;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Report Context</DialogTitle>
        </DialogHeader>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-full" />
          </div>
        ) : isError || !data ? (
          <p className="text-sm text-muted-foreground">
            Couldn't load this report's context. The underlying model data may be unavailable.
          </p>
        ) : (
          <div className="space-y-2.5">
            <Row label="Dataset" value={item.dataset_name} />
            {"total_rows" in data && typeof data.total_rows === "number" && (
              <Row label="Rows" value={data.total_rows.toLocaleString()} />
            )}
            <Row label="Features" value={String(data.features.length)} />
            <Row label={isClustering ? "Algorithm" : "Model"} value={item.model} />
            {"target" in data && data.target && <Row label="Target" value={data.target} />}

            <div className="pt-2 border-t space-y-2.5">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                Evaluation Metrics
              </p>
              {Object.entries(data.metrics)
                .filter((entry): entry is [string, number] => typeof entry[1] === "number")
                .map(([k, v]) => (
                  <Row key={k} label={k} value={v.toFixed(4)} />
                ))}
            </div>

            {"statistical_analysis" in data && data.statistical_analysis?.f_statistic != null && (
              <div className="pt-2 border-t space-y-2.5">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Statistical Results
                </p>
                <Row label="F-statistic" value={data.statistical_analysis.f_statistic.toFixed(3)} />
                <Row label="F p-value" value={data.statistical_analysis.f_pvalue.toFixed(4)} />
              </div>
            )}

            {"cluster_sizes" in data && Object.keys(data.cluster_sizes).length > 0 && (
              <div className="pt-2 border-t space-y-2">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Cluster Sizes
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(data.cluster_sizes).map(([k, v]) => (
                    <Badge key={k} variant="outline" className="text-xs font-normal">
                      {k === "-1" ? "Noise" : `Cluster ${k}`}: {v}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

export function ReportSelectorPanel({
  reports,
  isLoading,
  isError,
  selectedReportId,
  onSelect,
}: {
  reports: ModelListItem[];
  isLoading: boolean;
  isError: boolean;
  selectedReportId: string | null;
  onSelect: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | ReportType>("all");
  const [sort, setSort] = useState<ReportSort>("recent");
  const [contextModalOpen, setContextModalOpen] = useState(false);

  const filtered = useMemo(() => {
    let list = reports;
    if (filter !== "all") list = list.filter((r) => reportType(r) === filter);
    list = list.filter((r) => matchesReportSearch(r, search));
    return sortReports(list, sort);
  }, [reports, filter, search, sort]);

  const selected = reports.find((r) => r.model_id === selectedReportId) ?? null;
  const showControls = reports.length > MANY_REPORTS_THRESHOLD;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="p-3 border-b space-y-2 flex-shrink-0">
        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          Generated Reports
        </p>

        {showControls && (
          <>
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search reports…"
                className="h-8 pl-8 text-xs"
              />
            </div>
            <div className="flex flex-wrap gap-1">
              {FILTERS.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => setFilter(f.id)}
                  className={`text-[11px] rounded-full border px-2.5 py-1 transition-colors ${
                    filter === f.id
                      ? "bg-primary text-primary-foreground border-primary"
                      : "hover:bg-accent border-border"
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
            <Select value={sort} onValueChange={(v) => setSort(v as ReportSort)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="recent">Recent</SelectItem>
                <SelectItem value="oldest">Oldest</SelectItem>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="type">Analysis Type</SelectItem>
              </SelectContent>
            </Select>
          </>
        )}
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {isLoading ? (
          <div className="p-3 space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full rounded-lg" />
            ))}
          </div>
        ) : isError ? (
          <p className="p-4 text-xs text-muted-foreground">Couldn't load your reports.</p>
        ) : reports.length === 0 ? (
          <EmptyState
            icon={FileText}
            title="No reports yet"
            description="Generate a regression, classification, or clustering report first."
            action={{ label: "Go to Reports", to: "/reports" }}
            className="py-8 px-3"
          />
        ) : filtered.length === 0 ? (
          <p className="p-4 text-xs text-muted-foreground text-center">No reports match your search.</p>
        ) : (
          <div className="p-2 space-y-1.5">
            {filtered.map((item) => {
              const meta = REPORT_TYPE_META[reportType(item)];
              const Icon = meta.icon;
              const primary = reportPrimaryMetric(item);
              const isSelected = item.model_id === selectedReportId;
              return (
                <button
                  key={item.model_id}
                  type="button"
                  onClick={() => onSelect(item.model_id)}
                  className={`w-full text-left rounded-lg border-l-2 px-2.5 py-2 transition-colors ${
                    isSelected ? `${meta.accentBorder} bg-accent` : "border-l-transparent hover:bg-accent/50"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <Icon className={`h-3.5 w-3.5 mt-0.5 flex-shrink-0 ${meta.text}`} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        {isSelected && <Check className="h-3 w-3 flex-shrink-0" />}
                        <span className="text-xs font-medium truncate">{reportTitle(item)}</span>
                      </div>
                      <p className="text-[11px] text-muted-foreground truncate">{item.model}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        {primary && (
                          <span className="text-[11px] font-medium">
                            {primary.label}: {primary.value}
                          </span>
                        )}
                        <span className="text-[10px] text-muted-foreground">
                          {item.created_at
                            ? formatDistanceToNow(new Date(item.created_at), { addSuffix: true })
                            : ""}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {selected && (
        <div className="border-t p-3 space-y-2 flex-shrink-0">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Selected Report
          </p>
          <p className="text-sm font-medium">{reportTitle(selected)}</p>
          <div className="space-y-1">
            {reportDetailRows(selected).map((row) => (
              <div key={row.label} className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{row.label}</span>
                <span className="font-medium">{row.value}</span>
              </div>
            ))}
          </div>
          <Button size="sm" variant="outline" className="w-full" onClick={() => setContextModalOpen(true)}>
            <Eye className="h-3.5 w-3.5 mr-1.5" /> View Report Context
          </Button>
          <ReportContextModal item={selected} open={contextModalOpen} onOpenChange={setContextModalOpen} />
        </div>
      )}
    </div>
  );
}
