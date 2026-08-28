import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { EmptyState } from "@/components/empty-state";
import { ScrollReveal } from "@/components/scroll-reveal";
import { useModelsList, useDeleteModel } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { Layers, Search, Trash2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import type { ClassificationMetrics, ModelListItem, ModelMetrics, ModelType } from "@/lib/api-types";

// A ModelListItem's `metrics` field is declared as the regression
// ModelMetrics shape, but the actual JSON for a classification model doc
// carries ClassificationMetrics's keys (Accuracy/Precision/...) instead —
// this permissive type lets both be read generically without an unsafe cast
// at every access site.
type MixedMetrics = ModelMetrics & Partial<ClassificationMetrics>;

export const Route = createFileRoute("/history")({
  beforeLoad: requireAuth,
  component: HistoryPage,
});

type SortKey = "date" | "score";
type TypeFilter = "all" | ModelType;

const PAGE_SIZE = 8;

// A model doc trained before model_type existed has no such field at all —
// every consumer must default a missing value to "regression".
function modelType(m: ModelListItem): ModelType {
  return m.model_type ?? "regression";
}

function score(m: ModelListItem): number {
  const metrics = m.metrics as MixedMetrics;
  const val = modelType(m) === "classification" ? metrics.Accuracy : metrics.R2;
  return typeof val === "number" ? val : -Infinity;
}

function HistoryPage() {
  const { data, isLoading, isError } = useModelsList();
  const deleteMutation = useDeleteModel();
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all");
  const [page, setPage] = useState(1);

  const handleDelete = async (m: ModelListItem) => {
    if (!window.confirm(`Delete "${m.dataset_name} — ${m.model}"? This can't be undone.`)) {
      return;
    }
    try {
      await deleteMutation.mutateAsync(m.model_id);
      toast.success("Model deleted");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to delete model");
    }
  };

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    let rows = data.filter((m) => typeFilter === "all" || modelType(m) === typeFilter);
    if (q) {
      rows = rows.filter(
        (m) => m.dataset_name.toLowerCase().includes(q) || m.model.toLowerCase().includes(q),
      );
    }
    rows = [...rows].sort((a, b) => {
      if (sortKey === "score") return score(b) - score(a);
      return (b.created_at ?? "").localeCompare(a.created_at ?? "");
    });
    return rows;
  }, [data, query, sortKey, typeFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const setQueryAndResetPage = (v: string) => {
    setQuery(v);
    setPage(1);
  };

  const setTypeFilterAndResetPage = (v: TypeFilter) => {
    setTypeFilter(v);
    setPage(1);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Saved Models</h1>
          <p className="text-sm text-muted-foreground">
            Every regression and classification model you've trained.
          </p>
        </div>
        <Button asChild variant="gradient">
          <Link to="/new/upload">New Analysis</Link>
        </Button>
      </div>
      <ScrollReveal>
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-3 flex-wrap space-y-0">
          <CardTitle>All Models</CardTitle>
          {data && data.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(e) => setQueryAndResetPage(e.target.value)}
                  placeholder="Search dataset or model…"
                  className="h-8 w-56 pl-8 text-xs"
                />
              </div>
              <Select value={typeFilter} onValueChange={(v) => setTypeFilterAndResetPage(v as TypeFilter)}>
                <SelectTrigger className="h-8 w-36 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  <SelectItem value="regression">Regression</SelectItem>
                  <SelectItem value="classification">Classification</SelectItem>
                </SelectContent>
              </Select>
              <Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
                <SelectTrigger className="h-8 w-36 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="date">Newest first</SelectItem>
                  <SelectItem value="score">Best score</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : isError ? (
            <p className="text-sm text-destructive">Couldn't load your saved models.</p>
          ) : !data || data.length === 0 ? (
            <EmptyState
              icon={Layers}
              title="No models yet"
              description="Train your first regression or classification model to see it here."
              action={{ label: "New Analysis", to: "/new/upload" }}
            />
          ) : filtered.length === 0 ? (
            <EmptyState icon={Search} title="No matches" description="Try a different search term or filter." />
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Dataset</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pageRows.map((m: ModelListItem) => {
                    const type = modelType(m);
                    const isClassification = type === "classification";
                    const metrics = m.metrics as MixedMetrics;
                    const scoreVal = isClassification ? metrics.Accuracy : metrics.R2;
                    return (
                      <TableRow key={m.model_id}>
                        <TableCell className="font-medium">{m.dataset_name}</TableCell>
                        <TableCell>{m.model}</TableCell>
                        <TableCell>
                          <Badge variant={isClassification ? "outline" : "secondary"} className="text-[10px]">
                            {isClassification ? "Classification" : "Regression"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant="secondary">
                            {typeof scoreVal === "number"
                              ? isClassification
                                ? `${(scoreVal * 100).toFixed(1)}%`
                                : scoreVal.toFixed(3)
                              : "—"}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {m.created_at ? new Date(m.created_at).toLocaleDateString() : "—"}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button asChild variant="ghost" size="sm">
                              <Link to="/models/$modelId" params={{ modelId: m.model_id }}>
                                View
                              </Link>
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-destructive hover:text-destructive"
                              aria-label={`Delete ${m.dataset_name} — ${m.model}`}
                              disabled={deleteMutation.isPending && deleteMutation.variables === m.model_id}
                              onClick={() => handleDelete(m)}
                            >
                              {deleteMutation.isPending && deleteMutation.variables === m.model_id ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              ) : (
                                <Trash2 className="h-3.5 w-3.5" />
                              )}
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>

              {totalPages > 1 && (
                <Pagination className="mt-4 justify-between">
                  <p className="text-xs text-muted-foreground">
                    Page {currentPage} of {totalPages} · {filtered.length} total
                  </p>
                  <PaginationContent>
                    <PaginationItem>
                      <PaginationPrevious
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setPage((p) => Math.max(1, p - 1));
                        }}
                        className={currentPage === 1 ? "pointer-events-none opacity-50" : undefined}
                      />
                    </PaginationItem>
                    {Array.from({ length: totalPages }).map((_, i) => (
                      <PaginationItem key={i}>
                        <PaginationLink
                          href="#"
                          isActive={currentPage === i + 1}
                          onClick={(e) => {
                            e.preventDefault();
                            setPage(i + 1);
                          }}
                        >
                          {i + 1}
                        </PaginationLink>
                      </PaginationItem>
                    ))}
                    <PaginationItem>
                      <PaginationNext
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setPage((p) => Math.min(totalPages, p + 1));
                        }}
                        className={
                          currentPage === totalPages ? "pointer-events-none opacity-50" : undefined
                        }
                      />
                    </PaginationItem>
                  </PaginationContent>
                </Pagination>
              )}
            </>
          )}
        </CardContent>
      </Card>
      </ScrollReveal>
    </div>
  );
}
