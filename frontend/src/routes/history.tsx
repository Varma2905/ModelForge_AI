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
import { useModelsList } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { PlusCircle, History, Search } from "lucide-react";
import type { ModelListItem } from "@/lib/api-types";

export const Route = createFileRoute("/history")({
  beforeLoad: requireAuth,
  component: HistoryPage,
});

type SortKey = "date" | "r2";

const PAGE_SIZE = 8;

function HistoryPage() {
  const { data, isLoading, isError } = useModelsList();
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    let rows = q
      ? data.filter(
          (m) => m.dataset_name.toLowerCase().includes(q) || m.model.toLowerCase().includes(q),
        )
      : data;
    rows = [...rows].sort((a, b) => {
      if (sortKey === "r2") {
        const ra = typeof a.metrics.R2 === "number" ? a.metrics.R2 : -Infinity;
        const rb = typeof b.metrics.R2 === "number" ? b.metrics.R2 : -Infinity;
        return rb - ra;
      }
      return (b.created_at ?? "").localeCompare(a.created_at ?? "");
    });
    return rows;
  }, [data, query, sortKey]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const setQueryAndResetPage = (v: string) => {
    setQuery(v);
    setPage(1);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dataset History</h1>
          <p className="text-sm text-muted-foreground">Past regression analyses and their results.</p>
        </div>
        <Button asChild variant="gradient">
          <Link to="/new/upload">New Analysis</Link>
        </Button>
      </div>
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
          <CardTitle>All Analyses</CardTitle>
          {data && data.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(e) => setQueryAndResetPage(e.target.value)}
                  placeholder="Search dataset or model…"
                  className="h-8 w-56 pl-8 text-xs"
                />
              </div>
              <Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
                <SelectTrigger className="h-8 w-36 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="date">Newest first</SelectItem>
                  <SelectItem value="r2">Highest R²</SelectItem>
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
            <p className="text-sm text-destructive">Couldn't load your analysis history.</p>
          ) : !data || data.length === 0 ? (
            <EmptyState
              icon={History}
              title="No analyses yet"
              description="Start your first regression analysis by uploading a dataset."
              action={{ label: "New Analysis", to: "/new/upload" }}
            />
          ) : filtered.length === 0 ? (
            <EmptyState icon={Search} title="No matches" description="Try a different search term." />
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Dataset</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>R²</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pageRows.map((m: ModelListItem) => (
                    <TableRow key={m.model_id}>
                      <TableCell className="font-medium">{m.dataset_name}</TableCell>
                      <TableCell>{m.model}</TableCell>
                      <TableCell>
                        <Badge variant="secondary">
                          {typeof m.metrics.R2 === "number" ? m.metrics.R2.toFixed(3) : "—"}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {m.created_at ? new Date(m.created_at).toLocaleDateString() : "—"}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button asChild variant="ghost" size="sm">
                          <Link to="/models/$modelId" params={{ modelId: m.model_id }}>
                            View
                          </Link>
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
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
    </div>
  );
}
