import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Link } from "@tanstack/react-router";
import { Database, Loader2, Plug, Search, Table2 } from "lucide-react";
import {
  useDataSourceDatabases,
  useDataSourcePreview,
  useDataSourceTables,
  useDataSourcesList,
  useImportDataSourceTable,
} from "@/hooks/use-data-sources";
import { ApiError } from "@/lib/api-service";
import type { ImportDatasetResult } from "@/lib/api-types";

const DEFAULT_ROW_LIMIT = 10_000;
const MAX_ROW_LIMIT = 100_000;

export function DatabaseImportPanel({ onImported }: { onImported: (result: ImportDatasetResult) => void }) {
  const { data: sources, isLoading: sourcesLoading } = useDataSourcesList();

  const [dataSourceId, setDataSourceId] = useState<string | null>(null);
  const [database, setDatabase] = useState<string | null>(null);
  const [table, setTable] = useState<string | null>(null);
  const [useCustomSql, setUseCustomSql] = useState(false);
  const [customSql, setCustomSql] = useState("");
  const [mongoFilter, setMongoFilter] = useState("");
  const [mongoSortField, setMongoSortField] = useState("");
  const [mongoSortDir, setMongoSortDir] = useState<"1" | "-1">("-1");
  const [rowLimit, setRowLimit] = useState(String(DEFAULT_ROW_LIMIT));
  const [showPreview, setShowPreview] = useState(false);

  const selectedSource = sources?.find((s) => s.id === dataSourceId) ?? null;
  const isMongo = selectedSource?.engine === "mongodb";

  const databasesQuery = useDataSourceDatabases(dataSourceId);
  const tablesQuery = useDataSourceTables(dataSourceId, database);
  const previewQuery = useDataSourcePreview(dataSourceId, database, table, 20);
  const importMutation = useImportDataSourceTable();

  const selectSource = (id: string) => {
    setDataSourceId(id);
    setDatabase(null);
    setTable(null);
    setUseCustomSql(false);
    setCustomSql("");
    setMongoFilter("");
    setShowPreview(false);
  };

  const selectDatabase = (db: string) => {
    setDatabase(db);
    setTable(null);
    setShowPreview(false);
  };

  const selectTable = (t: string) => {
    setTable(t);
    setShowPreview(false);
  };

  const handleImport = async () => {
    if (!dataSourceId || !database) {
      toast.error("Select a data source and database first.");
      return;
    }

    let filter: Record<string, unknown> | undefined;
    if (isMongo && mongoFilter.trim()) {
      try {
        filter = JSON.parse(mongoFilter);
      } catch {
        toast.error("Filter must be valid JSON, e.g. {\"category\": \"Electronics\"}");
        return;
      }
    }

    const limit = Math.min(Number(rowLimit) || DEFAULT_ROW_LIMIT, MAX_ROW_LIMIT);

    try {
      const result = await importMutation.mutateAsync({
        dataSourceId,
        payload: {
          database,
          table: table ?? undefined,
          row_limit: limit,
          custom_sql: !isMongo && useCustomSql && customSql.trim() ? customSql.trim() : undefined,
          filter,
          sort: isMongo && mongoSortField.trim() ? [[mongoSortField.trim(), mongoSortDir === "1" ? 1 : -1]] : undefined,
        },
      });
      onImported(result);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to import dataset");
    }
  };

  if (sourcesLoading) {
    return <Skeleton className="h-40 w-full" />;
  }

  if (!sources || sources.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center">
        <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mb-3">
          <Plug className="h-6 w-6 text-muted-foreground" />
        </div>
        <p className="font-medium">No data sources connected yet</p>
        <p className="text-xs text-muted-foreground mt-1 mb-4">
          Connect a MySQL, PostgreSQL, or MongoDB database first.
        </p>
        <Button asChild size="sm">
          <Link to="/data-sources">Go to Data Sources</Link>
        </Button>
      </div>
    );
  }

  const canImport = !!dataSourceId && !!database && (isMongo ? !!table : useCustomSql ? !!customSql.trim() : !!table);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="space-y-1.5">
          <Label className="text-xs">Data source</Label>
          <Select value={dataSourceId ?? undefined} onValueChange={selectSource}>
            <SelectTrigger>
              <SelectValue placeholder="Select a data source" />
            </SelectTrigger>
            <SelectContent>
              {sources.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-xs">Database</Label>
          <Select value={database ?? undefined} onValueChange={selectDatabase} disabled={!dataSourceId || databasesQuery.isLoading}>
            <SelectTrigger>
              <SelectValue placeholder={databasesQuery.isLoading ? "Loading…" : "Select a database"} />
            </SelectTrigger>
            <SelectContent>
              {databasesQuery.data?.databases.map((db) => (
                <SelectItem key={db} value={db}>
                  {db}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-xs">{isMongo ? "Collection" : "Table"}</Label>
          <Select value={table ?? undefined} onValueChange={selectTable} disabled={!database || tablesQuery.isLoading}>
            <SelectTrigger>
              <SelectValue placeholder={tablesQuery.isLoading ? "Loading…" : `Select a ${isMongo ? "collection" : "table"}`} />
            </SelectTrigger>
            <SelectContent>
              {tablesQuery.data?.tables.map((t) => (
                <SelectItem key={t.name} value={t.name}>
                  {t.name}
                  {t.row_count_estimate != null ? ` (~${t.row_count_estimate.toLocaleString()} rows)` : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {table && tablesQuery.data && (
        <div className="flex flex-wrap gap-1.5">
          {tablesQuery.data.tables
            .find((t) => t.name === table)
            ?.columns.slice(0, 12)
            .map((c) => (
              <Badge key={c.name} variant="outline" className="font-normal text-xs">
                {c.name}
                <span className="text-muted-foreground ml-1">{c.type}</span>
              </Badge>
            ))}
        </div>
      )}

      {database && !isMongo && (
        <div className="space-y-2 rounded-lg border p-3">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="use-custom-sql"
              checked={useCustomSql}
              onChange={(e) => setUseCustomSql(e.target.checked)}
              className="h-3.5 w-3.5"
            />
            <Label htmlFor="use-custom-sql" className="text-xs font-normal cursor-pointer">
              Use a custom read-only SQL query instead
            </Label>
          </div>
          {useCustomSql && (
            <>
              <Textarea
                value={customSql}
                onChange={(e) => setCustomSql(e.target.value)}
                placeholder="SELECT age, income, experience, salary FROM employees WHERE department = 'Engineering'"
                className="font-mono text-xs h-24"
              />
              <p className="text-[11px] text-muted-foreground">
                Read-only SELECT statements only. INSERT/UPDATE/DELETE/DROP and multi-statement queries are rejected.
              </p>
            </>
          )}
        </div>
      )}

      {database && isMongo && table && (
        <div className="space-y-2 rounded-lg border p-3">
          <Label className="text-xs">Filter (JSON, optional)</Label>
          <Textarea
            value={mongoFilter}
            onChange={(e) => setMongoFilter(e.target.value)}
            placeholder='{"category": "Electronics", "price": {"$gt": 100}}'
            className="font-mono text-xs h-16"
          />
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Sort field (optional)</Label>
              <Input value={mongoSortField} onChange={(e) => setMongoSortField(e.target.value)} placeholder="price" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Sort direction</Label>
              <Select value={mongoSortDir} onValueChange={(v) => setMongoSortDir(v as "1" | "-1")}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="-1">Descending</SelectItem>
                  <SelectItem value="1">Ascending</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-end gap-3">
        <div className="space-y-1.5">
          <Label className="text-xs">Row limit</Label>
          <Input
            className="w-32"
            value={rowLimit}
            onChange={(e) => setRowLimit(e.target.value)}
            placeholder={String(DEFAULT_ROW_LIMIT)}
          />
        </div>
        <p className="text-[11px] text-muted-foreground pb-2">
          Capped at {MAX_ROW_LIMIT.toLocaleString()} rows. You'll be told if the import was truncated.
        </p>
      </div>

      <div className="flex gap-2">
        <Button
          variant="outline"
          onClick={() => setShowPreview(true)}
          disabled={!database || (!table && !useCustomSql) || previewQuery.isFetching}
        >
          {previewQuery.isFetching ? <Loader2 className="h-4 w-4 animate-spin mr-1.5" /> : <Search className="h-4 w-4 mr-1.5" />}
          Preview
        </Button>
        <Button onClick={handleImport} disabled={!canImport || importMutation.isPending}>
          {importMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1.5" /> : <Table2 className="h-4 w-4 mr-1.5" />}
          Import as Dataset
        </Button>
      </div>

      {showPreview && table && (
        <div className="rounded-lg border overflow-hidden">
          {previewQuery.isLoading ? (
            <div className="p-4">
              <Skeleton className="h-32 w-full" />
            </div>
          ) : previewQuery.isError ? (
            <p className="p-4 text-sm text-destructive">Couldn't load a preview for this table.</p>
          ) : previewQuery.data ? (
            <>
              <div className="flex items-center gap-2 px-3 py-2 bg-muted text-xs text-muted-foreground">
                <Database className="h-3.5 w-3.5" />
                Sampled preview — {previewQuery.data.row_count_returned} rows shown, statistics computed over this
                sample only.
              </div>
              <div className="overflow-x-auto max-h-64">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 sticky top-0">
                    <tr>
                      {previewQuery.data.columns.map((c) => (
                        <th key={c} className="text-left p-2 font-medium whitespace-nowrap">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewQuery.data.rows.map((r, i) => (
                      <tr key={i} className="border-t">
                        {r.map((v, j) => (
                          <td key={j} className="p-2 whitespace-nowrap">
                            {v === null ? <span className="text-muted-foreground">—</span> : String(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
        </div>
      )}
    </div>
  );
}
