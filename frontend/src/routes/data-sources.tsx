import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Database,
  Leaf,
  Loader2,
  Plug,
  PlusCircle,
  CheckCircle2,
  XCircle,
  Trash2,
  MessageCircleQuestion,
  Save,
} from "lucide-react";
import {
  useCreateDataSource,
  useDataSourceDatabases,
  useDataSourcesList,
  useDeleteDataSource,
  useImportDataSourceTable,
  useTestConnection,
  useTestSavedDataSource,
} from "@/hooks/use-data-sources";
import { useAskDatabaseQuestion } from "@/hooks/use-db-query";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import type {
  ConnectionInput,
  DataSourceEngine,
  DataSourceSummary,
  NLQueryResult,
} from "@/lib/api-types";

export const Route = createFileRoute("/data-sources")({
  beforeLoad: requireAuth,
  component: DataSourcesPage,
});

const ENGINE_LABEL: Record<DataSourceEngine, string> = {
  mysql: "MySQL",
  postgresql: "PostgreSQL",
  mongodb: "MongoDB",
};

function EngineIcon({ engine, className }: { engine: DataSourceEngine; className?: string }) {
  if (engine === "mongodb") return <Leaf className={className} />;
  return <Database className={className} />;
}

type FormState = {
  name: string;
  engine: DataSourceEngine;
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  ssl_enabled: boolean;
  connection_timeout_sec: string;
  uri: string;
};

const DEFAULT_FORM: FormState = {
  name: "",
  engine: "mysql",
  host: "",
  port: "3306",
  database: "",
  username: "",
  password: "",
  ssl_enabled: false,
  connection_timeout_sec: "10",
  uri: "",
};

function buildConnectionInput(form: FormState): ConnectionInput | null {
  const timeout = Number(form.connection_timeout_sec) || 10;
  if (form.engine === "mongodb") {
    if (!form.uri.trim() || !form.database.trim()) return null;
    return { engine: "mongodb", uri: form.uri.trim(), database: form.database.trim(), connection_timeout_sec: timeout };
  }
  const port = Number(form.port);
  if (!form.host.trim() || !port || !form.database.trim() || !form.username.trim() || !form.password) return null;
  return {
    engine: form.engine as "mysql" | "postgresql",
    host: form.host.trim(),
    port,
    database: form.database.trim(),
    username: form.username.trim(),
    password: form.password,
    ssl_enabled: form.ssl_enabled,
    connection_timeout_sec: timeout,
  };
}

function AddConnectionDialog({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  const testMutation = useTestConnection();
  const createMutation = useCreateDataSource();

  const patch = (p: Partial<FormState>) => {
    setForm((f) => ({ ...f, ...p }));
    setTestResult(null);
  };

  const setEngine = (engine: DataSourceEngine) => {
    const defaultPort = engine === "mysql" ? "3306" : engine === "postgresql" ? "5432" : form.port;
    patch({ engine, port: defaultPort });
  };

  const resetAndClose = () => {
    setForm(DEFAULT_FORM);
    setTestResult(null);
    setOpen(false);
  };

  const handleTest = async () => {
    const connection = buildConnectionInput(form);
    if (!connection) {
      toast.error("Fill in all required fields before testing.");
      return;
    }
    try {
      const result = await testMutation.mutateAsync(connection);
      setTestResult({ ok: true, message: result.server_version ? `Connected — server ${result.server_version}` : "Connected successfully" });
    } catch (err) {
      setTestResult({ ok: false, message: err instanceof ApiError ? err.message : "Connection test failed." });
    }
  };

  const handleSave = async () => {
    const connection = buildConnectionInput(form);
    if (!form.name.trim() || !connection) {
      toast.error("Fill in a connection name and all required fields before saving.");
      return;
    }
    try {
      await createMutation.mutateAsync({ name: form.name.trim(), connection });
      toast.success("Data source saved");
      onCreated();
      resetAndClose();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to save data source");
    }
  };

  const busy = testMutation.isPending || createMutation.isPending;

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : resetAndClose())}>
      <DialogTrigger asChild>
        <Button>
          <PlusCircle className="h-4 w-4 mr-1.5" /> Add data source
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Connect a database</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 max-h-[65vh] overflow-y-auto pr-1">
          <div className="space-y-1.5">
            <Label>Connection name</Label>
            <Input
              value={form.name}
              onChange={(e) => patch({ name: e.target.value })}
              placeholder="e.g. Production Analytics DB"
            />
          </div>

          <div className="space-y-1.5">
            <Label>Database type</Label>
            <Select value={form.engine} onValueChange={(v) => setEngine(v as DataSourceEngine)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mysql">MySQL</SelectItem>
                <SelectItem value="postgresql">PostgreSQL</SelectItem>
                <SelectItem value="mongodb">MongoDB</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {form.engine === "mongodb" ? (
            <>
              <div className="space-y-1.5">
                <Label>MongoDB URI</Label>
                <Input
                  value={form.uri}
                  onChange={(e) => patch({ uri: e.target.value })}
                  placeholder="mongodb+srv://user:password@cluster.example.net"
                  type="password"
                />
                <p className="text-xs text-muted-foreground">
                  Stored encrypted. The full URI is never shown again after saving.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label>Database name</Label>
                <Input value={form.database} onChange={(e) => patch({ database: e.target.value })} placeholder="mydb" />
              </div>
            </>
          ) : (
            <>
              <div className="grid grid-cols-3 gap-3">
                <div className="col-span-2 space-y-1.5">
                  <Label>Host</Label>
                  <Input value={form.host} onChange={(e) => patch({ host: e.target.value })} placeholder="127.0.0.1" />
                </div>
                <div className="space-y-1.5">
                  <Label>Port</Label>
                  <Input value={form.port} onChange={(e) => patch({ port: e.target.value })} />
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>Database</Label>
                <Input value={form.database} onChange={(e) => patch({ database: e.target.value })} placeholder="mydb" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <Label>Username</Label>
                  <Input value={form.username} onChange={(e) => patch({ username: e.target.value })} />
                </div>
                <div className="space-y-1.5">
                  <Label>Password</Label>
                  <Input
                    type="password"
                    value={form.password}
                    onChange={(e) => patch({ password: e.target.value })}
                  />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="ssl"
                    checked={form.ssl_enabled}
                    onCheckedChange={(v) => patch({ ssl_enabled: v === true })}
                  />
                  <Label htmlFor="ssl" className="font-normal">
                    Use SSL
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Label className="font-normal text-xs text-muted-foreground">Timeout (sec)</Label>
                  <Input
                    className="w-16 h-8"
                    value={form.connection_timeout_sec}
                    onChange={(e) => patch({ connection_timeout_sec: e.target.value })}
                  />
                </div>
              </div>
            </>
          )}

          {testResult && (
            <div
              className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm ${
                testResult.ok ? "border-green-600/30 bg-green-600/10 text-green-700 dark:text-green-400" : "border-destructive/30 bg-destructive/10 text-destructive"
              }`}
            >
              {testResult.ok ? <CheckCircle2 className="h-4 w-4 flex-shrink-0" /> : <XCircle className="h-4 w-4 flex-shrink-0" />}
              <span>{testResult.message}</span>
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-2">
          <Button variant="outline" onClick={handleTest} disabled={busy}>
            {testMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1.5" /> : <Plug className="h-4 w-4 mr-1.5" />}
            Test Connection
          </Button>
          <Button onClick={handleSave} disabled={busy}>
            {createMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
            Save Connection
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function isMongoQuery(query: NLQueryResult["query"]): query is Extract<NLQueryResult["query"], object> {
  return typeof query === "object" && query !== null;
}

function AskQuestionDialog({ source }: { source: DataSourceSummary }) {
  const [open, setOpen] = useState(false);
  const [database, setDatabase] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<NLQueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState("");

  const databasesQuery = useDataSourceDatabases(open ? source.id : null);
  const askMutation = useAskDatabaseQuestion();
  const importMutation = useImportDataSourceTable();

  const reset = () => {
    setDatabase(null);
    setQuestion("");
    setResult(null);
    setError(null);
    setDatasetName("");
  };

  const handleAsk = async () => {
    if (!database || !question.trim()) return;
    setError(null);
    setResult(null);
    try {
      const res = await askMutation.mutateAsync({ data_source_id: source.id, database, question: question.trim() });
      setResult(res);
      setDatasetName(question.trim().slice(0, 60));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to answer the question.");
    }
  };

  const handleSaveAsDataset = async () => {
    if (!result || !database) return;
    try {
      const payload = isMongoQuery(result.query)
        ? {
            database,
            table: result.query.collection,
            filter: result.query.filter,
            sort: result.query.sort,
            fields: result.query.fields ?? undefined,
            dataset_name: datasetName || undefined,
          }
        : {
            database,
            custom_sql: result.query,
            dataset_name: datasetName || undefined,
          };
      await importMutation.mutateAsync({ dataSourceId: source.id, payload });
      toast.success("Saved as a new dataset");
      setOpen(false);
      reset();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to save as dataset");
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : (setOpen(false), reset()))}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <MessageCircleQuestion className="h-3.5 w-3.5 mr-1" />
          Ask
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Ask "{source.name}" a question</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-1">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="sm:col-span-1 space-y-1.5">
              <Label className="text-xs">Database</Label>
              <Select value={database ?? undefined} onValueChange={setDatabase} disabled={databasesQuery.isLoading}>
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
            <div className="sm:col-span-2 space-y-1.5">
              <Label className="text-xs">Question</Label>
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. What are the top 5 customers by revenue?"
              />
            </div>
          </div>

          <Button onClick={handleAsk} disabled={!database || !question.trim() || askMutation.isPending}>
            {askMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin mr-1.5" />
            ) : (
              <MessageCircleQuestion className="h-4 w-4 mr-1.5" />
            )}
            Ask
          </Button>

          {error && (
            <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              <XCircle className="h-4 w-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="space-y-3">
              <div>
                <Label className="text-xs text-muted-foreground">Generated query (read-only)</Label>
                <pre className="mt-1 rounded-md border bg-muted/50 p-3 text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                  {typeof result.query === "string" ? result.query : JSON.stringify(result.query, null, 2)}
                </pre>
              </div>

              <div className="rounded-lg border overflow-hidden">
                <div className="flex items-center gap-2 px-3 py-2 bg-muted text-xs text-muted-foreground">
                  <Database className="h-3.5 w-3.5" />
                  {result.row_count_returned} row(s) returned
                  {result.truncated ? " (truncated)" : ""}
                </div>
                <div className="overflow-x-auto max-h-64">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50 sticky top-0">
                      <tr>
                        {result.columns.map((c) => (
                          <th key={c} className="text-left p-2 font-medium whitespace-nowrap">
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.rows.map((r, i) => (
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
              </div>

              {result.explanation && (
                <div className="rounded-md border p-3 text-sm text-muted-foreground whitespace-pre-wrap">
                  {result.explanation}
                </div>
              )}

              <div className="flex items-end gap-2">
                <div className="flex-1 space-y-1.5">
                  <Label className="text-xs">Dataset name</Label>
                  <Input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} />
                </div>
                <Button onClick={handleSaveAsDataset} disabled={importMutation.isPending} variant="outline">
                  {importMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1.5" />
                  ) : (
                    <Save className="h-4 w-4 mr-1.5" />
                  )}
                  Save as dataset
                </Button>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function DataSourceCard({ source }: { source: DataSourceSummary }) {
  const testMutation = useTestSavedDataSource();
  const deleteMutation = useDeleteDataSource();
  const [lastTest, setLastTest] = useState<{ ok: boolean; message: string } | null>(null);

  const handleTest = async () => {
    try {
      const result = await testMutation.mutateAsync(source.id);
      setLastTest({ ok: true, message: result.server_version ?? "Connected" });
      toast.success("Connection is healthy");
    } catch (err) {
      setLastTest({ ok: false, message: err instanceof ApiError ? err.message : "Test failed" });
      toast.error("Connection test failed");
    }
  };

  const handleDelete = async () => {
    try {
      await deleteMutation.mutateAsync(source.id);
      toast.success("Data source disconnected");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to disconnect");
    }
  };

  const statusOk = lastTest ? lastTest.ok : source.last_test_status === "ok";
  const statusKnown = lastTest !== null || source.last_test_status !== null;

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start gap-3">
          <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-indigo-500 to-fuchsia-500 flex items-center justify-center text-white flex-shrink-0">
            <EngineIcon engine={source.engine} className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="font-medium truncate">{source.name}</p>
            <p className="text-xs text-muted-foreground truncate">
              {ENGINE_LABEL[source.engine]}
              {source.host ? ` · ${source.host}${source.port ? `:${source.port}` : ""}` : ""} · {source.database}
            </p>
            {source.username && <p className="text-xs text-muted-foreground truncate">User: {source.username}</p>}
          </div>
        </div>

        <div className="flex items-center justify-between">
          {statusKnown ? (
            <Badge variant={statusOk ? "secondary" : "destructive"} className="gap-1">
              {statusOk ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
              {statusOk ? "Connected" : "Failed"}
            </Badge>
          ) : (
            <Badge variant="outline">Not tested</Badge>
          )}
        </div>

        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="flex-1" onClick={handleTest} disabled={testMutation.isPending}>
            {testMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" /> : <Plug className="h-3.5 w-3.5 mr-1" />}
            Test
          </Button>
          <AskQuestionDialog source={source} />
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button size="sm" variant="outline" className="text-destructive hover:text-destructive">
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Disconnect "{source.name}"?</AlertDialogTitle>
                <AlertDialogDescription>
                  This removes the saved connection. Datasets and models already imported from it are kept.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                  Disconnect
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </CardContent>
    </Card>
  );
}

function DataSourcesPage() {
  const { data, isLoading, isError, refetch } = useDataSourcesList();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Data Sources</h1>
          <p className="text-sm text-muted-foreground">
            Connect directly to MySQL, PostgreSQL, or MongoDB and analyze it like any other dataset.
          </p>
        </div>
        <AddConnectionDialog onCreated={() => refetch()} />
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-36 w-full" />
          ))}
        </div>
      ) : isError ? (
        <Card>
          <CardContent className="p-6 text-sm text-destructive">Couldn't load your data sources.</CardContent>
        </Card>
      ) : !data || data.length === 0 ? (
        <Card>
          <CardContent className="p-10 text-center">
            <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
              <Plug className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="font-medium">No data sources connected</p>
            <p className="text-sm text-muted-foreground mt-1">
              Connect a MySQL, PostgreSQL, or MongoDB database to analyze its data directly.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((s) => (
            <DataSourceCard key={s.id} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}
