import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  ArrowLeft,
  PlusCircle,
  Loader2,
  Trash2,
  AlertTriangle,
  Sparkles,
  Ban,
  Pencil,
} from "lucide-react";
import {
  useBaasProject,
  useBaasKeys,
  useCreateBaasKey,
  useRevokeBaasKey,
  useBaasTables,
  useProposeBaasSchema,
  useCreateBaasTable,
  useBaasRecords,
  useCreateBaasRecord,
  useUpdateBaasRecord,
  useDeleteBaasRecord,
} from "@/hooks/use-baas";
import { CopyableSecret } from "@/components/copyable-secret";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import type {
  BaasApiKeyCreateResult,
  BaasColumn,
  BaasColumnType,
  BaasCredentials,
  BaasRecord,
  BaasTable,
} from "@/lib/api-types";

export const Route = createFileRoute("/baas/$projectId")({
  beforeLoad: requireAuth,
  component: ProjectDetailPage,
});

// ── Keys tab ─────────────────────────────────────────────────────────────
function CreateKeyDialog({ projectId, onCreated }: { projectId: string; onCreated: () => void }) {
  const [open, setOpen] = useState(false);
  const [label, setLabel] = useState("");
  const [result, setResult] = useState<BaasApiKeyCreateResult | null>(null);
  const createMutation = useCreateBaasKey();

  const resetAndClose = () => {
    setLabel("");
    setResult(null);
    setOpen(false);
  };

  const handleCreate = async () => {
    try {
      const res = await createMutation.mutateAsync({ projectId, label: label.trim() || undefined });
      setResult(res);
      onCreated();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to create key");
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : result ? undefined : resetAndClose())}>
      <DialogTrigger asChild>
        <Button size="sm">
          <PlusCircle className="h-3.5 w-3.5 mr-1.5" /> New key
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{result ? "Key created" : "New API key"}</DialogTitle>
        </DialogHeader>
        {!result ? (
          <div className="space-y-1.5">
            <Label>Label (optional)</Label>
            <Input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. Production" />
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
              <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <span>The secret key is shown only this once. Copy both keys now.</span>
            </div>
            <CopyableSecret label="Public key" value={result.key.public_key} />
            <CopyableSecret label="Secret key" value={result.secret_key} />
          </div>
        )}
        <DialogFooter>
          {!result ? (
            <Button onClick={handleCreate} disabled={createMutation.isPending}>
              {createMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
              Create key
            </Button>
          ) : (
            <Button onClick={resetAndClose}>I've saved it — done</Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function KeysTab({ projectId }: { projectId: string }) {
  const { data: keys, isLoading, refetch } = useBaasKeys(projectId);
  const revokeMutation = useRevokeBaasKey();

  const handleRevoke = async (keyId: string) => {
    try {
      await revokeMutation.mutateAsync({ projectId, keyId });
      toast.success("Key revoked");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to revoke key");
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">API Keys</CardTitle>
        <CreateKeyDialog projectId={projectId} onCreated={() => refetch()} />
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-24 w-full" />
        ) : !keys || keys.length === 0 ? (
          <p className="text-sm text-muted-foreground">No keys yet.</p>
        ) : (
          <div className="space-y-2">
            {keys.map((k) => (
              <div key={k.id} className="flex items-center justify-between gap-3 rounded-md border p-3">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{k.label || "Untitled key"}</p>
                  <code className="text-xs text-muted-foreground break-all">{k.public_key}</code>
                </div>
                {k.disabled ? (
                  <Badge variant="destructive" className="gap-1">
                    <Ban className="h-3 w-3" /> Revoked
                  </Badge>
                ) : (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleRevoke(k.id)}
                    disabled={revokeMutation.isPending}
                  >
                    Revoke
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Tables tab ───────────────────────────────────────────────────────────
const COLUMN_TYPES: BaasColumnType[] = ["string", "number", "boolean", "date"];

function CreateTableDialog({ projectId, onCreated }: { projectId: string; onCreated: () => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [columns, setColumns] = useState<BaasColumn[]>([]);
  const [proposeError, setProposeError] = useState<string | null>(null);

  const proposeMutation = useProposeBaasSchema();
  const createMutation = useCreateBaasTable();

  const resetAndClose = () => {
    setName("");
    setDescription("");
    setColumns([]);
    setProposeError(null);
    setOpen(false);
  };

  const handlePropose = async () => {
    if (!description.trim()) {
      toast.error("Describe the table first.");
      return;
    }
    setProposeError(null);
    try {
      const res = await proposeMutation.mutateAsync({ projectId, description: description.trim() });
      setColumns(res.columns);
    } catch (err) {
      setProposeError(
        err instanceof ApiError ? err.message : "AI proposal failed. Add columns manually below.",
      );
    }
  };

  const addColumn = () => setColumns((c) => [...c, { name: "", type: "string" }]);
  const removeColumn = (i: number) => setColumns((c) => c.filter((_, idx) => idx !== i));
  const updateColumn = (i: number, patch: Partial<BaasColumn>) =>
    setColumns((c) => c.map((col, idx) => (idx === i ? { ...col, ...patch } : col)));

  const handleCreate = async () => {
    if (!name.trim()) {
      toast.error("Give the table a name.");
      return;
    }
    const cleanColumns = columns.filter((c) => c.name.trim());
    if (cleanColumns.length === 0) {
      toast.error("Add at least one column.");
      return;
    }
    try {
      await createMutation.mutateAsync({ projectId, name: name.trim(), columns: cleanColumns });
      toast.success("Table created");
      onCreated();
      resetAndClose();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to create table");
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : resetAndClose())}>
      <DialogTrigger asChild>
        <Button size="sm">
          <PlusCircle className="h-3.5 w-3.5 mr-1.5" /> New table
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>New table</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 max-h-[65vh] overflow-y-auto pr-1">
          <div className="space-y-1.5">
            <Label>Table name</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="products" />
          </div>
          <div className="space-y-1.5">
            <Label>Describe the table (optional — AI will propose columns)</Label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="A products table with name, price, category and stock"
              className="h-16"
            />
            <Button type="button" size="sm" variant="outline" onClick={handlePropose} disabled={proposeMutation.isPending}>
              {proposeMutation.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
              ) : (
                <Sparkles className="h-3.5 w-3.5 mr-1.5" />
              )}
              Propose with AI
            </Button>
            {proposeError && <p className="text-xs text-destructive">{proposeError}</p>}
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Columns</Label>
              <Button type="button" size="sm" variant="ghost" onClick={addColumn}>
                <PlusCircle className="h-3.5 w-3.5 mr-1" /> Add column
              </Button>
            </div>
            {columns.length === 0 ? (
              <p className="text-xs text-muted-foreground">No columns yet — add one manually or propose with AI.</p>
            ) : (
              <div className="space-y-2">
                {columns.map((col, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <Input
                      value={col.name}
                      onChange={(e) => updateColumn(i, { name: e.target.value })}
                      placeholder="column_name"
                      className="h-8"
                    />
                    <Select value={col.type} onValueChange={(v) => updateColumn(i, { type: v as BaasColumnType })}>
                      <SelectTrigger className="h-8 w-28">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {COLUMN_TYPES.map((t) => (
                          <SelectItem key={t} value={t}>
                            {t}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button type="button" size="icon" variant="ghost" onClick={() => removeColumn(i)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        <DialogFooter>
          <Button onClick={handleCreate} disabled={createMutation.isPending}>
            {createMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
            Create table
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function TablesTab({ projectId }: { projectId: string }) {
  const { data: tables, isLoading, refetch } = useBaasTables(projectId);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">Tables</CardTitle>
        <CreateTableDialog projectId={projectId} onCreated={() => refetch()} />
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-24 w-full" />
        ) : !tables || tables.length === 0 ? (
          <p className="text-sm text-muted-foreground">No tables yet.</p>
        ) : (
          <div className="space-y-3">
            {tables.map((t) => (
              <div key={t.id} className="rounded-md border p-3 space-y-2">
                <p className="text-sm font-medium">{t.name}</p>
                <div className="flex flex-wrap gap-1.5">
                  {t.columns.map((c) => (
                    <Badge key={c.name} variant="outline" className="text-xs font-normal">
                      {c.name} <span className="text-muted-foreground ml-1">{c.type}</span>
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Data tab ─────────────────────────────────────────────────────────────
function columnInput(col: BaasColumn, value: unknown, onChange: (v: unknown) => void) {
  if (col.type === "boolean") {
    return (
      <Checkbox checked={Boolean(value)} onCheckedChange={(v) => onChange(v === true)} />
    );
  }
  if (col.type === "number") {
    return (
      <Input
        type="number"
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
    );
  }
  if (col.type === "date") {
    return (
      <Input
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value)}
        placeholder="YYYY-MM-DD"
      />
    );
  }
  return (
    <Input
      value={value === undefined || value === null ? "" : String(value)}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function RecordFormDialog({
  table,
  creds,
  record,
  trigger,
}: {
  table: BaasTable;
  creds: BaasCredentials;
  record?: BaasRecord;
  trigger: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const createMutation = useCreateBaasRecord();
  const updateMutation = useUpdateBaasRecord();

  useEffect(() => {
    if (open) {
      const initial: Record<string, unknown> = {};
      for (const col of table.columns) initial[col.name] = record ? record[col.name] : undefined;
      setValues(initial);
    }
  }, [open, table, record]);

  const isPending = createMutation.isPending || updateMutation.isPending;

  const handleSubmit = async () => {
    try {
      if (record) {
        await updateMutation.mutateAsync({ creds, tableName: table.name, recordId: record.id, data: values });
        toast.success("Record updated");
      } else {
        await createMutation.mutateAsync({ creds, tableName: table.name, data: values });
        toast.success("Record created");
      }
      setOpen(false);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to save record");
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{record ? "Edit record" : "New record"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
          {table.columns.map((col) => (
            <div key={col.name} className="space-y-1.5">
              <Label className="text-xs">
                {col.name} <span className="text-muted-foreground">({col.type})</span>
              </Label>
              {columnInput(col, values[col.name], (v) => setValues((old) => ({ ...old, [col.name]: v })))}
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button onClick={handleSubmit} disabled={isPending}>
            {isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
            {record ? "Save changes" : "Create record"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function DataTab({ tables }: { tables: BaasTable[] | undefined }) {
  const [tableName, setTableName] = useState<string | null>(null);
  const [publicKey, setPublicKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [unlocked, setUnlocked] = useState(false);

  // Secret key is intentionally never persisted (no localStorage/sessionStorage) —
  // cleared whenever this component unmounts (e.g. navigating to another tab's
  // route or away from the page).
  useEffect(() => {
    return () => setSecretKey("");
  }, []);

  const selectedTable = tables?.find((t) => t.name === tableName) ?? null;
  const creds: BaasCredentials | null = unlocked && publicKey && secretKey ? { publicKey, secretKey } : null;

  const recordsQuery = useBaasRecords(creds, tableName);
  const deleteMutation = useDeleteBaasRecord();

  const handleDelete = async (recordId: string) => {
    if (!creds || !tableName) return;
    try {
      await deleteMutation.mutateAsync({ creds, tableName, recordId });
      toast.success("Record deleted");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to delete record");
    }
  };

  if (!tables || tables.length === 0) {
    return (
      <Card>
        <CardContent className="p-10 text-center text-sm text-muted-foreground">
          Create a table first to browse its data.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Data</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label className="text-xs">Table</Label>
            <Select value={tableName ?? undefined} onValueChange={(v) => { setTableName(v); setUnlocked(false); }}>
              <SelectTrigger>
                <SelectValue placeholder="Select a table" />
              </SelectTrigger>
              <SelectContent>
                {tables.map((t) => (
                  <SelectItem key={t.id} value={t.name}>
                    {t.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Public key</Label>
            <Input value={publicKey} onChange={(e) => setPublicKey(e.target.value)} placeholder="pk_live_…" />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Secret key</Label>
            <Input
              type="password"
              value={secretKey}
              onChange={(e) => setSecretKey(e.target.value)}
              placeholder="sk_live_…"
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Keys are kept only in this browser tab for this session — never saved, cleared when you leave this
          page.
        </p>
        <Button
          size="sm"
          variant="outline"
          disabled={!tableName || !publicKey.trim() || !secretKey.trim()}
          onClick={() => setUnlocked(true)}
        >
          Unlock data browser
        </Button>

        {creds && selectedTable && (
          <div className="space-y-3 border-t pt-4">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium">{selectedTable.name}</p>
              <RecordFormDialog
                table={selectedTable}
                creds={creds}
                trigger={
                  <Button size="sm">
                    <PlusCircle className="h-3.5 w-3.5 mr-1.5" /> New record
                  </Button>
                }
              />
            </div>

            {recordsQuery.isLoading ? (
              <Skeleton className="h-32 w-full" />
            ) : recordsQuery.isError ? (
              <p className="text-sm text-destructive">
                Couldn't load records — check your public/secret key are correct for this project.
              </p>
            ) : !recordsQuery.data || recordsQuery.data.records.length === 0 ? (
              <p className="text-sm text-muted-foreground">No records yet.</p>
            ) : (
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      {selectedTable.columns.map((c) => (
                        <th key={c.name} className="text-left p-2 font-medium whitespace-nowrap">
                          {c.name}
                        </th>
                      ))}
                      <th className="p-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {recordsQuery.data.records.map((r) => (
                      <tr key={r.id} className="border-t">
                        {selectedTable.columns.map((c) => (
                          <td key={c.name} className="p-2 whitespace-nowrap">
                            {r[c.name] === null || r[c.name] === undefined ? (
                              <span className="text-muted-foreground">—</span>
                            ) : (
                              String(r[c.name])
                            )}
                          </td>
                        ))}
                        <td className="p-2 whitespace-nowrap text-right">
                          <div className="flex justify-end gap-1">
                            <RecordFormDialog
                              table={selectedTable}
                              creds={creds}
                              record={r}
                              trigger={
                                <Button size="icon" variant="ghost">
                                  <Pencil className="h-3.5 w-3.5" />
                                </Button>
                              }
                            />
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button size="icon" variant="ghost">
                                  <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>Delete this record?</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    This permanently deletes the record. This cannot be undone.
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction
                                    onClick={() => handleDelete(r.id)}
                                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                  >
                                    Delete
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ProjectDetailPage() {
  const { projectId } = Route.useParams();
  const projectQuery = useBaasProject(projectId);
  const tablesQuery = useBaasTables(projectId);

  if (projectQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (projectQuery.isError || !projectQuery.data) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <p>This project couldn't be found.</p>
          <Button asChild className="mt-4">
            <Link to="/baas">Back to projects</Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <Button asChild variant="ghost" size="sm" className="mb-2 -ml-2">
          <Link to="/baas">
            <ArrowLeft className="h-4 w-4 mr-1" /> Back to projects
          </Link>
        </Button>
        <h1 className="text-2xl font-bold">{projectQuery.data.name}</h1>
        <p className="text-sm text-muted-foreground">
          Created {new Date(projectQuery.data.created_at).toLocaleDateString()}
        </p>
      </div>

      <Tabs defaultValue="keys">
        <TabsList>
          <TabsTrigger value="keys">Keys</TabsTrigger>
          <TabsTrigger value="tables">Tables</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
        </TabsList>
        <TabsContent value="keys">
          <KeysTab projectId={projectId} />
        </TabsContent>
        <TabsContent value="tables">
          <TablesTab projectId={projectId} />
        </TabsContent>
        <TabsContent value="data">
          <DataTab tables={tablesQuery.data} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
