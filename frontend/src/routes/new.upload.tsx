import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Upload, Plus, Trash2, Loader2 } from "lucide-react";
import { WizardSteps } from "@/components/wizard-steps";
import { DatabaseImportPanel } from "@/components/database-import-panel";
import { useAnalysis, type Dataset } from "@/lib/analysis-store";
import { useCreateDataset, usePreviewDataset, useUploadDataset } from "@/hooks/use-datasets";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import type { ImportDatasetResult } from "@/lib/api-types";
import { toast } from "sonner";

export const Route = createFileRoute("/new/upload")({
  beforeLoad: requireAuth,
  component: UploadPage,
});

function UploadPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const [dragOver, setDragOver] = useState(false);
  const [manual, setManual] = useState<Dataset>({
    name: "manual_dataset",
    columns: ["Area", "Bedrooms", "Age", "Price"],
    rows: [
      [1500, 3, 10, 5000000],
      [2400, 4, 5, 8200000],
    ],
  });

  const uploadMutation = useUploadDataset();
  const createMutation = useCreateDataset();
  const isSaving = uploadMutation.isPending || createMutation.isPending;

  const preview = usePreviewDataset(state.datasetId);

  const applyNewDataset = (
    datasetId: string,
    name: string,
    missingValues: number,
    dataTypes: Record<string, string>,
  ) => {
    // A new dataset invalidates everything downstream in the wizard.
    update({
      datasetId,
      preprocessedDatasetId: null,
      modelId: null,
      dataset: { name, columns: [], rows: [] },
      datasetStats: { missingValues, dataTypes },
      features: [],
      target: null,
      preprocessingSummary: null,
      trained: false,
      results: {},
      statisticalAnalysis: null,
      chartData: null,
      prediction: null,
    });
  };

  const handleFile = async (file: File) => {
    const isSupported = /\.(csv|xlsx|xls)$/i.test(file.name);
    if (!isSupported) {
      toast.error("Only CSV and Excel (.xlsx, .xls) files are supported.");
      return;
    }
    try {
      const result = await uploadMutation.mutateAsync(file);
      applyNewDataset(
        result.dataset_id,
        result.dataset_name,
        result.missing_values,
        result.data_types,
      );
      toast.success("Dataset uploaded");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to upload dataset");
    }
  };

  const saveManualDataset = async () => {
    try {
      const result = await createMutation.mutateAsync(manual);
      applyNewDataset(
        result.dataset_id,
        result.dataset_name,
        result.missing_values,
        result.data_types,
      );
      toast.success("Dataset saved");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to save dataset");
    }
  };

  const handleDatabaseImport = (result: ImportDatasetResult) => {
    applyNewDataset(result.dataset_id, result.dataset_name, result.missing_values, result.data_types);
    toast.success(
      result.truncated
        ? `Dataset imported — row limit applied (${result.rows.toLocaleString()} rows)`
        : "Dataset imported",
    );
  };

  const ds = preview.data;

  return (
    <div>
      <WizardSteps />
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Dataset</h1>
          <p className="text-muted-foreground text-sm">Upload a file or build one from scratch.</p>
        </div>

        <Tabs defaultValue="upload">
          <TabsList>
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="manual">Create manually</TabsTrigger>
            <TabsTrigger value="database">Connect Database</TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <Card>
              <CardContent className="p-6">
                <label
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) handleFile(f);
                  }}
                  className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 cursor-pointer transition ${
                    dragOver ? "border-primary bg-primary/5" : "border-muted-foreground/30"
                  } ${isSaving ? "pointer-events-none opacity-60" : ""}`}
                >
                  {isSaving ? (
                    <Loader2 className="h-10 w-10 text-muted-foreground mb-3 animate-spin" />
                  ) : (
                    <Upload className="h-10 w-10 text-muted-foreground mb-3" />
                  )}
                  <div className="font-medium">
                    {isSaving ? "Uploading dataset…" : "Drop CSV / XLSX here"}
                  </div>
                  <div className="text-xs text-muted-foreground">or click to browse</div>
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    className="hidden"
                    disabled={isSaving}
                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  />
                </label>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="manual">
            <Card>
              <CardHeader>
                <CardTitle>Spreadsheet Editor</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex gap-2 flex-wrap">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      setManual((m) => ({
                        ...m,
                        columns: [...m.columns, `col${m.columns.length + 1}`],
                        rows: m.rows.map((r) => [...r, 0]),
                      }))
                    }
                  >
                    <Plus className="h-3 w-3 mr-1" /> Column
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      setManual((m) => ({
                        ...m,
                        rows: [...m.rows, new Array(m.columns.length).fill(0)],
                      }))
                    }
                  >
                    <Plus className="h-3 w-3 mr-1" /> Row
                  </Button>
                  <Button size="sm" onClick={saveManualDataset} disabled={isSaving}>
                    {isSaving ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : null}
                    Save dataset
                  </Button>
                </div>
                <div className="overflow-x-auto rounded-lg border">
                  <table className="w-full text-sm">
                    <thead className="bg-muted">
                      <tr>
                        {manual.columns.map((c, i) => (
                          <th key={i} className="p-1">
                            <div className="flex items-center gap-1">
                              <Input
                                value={c}
                                onChange={(e) =>
                                  setManual((m) => {
                                    const cols = [...m.columns];
                                    cols[i] = e.target.value;
                                    return { ...m, columns: cols };
                                  })
                                }
                                className="h-8"
                              />
                              <Button
                                size="icon"
                                variant="ghost"
                                onClick={() =>
                                  setManual((m) => ({
                                    ...m,
                                    columns: m.columns.filter((_, x) => x !== i),
                                    rows: m.rows.map((r) => r.filter((_, x) => x !== i)),
                                  }))
                                }
                              >
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {manual.rows.map((r, ri) => (
                        <tr key={ri} className="border-t">
                          {r.map((v, ci) => (
                            <td key={ci} className="p-1">
                              <Input
                                value={String(v)}
                                onChange={(e) =>
                                  setManual((m) => {
                                    const rows = m.rows.map((row) => [...row]);
                                    const n = Number(e.target.value);
                                    rows[ri][ci] = Number.isFinite(n) ? n : e.target.value;
                                    return { ...m, rows };
                                  })
                                }
                                className="h-8"
                              />
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="database">
            <Card>
              <CardHeader>
                <CardTitle>Connect a Database</CardTitle>
              </CardHeader>
              <CardContent>
                <DatabaseImportPanel onImported={handleDatabaseImport} />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {state.datasetId && (
          <Card>
            <CardHeader>
              <CardTitle>Preview — {ds?.name ?? state.dataset?.name}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {preview.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-40 w-full" />
                </div>
              ) : preview.isError ? (
                <p className="text-sm text-destructive">
                  Failed to load dataset preview. Try re-uploading.
                </p>
              ) : ds ? (
                <>
                  <div className="grid gap-3 sm:grid-cols-4">
                    <Stat label="Rows" value={ds.total_rows} />
                    <Stat label="Columns" value={ds.total_columns} />
                    <Stat label="Missing" value={ds.missing_values} />
                    <Stat
                      label="Numeric"
                      value={
                        Object.values(ds.data_types).filter(
                          (t) => t.includes("int") || t.includes("float"),
                        ).length
                      }
                    />
                  </div>
                  <div className="overflow-x-auto rounded-lg border max-h-80">
                    <table className="w-full text-sm">
                      <thead className="bg-muted sticky top-0">
                        <tr>
                          {ds.columns.map((c) => (
                            <th key={c} className="text-left p-2 font-medium">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {ds.rows.slice(0, 20).map((r, i) => (
                          <tr key={i} className="border-t">
                            {r.map((v, j) => (
                              <td key={j} className="p-2">
                                {v === null ? (
                                  <span className="text-muted-foreground">—</span>
                                ) : (
                                  String(v)
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : null}
              <div className="flex justify-end gap-2">
                <Button asChild variant="outline">
                  <Link to="/">Cancel</Link>
                </Button>
                <Button
                  disabled={!ds}
                  onClick={() => {
                    if (ds)
                      update({ dataset: { name: ds.name, columns: ds.columns, rows: ds.rows } });
                    navigate({ to: "/new/variables" });
                  }}
                >
                  Continue to Variable Selection
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}
