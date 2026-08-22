import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
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
  Boxes,
  Search,
  Download,
  Heart,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ShieldQuestion,
  PlayCircle,
} from "lucide-react";
import { useHfSearch, useHfCompatibilityCheck, useRunHfModel } from "@/hooks/use-hf-models";
import { useDatasetsList, usePreviewDataset } from "@/hooks/use-datasets";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import type {
  HFCompatibilityResult,
  HFExecutionMode,
  HFModelSummary,
  HFRunModelResult,
} from "@/lib/api-types";

function isNumericDtype(dtype: string) {
  const d = dtype.toLowerCase();
  return d.includes("int") || d.includes("float");
}

export const Route = createFileRoute("/hf-models")({
  beforeLoad: requireAuth,
  component: HfModelsPage,
});

const EXECUTION_MODE_META: Record<
  HFExecutionMode,
  { label: string; icon: typeof CheckCircle2; className: string }
> = {
  inference_only: {
    label: "Ready to run",
    icon: CheckCircle2,
    className: "border-green-600/30 bg-green-600/10 text-green-700 dark:text-green-400",
  },
  fine_tune_required: {
    label: "Fine-tuning required",
    icon: AlertTriangle,
    className: "border-amber-600/30 bg-amber-600/10 text-amber-700 dark:text-amber-400",
  },
  unsupported: {
    label: "Not compatible",
    icon: XCircle,
    className: "border-destructive/30 bg-destructive/10 text-destructive",
  },
};

function CompatibilityDialog({ model }: { model: HFModelSummary }) {
  const [open, setOpen] = useState(false);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [result, setResult] = useState<HFCompatibilityResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [features, setFeatures] = useState<string[]>([]);
  const [runResult, setRunResult] = useState<HFRunModelResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const { data: datasets, isLoading: datasetsLoading } = useDatasetsList();
  const checkMutation = useHfCompatibilityCheck();
  const runMutation = useRunHfModel();
  const previewQuery = usePreviewDataset(
    result?.execution_mode === "inference_only" ? datasetId : null,
  );

  const numericColumns =
    previewQuery.data?.columns.filter((c) => isNumericDtype(previewQuery.data!.data_types[c] ?? "")) ?? [];

  useEffect(() => {
    if (numericColumns.length > 0 && !target) {
      const lastCol = numericColumns[numericColumns.length - 1];
      setTarget(lastCol);
      setFeatures(numericColumns.filter((c) => c !== lastCol));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewQuery.data]);

  const reset = () => {
    setDatasetId(null);
    setResult(null);
    setError(null);
    setTarget(null);
    setFeatures([]);
    setRunResult(null);
    setRunError(null);
  };

  const handleCheck = async () => {
    if (!datasetId) return;
    setError(null);
    setRunResult(null);
    setRunError(null);
    setTarget(null);
    setFeatures([]);
    try {
      const res = await checkMutation.mutateAsync({ modelId: model.model_id, datasetId });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Compatibility check failed.");
    }
  };

  const toggleFeature = (col: string) => {
    setFeatures((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  };

  const handleRun = async () => {
    if (!datasetId || !target || features.length === 0) return;
    setRunError(null);
    try {
      const res = await runMutation.mutateAsync({
        modelId: model.model_id,
        payload: { dataset_id: datasetId, features, target },
      });
      setRunResult(res);
    } catch (err) {
      setRunError(err instanceof ApiError ? err.message : "Model execution failed.");
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : (setOpen(false), reset()))}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <ShieldQuestion className="h-3.5 w-3.5 mr-1.5" />
          Check compatibility
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="break-all">{model.model_id}</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <p className="text-sm text-muted-foreground">
              Check this model against one of your datasets. This looks at the model's real Hub
              metadata (framework, task, files) — nothing is executed yet.
            </p>
          </div>

          {datasetsLoading ? (
            <Skeleton className="h-9 w-full" />
          ) : !datasets || datasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Upload a dataset first to check compatibility against it.
            </p>
          ) : (
            <Select value={datasetId ?? undefined} onValueChange={(v) => { setDatasetId(v); setResult(null); setError(null); }}>
              <SelectTrigger>
                <SelectValue placeholder="Select a dataset" />
              </SelectTrigger>
              <SelectContent>
                {datasets.map((d) => (
                  <SelectItem key={d.dataset_id} value={d.dataset_id}>
                    {d.name} ({d.rows} rows)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          {error && (
            <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              <XCircle className="h-4 w-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="space-y-2">
              {(() => {
                const meta = EXECUTION_MODE_META[result.execution_mode];
                const Icon = meta.icon;
                return (
                  <div className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium ${meta.className}`}>
                    <Icon className="h-4 w-4 flex-shrink-0" />
                    <span>{meta.label}</span>
                  </div>
                );
              })()}
              <ul className="space-y-1 text-sm text-muted-foreground list-disc pl-5">
                {result.reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          {result?.execution_mode === "inference_only" && !runResult && (
            <div className="space-y-3 border-t pt-4">
              {previewQuery.isLoading ? (
                <Skeleton className="h-24 w-full" />
              ) : numericColumns.length < 2 ? (
                <p className="text-sm text-muted-foreground">
                  This dataset needs at least 2 numeric columns (1 target + 1 feature) to run
                  inference.
                </p>
              ) : (
                <>
                  <div className="space-y-1.5">
                    <Label>Target column</Label>
                    <Select
                      value={target ?? undefined}
                      onValueChange={(v) => {
                        setTarget(v);
                        setFeatures((prev) => prev.filter((c) => c !== v));
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select target" />
                      </SelectTrigger>
                      <SelectContent>
                        {numericColumns.map((c) => (
                          <SelectItem key={c} value={c}>
                            {c}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <Label>Feature columns</Label>
                    <div className="max-h-32 overflow-y-auto space-y-1.5 rounded-md border p-2">
                      {numericColumns
                        .filter((c) => c !== target)
                        .map((c) => (
                          <div key={c} className="flex items-center gap-2">
                            <Checkbox
                              id={`feat-${c}`}
                              checked={features.includes(c)}
                              onCheckedChange={() => toggleFeature(c)}
                            />
                            <Label htmlFor={`feat-${c}`} className="font-normal text-sm">
                              {c}
                            </Label>
                          </div>
                        ))}
                    </div>
                  </div>

                  {runError && (
                    <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                      <XCircle className="h-4 w-4 flex-shrink-0" />
                      <span>{runError}</span>
                    </div>
                  )}

                  <Button
                    onClick={handleRun}
                    disabled={!target || features.length === 0 || runMutation.isPending}
                    className="w-full"
                  >
                    {runMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-1.5" />
                    ) : (
                      <PlayCircle className="h-4 w-4 mr-1.5" />
                    )}
                    Run inference
                  </Button>
                </>
              )}
            </div>
          )}

          {runResult && (
            <div className="space-y-3 border-t pt-4">
              <div className="flex items-center gap-2 rounded-md border border-green-600/30 bg-green-600/10 px-3 py-2 text-sm font-medium text-green-700 dark:text-green-400">
                <CheckCircle2 className="h-4 w-4 flex-shrink-0" />
                <span>Model executed — real zero-shot metrics on your data</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center text-sm">
                <div className="rounded-md border p-2">
                  <p className="text-xs text-muted-foreground">R²</p>
                  <p className="font-semibold">{runResult.metrics.R2.toFixed(3)}</p>
                </div>
                <div className="rounded-md border p-2">
                  <p className="text-xs text-muted-foreground">RMSE</p>
                  <p className="font-semibold">{runResult.metrics.RMSE.toFixed(3)}</p>
                </div>
                <div className="rounded-md border p-2">
                  <p className="text-xs text-muted-foreground">MAE</p>
                  <p className="font-semibold">{runResult.metrics.MAE.toFixed(3)}</p>
                </div>
              </div>
              {runResult.version_warning && (
                <p className="text-xs text-muted-foreground">
                  Note: {runResult.version_warning}
                </p>
              )}
              <Button asChild variant="outline" className="w-full">
                <Link to="/models/$modelId" params={{ modelId: runResult.model_id }}>
                  View model details
                </Link>
              </Button>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button onClick={handleCheck} disabled={!datasetId || checkMutation.isPending}>
            {checkMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
            Run check
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ModelCard({ model }: { model: HFModelSummary }) {
  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start gap-3">
          <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-indigo-500 to-fuchsia-500 flex items-center justify-center text-white flex-shrink-0">
            <Boxes className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="font-medium truncate" title={model.model_id}>
              {model.model_id}
            </p>
            <p className="text-xs text-muted-foreground truncate">
              {model.author ?? "Unknown author"}
              {model.library_name ? ` · ${model.library_name}` : ""}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {model.pipeline_tag && <Badge variant="secondary">{model.pipeline_tag}</Badge>}
          {model.tags.slice(0, 3).map((t) => (
            <Badge key={t} variant="outline" className="text-xs">
              {t}
            </Badge>
          ))}
        </div>

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Download className="h-3.5 w-3.5" /> {model.downloads?.toLocaleString() ?? 0}
          </span>
          <span className="flex items-center gap-1">
            <Heart className="h-3.5 w-3.5" /> {model.likes?.toLocaleString() ?? 0}
          </span>
        </div>

        <CompatibilityDialog model={model} />
      </CardContent>
    </Card>
  );
}

function HfModelsPage() {
  const [queryInput, setQueryInput] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");

  const { data, isLoading, isError, isFetched } = useHfSearch(submittedQuery);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittedQuery(queryInput.trim());
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Hugging Face Models</h1>
        <p className="text-sm text-muted-foreground">
          Search the Hugging Face Hub and check whether a model is actually usable for tabular
          regression on your data. Most results will need fine-tuning or aren't compatible —
          that's expected, not an error.
        </p>
      </div>

      <form onSubmit={handleSearch} className="flex gap-2">
        <Input
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          placeholder="Search models, e.g. tabular, sklearn, house price…"
          className="max-w-md"
        />
        <Button type="submit" disabled={!queryInput.trim()}>
          <Search className="h-4 w-4 mr-1.5" />
          Search
        </Button>
      </form>

      {!submittedQuery && !isFetched ? (
        <Card>
          <CardContent className="p-10 text-center">
            <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
              <Boxes className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="font-medium">Search the Hugging Face Hub</p>
            <p className="text-sm text-muted-foreground mt-1">
              Try a single keyword — the Hub's search matches model names and tags, not full
              phrases.
            </p>
          </CardContent>
        </Card>
      ) : isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full" />
          ))}
        </div>
      ) : isError ? (
        <Card>
          <CardContent className="p-6 text-sm text-destructive">
            Couldn't reach the Hugging Face Hub. Please try again.
          </CardContent>
        </Card>
      ) : !data || data.length === 0 ? (
        <Card>
          <CardContent className="p-10 text-center">
            <p className="font-medium">No models found</p>
            <p className="text-sm text-muted-foreground mt-1">Try a different search term.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((m) => (
            <ModelCard key={m.model_id} model={m} />
          ))}
        </div>
      )}
    </div>
  );
}
