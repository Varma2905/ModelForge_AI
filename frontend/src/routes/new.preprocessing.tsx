import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { usePreprocess } from "@/hooks/use-preprocessing";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { toast } from "sonner";

export const Route = createFileRoute("/new/preprocessing")({
  beforeLoad: requireAuth,
  component: PreprocessPage,
});

function PreprocessPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const p = state.preprocessing;
  const set = (patch: Partial<typeof p>) => update({ preprocessing: { ...p, ...patch } });
  const preprocess = usePreprocess();

  if (!state.datasetId) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Please upload a dataset first.</p>
            <Button asChild className="mt-4">
              <Link to="/new/upload">Go to upload</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const handleContinue = async () => {
    try {
      const result = await preprocess.mutateAsync({
        dataset_id: state.datasetId!,
        config: {
          missing: p.missing as "remove" | "mean" | "median" | "mode" | "none",
          dedupe: p.dedupe,
          outlier: p.outlier as "iqr" | "zscore" | "none",
          scaling: p.scaling as "standard" | "minmax" | "none",
        },
      });
      update({
        preprocessedDatasetId: result.preprocessed_dataset_id,
        preprocessingSummary: result.summary,
      });
      toast.success("Preprocessing complete");
      navigate({ to: "/new/split" });
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Preprocessing failed");
    }
  };

  return (
    <div>
      <WizardSteps />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Missing Values</CardTitle>
          </CardHeader>
          <CardContent>
            <Select value={p.missing} onValueChange={(v) => set({ missing: v })}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="remove">Remove rows</SelectItem>
                <SelectItem value="mean">Mean replacement</SelectItem>
                <SelectItem value="median">Median replacement</SelectItem>
                <SelectItem value="mode">Mode replacement</SelectItem>
              </SelectContent>
            </Select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Duplicate Data</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center gap-3">
            <Switch checked={p.dedupe} onCheckedChange={(v) => set({ dedupe: v })} />
            <Label>Remove duplicate rows</Label>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Outlier Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <RadioGroup value={p.outlier} onValueChange={(v) => set({ outlier: v })}>
              {[
                ["iqr", "IQR Method"],
                ["zscore", "Z-score Method"],
                ["none", "None"],
              ].map(([v, l]) => (
                <label key={v} className="flex items-center gap-2 p-2 rounded hover:bg-accent">
                  <RadioGroupItem value={v} id={v} />
                  <Label htmlFor={v}>{l}</Label>
                </label>
              ))}
            </RadioGroup>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Feature Scaling</CardTitle>
          </CardHeader>
          <CardContent>
            <RadioGroup value={p.scaling} onValueChange={(v) => set({ scaling: v })}>
              {[
                ["standard", "StandardScaler"],
                ["minmax", "MinMaxScaler"],
                ["none", "No Scaling"],
              ].map(([v, l]) => (
                <label key={v} className="flex items-center gap-2 p-2 rounded hover:bg-accent">
                  <RadioGroupItem value={v} id={`s-${v}`} />
                  <Label htmlFor={`s-${v}`}>{l}</Label>
                </label>
              ))}
            </RadioGroup>
            <p className="mt-2 text-xs text-muted-foreground">
              Scaling is fit on the training split only, at train time — this avoids leaking
              test-set statistics into the model.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Preprocessing Summary</CardTitle>
        </CardHeader>
        <CardContent>
          {state.preprocessingSummary ? (
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>
                Rows before:{" "}
                <span className="text-foreground font-medium">
                  {state.preprocessingSummary.initial_rows}
                </span>
              </li>
              <li>
                Rows after:{" "}
                <span className="text-foreground font-medium">
                  {state.preprocessingSummary.final_rows}
                </span>
              </li>
              <li>
                Missing values imputed:{" "}
                <span className="text-foreground font-medium">
                  {state.preprocessingSummary.missing_imputed}
                </span>
              </li>
              <li>
                Duplicates removed:{" "}
                <span className="text-foreground font-medium">
                  {state.preprocessingSummary.duplicates_removed}
                </span>
              </li>
              <li>
                Outliers removed:{" "}
                <span className="text-foreground font-medium">
                  {state.preprocessingSummary.outliers_removed}
                </span>
              </li>
            </ul>
          ) : (
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>
                Missing values: <span className="text-foreground font-medium">{p.missing}</span>
              </li>
              <li>
                Deduplicate:{" "}
                <span className="text-foreground font-medium">{p.dedupe ? "Yes" : "No"}</span>
              </li>
              <li>
                Outliers: <span className="text-foreground font-medium">{p.outlier}</span>
              </li>
              <li>
                Scaling: <span className="text-foreground font-medium">{p.scaling}</span>
              </li>
            </ul>
          )}
          <div className="flex justify-end mt-4">
            <Button onClick={handleContinue} disabled={preprocess.isPending}>
              {preprocess.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Continue
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
