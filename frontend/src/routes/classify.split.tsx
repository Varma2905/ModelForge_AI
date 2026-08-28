import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Loader2 } from "lucide-react";
import { WizardSteps, classificationSteps } from "@/components/wizard-steps";
import { useClassificationAnalysis, toBackendSplitConfig } from "@/lib/classification-analysis-store";
import { useSplitPreview } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { toast } from "sonner";

export const Route = createFileRoute("/classify/split")({
  beforeLoad: requireAuth,
  component: SplitPage,
});

function SplitPage() {
  const { state, update } = useClassificationAnalysis();
  const navigate = useNavigate();
  const s = state.split;
  const setSplit = (patch: Partial<typeof s>) => update({ split: { ...s, ...patch } });
  const splitPreview = useSplitPreview();

  if (!state.preprocessedDatasetId && !state.datasetId) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Please upload a dataset first.</p>
            <Button asChild className="mt-4">
              <Link to="/classify/upload">Go to upload</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const train = s.train;
  const val = s.useVal ? s.val : 0;
  const test = 100 - train - val;

  const handleContinue = async () => {
    const datasetId = state.preprocessedDatasetId ?? state.datasetId!;
    try {
      await splitPreview.mutateAsync({
        dataset_id: datasetId,
        config: toBackendSplitConfig(s),
      });
      navigate({ to: "/classify/model" });
    } catch (err) {
      toast.error(
        err instanceof ApiError
          ? err.message
          : "This split configuration isn't valid for your dataset size.",
      );
    }
  };

  return (
    <div>
      <WizardSteps steps={classificationSteps} />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Split Ratio</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <Switch
                checked={s.useVal}
                onCheckedChange={(v) => setSplit({ useVal: v, val: v ? 10 : 0 })}
              />
              <Label>Include Validation Set</Label>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Training</span>
                <span className="font-semibold">{train}%</span>
              </div>
              <Slider
                value={[train]}
                min={50}
                max={90}
                step={5}
                onValueChange={(v) => setSplit({ train: v[0] })}
              />
            </div>
            {s.useVal && (
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Validation</span>
                  <span className="font-semibold">{val}%</span>
                </div>
                <Slider
                  value={[val]}
                  min={5}
                  max={30}
                  step={5}
                  onValueChange={(v) => setSplit({ val: v[0] })}
                />
              </div>
            )}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Testing</span>
                <span className="font-semibold">{test}%</span>
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden flex">
                <div className="bg-indigo-500" style={{ width: `${train}%` }} />
                {s.useVal && <div className="bg-amber-500" style={{ width: `${val}%` }} />}
                <div className="bg-emerald-500" style={{ width: `${test}%` }} />
              </div>
            </div>
            <div>
              <Label>Random State</Label>
              <Input
                type="number"
                value={s.randomState}
                onChange={(e) => setSplit({ randomState: Number(e.target.value) })}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Visualization</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Row color="bg-indigo-500" label="Training" pct={train} />
            {s.useVal && <Row color="bg-amber-500" label="Validation" pct={val} />}
            <Row color="bg-emerald-500" label="Testing" pct={test} />
            <div className="flex justify-end pt-4">
              <Button onClick={handleContinue} disabled={splitPreview.isPending}>
                {splitPreview.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Continue
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Row({ color, label, pct }: { color: string; label: string; pct: number }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-6 rounded bg-muted overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
