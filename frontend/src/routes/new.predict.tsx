import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2, Sparkles } from "lucide-react";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { usePredict } from "@/hooks/use-prediction";
import { useModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { toast } from "sonner";

export const Route = createFileRoute("/new/predict")({
  beforeLoad: requireAuth,
  component: PredictPage,
});

function PredictPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const [values, setValues] = useState<Record<string, string>>({});
  const predict = usePredict();
  const metricsQuery = useModelMetrics(state.modelId);

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first before making predictions.</p>
            <Button asChild className="mt-4">
              <Link to="/new/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (metricsQuery.isLoading || !metricsQuery.data) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center text-sm text-muted-foreground">
            <Loader2 className="h-5 w-5 mx-auto mb-2 animate-spin" />
            Loading model…
          </CardContent>
        </Card>
      </div>
    );
  }

  const model = metricsQuery.data;
  const categoricalFeatures = new Set(model.categorical_features);
  const features = model.features;

  const missingFeatures = features.filter((f) => !values[f]?.trim());

  const runPredict = async () => {
    if (missingFeatures.length > 0) {
      toast.error(`Please fill in: ${missingFeatures.join(", ")}`);
      return;
    }
    try {
      const payload: Record<string, number | string> = {};
      for (const f of features) {
        payload[f] = categoricalFeatures.has(f) ? values[f] : Number(values[f]);
      }
      const result = await predict.mutateAsync({ model_id: state.modelId!, values: payload });
      update({ prediction: result.prediction });
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Prediction failed");
    }
  };

  return (
    <div>
      <WizardSteps />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Enter Feature Values</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {features.length === 0 && (
              <p className="text-sm text-muted-foreground">No features selected.</p>
            )}
            {features.map((f) => {
              const options = model.categorical_options[f];
              if (categoricalFeatures.has(f) && options) {
                return (
                  <div key={f}>
                    <Label>{f}</Label>
                    <Select
                      value={values[f] ?? ""}
                      onValueChange={(v) => setValues((prev) => ({ ...prev, [f]: v }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a value" />
                      </SelectTrigger>
                      <SelectContent>
                        {options.map((opt) => (
                          <SelectItem key={opt} value={opt}>
                            {opt}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                );
              }
              return (
                <div key={f}>
                  <Label>{f}</Label>
                  <Input
                    type="number"
                    value={values[f] ?? ""}
                    onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
                  />
                </div>
              );
            })}
            <Button onClick={runPredict} className="w-full" disabled={predict.isPending}>
              {predict.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4 mr-2" />
              )}
              Predict
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-[image:var(--gradient-brand)] text-white border-0 shadow-xl shadow-primary/20">
          <CardHeader>
            <CardTitle className="text-white">Predicted Value</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-5xl font-bold">
              {state.prediction !== null
                ? state.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })
                : "—"}
            </div>
            <p className="mt-2 text-sm text-white/80">
              Predicted {state.target ?? "target"} using {state.model ?? "your model"}.
            </p>
            {state.prediction !== null && (
              <div className="mt-4 rounded-lg bg-white/10 p-3 text-xs space-y-1">
                <div className="font-medium text-white/90 mb-1">Input summary</div>
                {features.map((f) => (
                  <div key={f} className="flex justify-between text-white/80">
                    <span>{f}</span>
                    <span>{values[f]}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/new/metrics" })}>
          Continue to Metrics
        </Button>
      </div>
    </div>
  );
}
