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
import { WizardSteps, classificationSteps } from "@/components/wizard-steps";
import { useClassificationAnalysis } from "@/lib/classification-analysis-store";
import { usePredictClass } from "@/hooks/use-prediction";
import { useClassificationModelMetrics } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { toast } from "sonner";

export const Route = createFileRoute("/classify/predict")({
  beforeLoad: requireAuth,
  component: PredictPage,
});

function PredictPage() {
  const { state, update } = useClassificationAnalysis();
  const navigate = useNavigate();
  const [values, setValues] = useState<Record<string, string>>({});
  const predict = usePredictClass();
  const metricsQuery = useClassificationModelMetrics(state.modelId);

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first before making predictions.</p>
            <Button asChild className="mt-4">
              <Link to="/classify/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (metricsQuery.isLoading || !metricsQuery.data) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
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
      update({ prediction: { label: result.prediction, probabilities: result.probabilities } });
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Prediction failed");
    }
  };

  const sortedProbabilities = state.prediction?.probabilities
    ? Object.entries(state.prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div>
      <WizardSteps steps={classificationSteps} />
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
            <CardTitle className="text-white">Predicted Class</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold break-words">
              {state.prediction ? state.prediction.label : "—"}
            </div>
            <p className="mt-2 text-sm text-white/80">
              Predicted {state.target ?? "class"} using {state.model ?? "your model"}.
            </p>
            {sortedProbabilities.length > 0 && (
              <div className="mt-4 rounded-lg bg-white/10 p-3 space-y-2">
                <div className="font-medium text-white/90 text-xs mb-1">Class probabilities</div>
                {sortedProbabilities.map(([cls, prob]) => (
                  <div key={cls}>
                    <div className="flex justify-between text-xs text-white/80 mb-0.5">
                      <span>{cls}</span>
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-white/20 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-white"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
            {state.prediction && (
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
        <Button onClick={() => navigate({ to: "/classify/metrics" })}>
          Continue to Metrics
        </Button>
      </div>
    </div>
  );
}
