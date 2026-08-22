import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { useSelectFeatures } from "@/hooks/use-preprocessing";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/new/variables")({
  beforeLoad: requireAuth,
  component: VariablesPage,
});

function VariablesPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const cols = state.dataset?.columns ?? [];
  const selectFeatures = useSelectFeatures();

  if (!state.datasetId || !state.dataset) {
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

  const toggleFeature = (c: string) => {
    const has = state.features.includes(c);
    update({ features: has ? state.features.filter((x) => x !== c) : [...state.features, c] });
  };

  const handleContinue = async () => {
    if (!state.target) return;
    try {
      await selectFeatures.mutateAsync({
        dataset_id: state.datasetId!,
        features: state.features,
        target: state.target,
      });
      navigate({ to: "/new/preprocessing" });
    } catch (err) {
      toast.error(
        err instanceof ApiError
          ? err.message
          : "Selected columns are invalid for regression. Please review your selection.",
      );
    }
  };

  return (
    <div>
      <WizardSteps />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Input Features (X)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {cols.map((c) => (
              <label
                key={c}
                className="flex items-center gap-3 rounded-lg border p-3 hover:bg-accent cursor-pointer"
              >
                <Checkbox
                  checked={state.features.includes(c)}
                  onCheckedChange={() => toggleFeature(c)}
                  disabled={state.target === c}
                />
                <span className="font-medium">{c}</span>
              </label>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Target Variable (Y)</CardTitle>
          </CardHeader>
          <CardContent>
            <RadioGroup
              value={state.target ?? ""}
              onValueChange={(v) =>
                update({ target: v, features: state.features.filter((f) => f !== v) })
              }
              className="space-y-2"
            >
              {cols.map((c) => (
                <label
                  key={c}
                  className="flex items-center gap-3 rounded-lg border p-3 hover:bg-accent cursor-pointer"
                >
                  <RadioGroupItem value={c} id={`t-${c}`} />
                  <Label htmlFor={`t-${c}`} className="cursor-pointer">
                    {c}
                  </Label>
                </label>
              ))}
            </RadioGroup>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Selection Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <div className="text-xs text-muted-foreground mb-1">Features</div>
            <div className="flex flex-wrap gap-1">
              {state.features.length === 0 && (
                <span className="text-sm text-muted-foreground">None</span>
              )}
              {state.features.map((f) => (
                <Badge key={f} variant="secondary">
                  {f}
                </Badge>
              ))}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1">Target</div>
            {state.target ? (
              <Badge>{state.target}</Badge>
            ) : (
              <span className="text-sm text-muted-foreground">Not selected</span>
            )}
          </div>
          <div className="flex justify-end">
            <Button
              disabled={state.features.length === 0 || !state.target || selectFeatures.isPending}
              onClick={handleContinue}
            >
              {selectFeatures.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Continue
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
