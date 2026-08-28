import type { ReactNode } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { useSelectFeatures } from "@/hooks/use-preprocessing";
import { useDatasetProfile } from "@/hooks/use-datasets";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Calendar, Hash, Loader2, Type } from "lucide-react";
import { toast } from "sonner";
import type { ColumnProfile } from "@/lib/api-types";

export const Route = createFileRoute("/new/variables")({
  beforeLoad: requireAuth,
  component: VariablesPage,
});

function VariablesPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const selectFeatures = useSelectFeatures();
  const profileQuery = useDatasetProfile(state.datasetId);

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

  if (profileQuery.isLoading) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center text-sm text-muted-foreground">
            <Loader2 className="h-5 w-5 mx-auto mb-2 animate-spin" />
            Analyzing column types…
          </CardContent>
        </Card>
      </div>
    );
  }

  if (profileQuery.isError || !profileQuery.data) {
    return (
      <div>
        <WizardSteps />
        <Card>
          <CardContent className="p-8 text-center text-sm text-muted-foreground">
            Couldn't analyze this dataset's columns. Please try again.
          </CardContent>
        </Card>
      </div>
    );
  }

  const allCols = profileQuery.data.features;
  const numericalCols = allCols.filter((c) => c.kind === "numerical");
  const categoricalCols = allCols.filter((c) => c.kind === "categorical");
  const datetimeCols = allCols.filter((c) => c.kind === "datetime");

  const toggleFeature = (c: string) => {
    const has = state.features.includes(c);
    update({ features: has ? state.features.filter((x) => x !== c) : [...state.features, c] });
  };

  const targetProfile = allCols.find((c) => c.name === state.target);

  // Explicit, user-visible reasons Continue is disabled — covers the cases
  // the UI already structurally prevents (categorical target, target as a
  // feature) as a defensive backstop, plus the two a user can actually hit.
  const validationMessages: string[] = [];
  if (state.features.length === 0) {
    validationMessages.push("Select at least one input feature.");
  }
  if (!state.target) {
    validationMessages.push("Select a target variable.");
  } else if (targetProfile && targetProfile.kind !== "numerical") {
    validationMessages.push("Regression requires a numerical target variable. Please select a numerical column.");
  } else if (state.features.includes(state.target)) {
    validationMessages.push("The target variable cannot also be selected as an input feature.");
  }

  const handleContinue = async () => {
    if (!state.target) return;
    try {
      const result = await selectFeatures.mutateAsync({
        dataset_id: state.datasetId!,
        features: state.features,
        target: state.target,
      });
      const featureKinds: Record<string, "numerical" | "categorical"> = {};
      for (const f of result.numerical_features) featureKinds[f] = "numerical";
      for (const f of result.categorical_features) featureKinds[f] = "categorical";
      update({ featureKinds });
      for (const w of result.warnings) toast.warning(w);
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
            <CardTitle>Input Variables (X)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <FeatureSection
              icon={<Hash className="h-3.5 w-3.5" />}
              title="Numerical Features"
              badgeLabel="Numerical"
              cols={numericalCols}
              state={state}
              onToggle={toggleFeature}
            />
            <FeatureSection
              icon={<Type className="h-3.5 w-3.5" />}
              title="Categorical Features"
              badgeLabel="Categorical"
              cols={categoricalCols}
              state={state}
              onToggle={toggleFeature}
            />
            {datetimeCols.length > 0 && (
              <div>
                <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground mb-2">
                  <Calendar className="h-3.5 w-3.5" />
                  Date/Time (not supported as features yet)
                </div>
                <div className="space-y-2">
                  {datetimeCols.map((c) => (
                    <div
                      key={c.name}
                      className="flex items-center gap-3 rounded-lg border border-dashed p-3 opacity-50"
                    >
                      <Checkbox checked={false} disabled />
                      <span className="font-medium">{c.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Target Variable (Y)</CardTitle>
            <p className="text-xs text-muted-foreground">
              Regression requires a numerical target — categorical columns never appear here.
            </p>
          </CardHeader>
          <CardContent>
            {numericalCols.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                This dataset has no numerical columns, so it has no valid regression target.
              </p>
            ) : (
              <RadioGroup
                value={state.target ?? ""}
                onValueChange={(v) => {
                  // A target switch restores the PREVIOUS target column back
                  // into the input-feature list (if it isn't already there) —
                  // the user picked it as a feature/target intentionally at
                  // some point, so losing it silently on a target change
                  // would be surprising.
                  const prevTarget = state.target;
                  let nextFeatures = state.features.filter((f) => f !== v);
                  if (prevTarget && prevTarget !== v && !nextFeatures.includes(prevTarget)) {
                    nextFeatures = [...nextFeatures, prevTarget];
                  }
                  update({ target: v, features: nextFeatures });
                }}
                className="space-y-2"
              >
                {numericalCols.map((c) => (
                  <label
                    key={c.name}
                    className="flex items-center gap-3 rounded-lg border p-3 hover:bg-accent cursor-pointer"
                  >
                    <RadioGroupItem value={c.name} id={`t-${c.name}`} />
                    <Label htmlFor={`t-${c.name}`} className="cursor-pointer flex-1">
                      {c.name}
                    </Label>
                    <Badge variant="outline" className="text-[10px] font-normal">
                      Numerical
                    </Badge>
                  </label>
                ))}
              </RadioGroup>
            )}
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

          {validationMessages.length > 0 && (
            <ul className="space-y-1 rounded-lg border border-destructive/40 bg-destructive/5 p-3">
              {validationMessages.map((msg) => (
                <li key={msg} className="text-xs text-destructive">
                  {msg}
                </li>
              ))}
            </ul>
          )}

          <div className="flex justify-end">
            <Button
              disabled={validationMessages.length > 0 || selectFeatures.isPending}
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

function FeatureSection({
  icon,
  title,
  badgeLabel,
  cols,
  state,
  onToggle,
}: {
  icon: ReactNode;
  title: string;
  badgeLabel: string;
  cols: ColumnProfile[];
  state: ReturnType<typeof useAnalysis>["state"];
  onToggle: (c: string) => void;
}) {
  if (cols.length === 0) return null;
  return (
    <div>
      <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground mb-2">
        {icon}
        {title}
      </div>
      <div className="space-y-2">
        {cols.map((c) => (
          <div key={c.name}>
            <label className="flex items-center gap-3 rounded-lg border p-3 hover:bg-accent cursor-pointer">
              <Checkbox
                checked={state.features.includes(c.name)}
                onCheckedChange={() => onToggle(c.name)}
                disabled={state.target === c.name}
              />
              <span className="font-medium flex-1">{c.name}</span>
              {c.missing_count > 0 && (
                <Badge variant="outline" className="text-[10px] font-normal">
                  {c.missing_count} missing
                </Badge>
              )}
              <Badge variant="outline" className="text-[10px] font-normal">
                {badgeLabel}
              </Badge>
            </label>
            {c.high_cardinality && (
              <p className="flex items-center gap-1.5 mt-1 pl-3 text-xs text-amber-600 dark:text-amber-500">
                <AlertTriangle className="h-3 w-3 shrink-0" />
                {`'${c.name}' appears to be an identifier and contains many unique values. Consider excluding it from the model.`}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
