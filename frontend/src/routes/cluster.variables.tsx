import { ReactNode, useEffect } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { useSelectClusteringFeatures } from "@/hooks/use-training";
import { useDatasetProfile } from "@/hooks/use-datasets";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Calendar, Hash, Loader2, Type } from "lucide-react";
import { toast } from "sonner";
import type { ColumnProfile } from "@/lib/api-types";

export const Route = createFileRoute("/cluster/variables")({
  beforeLoad: requireAuth,
  component: VariablesPage,
});

function VariablesPage() {
  const { state, update } = useClusteringAnalysis();
  const navigate = useNavigate();
  const selectFeatures = useSelectClusteringFeatures();
  const profileQuery = useDatasetProfile(state.datasetId);

  useEffect(() => {
    if (profileQuery.data && state.features.length === 0) {
      const eligible = profileQuery.data.features
        .filter((c) => c.kind === "numerical" && !c.is_identifier)
        .map((c) => c.name);
      if (eligible.length > 0) {
        update({ features: eligible });
      }
    }
  }, [profileQuery.data, state.features.length, update]);

  if (!state.datasetId || !state.dataset) {
    return (
      <div>
        <WizardSteps steps={clusteringSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Please upload a dataset first.</p>
            <Button asChild className="mt-4">
              <Link to="/cluster/upload">Go to upload</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (profileQuery.isLoading) {
    return (
      <div>
        <WizardSteps steps={clusteringSteps} />
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
        <WizardSteps steps={clusteringSteps} />
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

  const validationMessages: string[] = [];
  if (state.features.length === 0) {
    validationMessages.push("Select at least one input feature.");
  }

  const handleContinue = async () => {
    try {
      const result = await selectFeatures.mutateAsync({
        dataset_id: state.datasetId!,
        features: state.features,
      });
      const featureKinds: Record<string, "numerical" | "categorical"> = {};
      for (const f of result.numerical_features) featureKinds[f] = "numerical";
      for (const f of result.categorical_features) featureKinds[f] = "categorical";
      update({ featureKinds });
      for (const w of result.warnings) toast.warning(w);
      navigate({ to: "/cluster/model" });
    } catch (err) {
      toast.error(
        err instanceof ApiError
          ? err.message
          : "Selected columns are invalid for clustering. Please review your selection.",
      );
    }
  };

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />
      <Card>
        <CardHeader>
          <CardTitle>Input Features (X)</CardTitle>
          <p className="text-xs text-muted-foreground">
            Clustering is unsupervised — there's no target column to select. Just choose the
            features you want the algorithm to find natural groupings across.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <FeatureSection
            icon={<Hash className="h-3.5 w-3.5" />}
            title="Numerical Features"
            badgeLabel="Numerical"
            cols={numericalCols}
            selected={state.features}
            onToggle={toggleFeature}
          />
          <FeatureSection
            icon={<Type className="h-3.5 w-3.5" />}
            title="Categorical Features"
            badgeLabel="Categorical"
            cols={categoricalCols}
            selected={state.features}
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
  selected,
  onToggle,
}: {
  icon: ReactNode;
  title: string;
  badgeLabel: string;
  cols: ColumnProfile[];
  selected: string[];
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
                checked={selected.includes(c.name)}
                onCheckedChange={() => onToggle(c.name)}
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
            {(c.high_cardinality || c.is_identifier) && (
              <p className="flex items-center gap-1.5 mt-1 pl-3 text-xs text-amber-600 dark:text-amber-500">
                <AlertTriangle className="h-3 w-3 shrink-0" />
                {`'${c.name}' appears to be an identifier or high-cardinality column. Consider excluding it from clustering.`}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
