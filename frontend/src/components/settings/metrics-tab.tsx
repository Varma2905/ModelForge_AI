import { useState } from "react";
import { toast } from "sonner";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SettingsCard } from "@/components/settings/settings-card";
import {
  DEFAULT_METRICS,
  ALL_REGRESSION_METRICS,
  ALL_CLASSIFICATION_METRICS,
  type MetricsPreferences,
  type RegressionMetricKey,
  type ClassificationMetricKey,
  type PrimaryRegressionMetric,
  type PrimaryClassificationMetric,
} from "@/lib/settings-store";

// Display labels only — the underlying keys match the backend's actual
// metrics dict exactly (see app/ml/evaluation.py / ClassificationMetrics),
// so this tab never invents a metric that isn't already calculated.
const REGRESSION_LABELS: Record<RegressionMetricKey, string> = {
  MAE: "MAE",
  MSE: "MSE",
  RMSE: "RMSE",
  MAPE: "MAPE",
  MSLE: "MSLE",
  RMSLE: "RMSLE",
  "Median Absolute Error": "Median Absolute Error",
  "Max Error": "Max Error",
  R2: "R²",
  "Adjusted R2": "Adjusted R²",
  "Explained Variance": "Explained Variance",
  "Pearson Correlation": "Pearson Correlation",
  "Spearman Correlation": "Spearman Correlation",
};

const CLASSIFICATION_LABELS: Record<ClassificationMetricKey, string> = {
  Accuracy: "Accuracy",
  Precision: "Precision",
  Recall: "Recall",
  F1: "F1 Score",
  ConfusionMatrix: "Confusion Matrix",
  ROC_AUC: "ROC-AUC",
};

const PRIMARY_REGRESSION: PrimaryRegressionMetric[] = ["R2", "RMSE", "MAE"];
const PRIMARY_CLASSIFICATION: PrimaryClassificationMetric[] = ["Accuracy", "F1", "Precision", "Recall"];

function MetricCheckboxGrid<T extends string>({
  all,
  labels,
  selected,
  onToggle,
}: {
  all: T[];
  labels: Record<T, string>;
  selected: T[];
  onToggle: (key: T, checked: boolean) => void;
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
      {all.map((key) => (
        <label key={key} className="flex items-center gap-2 text-sm cursor-pointer rounded-md border p-2">
          <Checkbox checked={selected.includes(key)} onCheckedChange={(c) => onToggle(key, c === true)} />
          {labels[key]}
        </label>
      ))}
    </div>
  );
}

export function MetricsTab({
  value,
  onSave,
}: {
  value: MetricsPreferences;
  onSave: (next: MetricsPreferences) => void;
}) {
  const [draft, setDraft] = useState(value);

  const toggleRegressionMetric = (key: RegressionMetricKey, checked: boolean) =>
    setDraft((d) => ({
      ...d,
      regressionMetrics: checked
        ? [...d.regressionMetrics, key]
        : d.regressionMetrics.filter((k) => k !== key),
    }));

  const toggleClassificationMetric = (key: ClassificationMetricKey, checked: boolean) =>
    setDraft((d) => ({
      ...d,
      classificationMetrics: checked
        ? [...d.classificationMetrics, key]
        : d.classificationMetrics.filter((k) => k !== key),
    }));

  const save = () => {
    onSave(draft);
    toast.success("Settings saved successfully.");
  };

  const reset = () => {
    setDraft(DEFAULT_METRICS);
    onSave(DEFAULT_METRICS);
    toast.success("Settings saved successfully.");
  };

  return (
    <SettingsCard
      title="Metrics"
      description="Choose which already-calculated metrics appear in analysis results. This controls display only — the ML engine still computes every metric the same way regardless of these choices."
      onSave={save}
      onReset={reset}
    >
      <div>
        <p className="text-sm font-medium mb-2">Regression Metrics</p>
        <MetricCheckboxGrid
          all={ALL_REGRESSION_METRICS}
          labels={REGRESSION_LABELS}
          selected={draft.regressionMetrics}
          onToggle={toggleRegressionMetric}
        />
      </div>

      <div>
        <p className="text-sm font-medium mb-2">Classification Metrics</p>
        <MetricCheckboxGrid
          all={ALL_CLASSIFICATION_METRICS}
          labels={CLASSIFICATION_LABELS}
          selected={draft.classificationMetrics}
          onToggle={toggleClassificationMetric}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-md border p-3">
          <p className="text-sm font-medium">Primary Metric — Regression</p>
          <p className="text-xs text-muted-foreground mt-0.5 mb-2">
            Highlighted first on results and comparison views.
          </p>
          <RadioGroup
            value={draft.primaryRegressionMetric}
            onValueChange={(v) => setDraft((d) => ({ ...d, primaryRegressionMetric: v as PrimaryRegressionMetric }))}
            className="flex flex-wrap gap-3"
          >
            {PRIMARY_REGRESSION.map((m) => (
              <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer">
                <RadioGroupItem value={m} /> {REGRESSION_LABELS[m]}
              </label>
            ))}
          </RadioGroup>
        </div>

        <div className="rounded-md border p-3">
          <p className="text-sm font-medium">Primary Metric — Classification</p>
          <p className="text-xs text-muted-foreground mt-0.5 mb-2">
            Highlighted first on results and comparison views.
          </p>
          <RadioGroup
            value={draft.primaryClassificationMetric}
            onValueChange={(v) =>
              setDraft((d) => ({ ...d, primaryClassificationMetric: v as PrimaryClassificationMetric }))
            }
            className="flex flex-wrap gap-3"
          >
            {PRIMARY_CLASSIFICATION.map((m) => (
              <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer">
                <RadioGroupItem value={m} /> {CLASSIFICATION_LABELS[m]}
              </label>
            ))}
          </RadioGroup>
        </div>
      </div>
    </SettingsCard>
  );
}
