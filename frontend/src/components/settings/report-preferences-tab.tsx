import { useState } from "react";
import { toast } from "sonner";
import { Switch } from "@/components/ui/switch";
import { SettingsCard, SettingRow } from "@/components/settings/settings-card";
import { DEFAULT_REPORT_PREFERENCES, type ReportPreferences } from "@/lib/settings-store";

const TOGGLES: { key: keyof ReportPreferences; label: string; description: string }[] = [
  { key: "includeDatasetSummary", label: "Include Dataset Summary", description: "Dataset name, size, and column overview." },
  { key: "includeDataQuality", label: "Include Data Quality Information", description: "Missing values and data quality notes." },
  { key: "includePreprocessing", label: "Include Preprocessing Details", description: "Imputation, scaling, and encoding steps that were applied." },
  { key: "includeModelInfo", label: "Include Model Information", description: "Algorithm, hyperparameters, and training configuration." },
  { key: "includeModelEquation", label: "Include Model Equation", description: "Shown only where supported — currently Linear Regression." },
  { key: "includeErrorMetrics", label: "Include Error Metrics", description: "The evaluation metrics calculated by the ML engine." },
  { key: "includeStatisticalMetrics", label: "Include Statistical Metrics", description: "Coefficients, p-values, and related statistical analysis." },
  { key: "includeVisualizationPlots", label: "Include Visualization Plots", description: "The diagnostic charts generated for this analysis." },
  { key: "includeAiInsights", label: "Include AI Insights", description: "The AI Assistant's narrative interpretation of your results." },
  { key: "includeFeatureImportance", label: "Include Feature Importance", description: "Shown only where the trained model supports it." },
];

export function ReportPreferencesTab({
  value,
  onSave,
}: {
  value: ReportPreferences;
  onSave: (next: ReportPreferences) => void;
}) {
  const [draft, setDraft] = useState(value);

  const save = () => {
    onSave(draft);
    toast.success("Settings saved successfully.");
  };

  const reset = () => {
    setDraft(DEFAULT_REPORT_PREFERENCES);
    onSave(DEFAULT_REPORT_PREFERENCES);
    toast.success("Settings saved successfully.");
  };

  return (
    <SettingsCard
      title="Report Preferences"
      description="Sets your default preference for what a generated report includes. Every section listed here already exists in the report engine — this doesn't create anything new."
      onSave={save}
      onReset={reset}
    >
      {TOGGLES.map((t) => (
        <SettingRow key={t.key} label={t.label} description={t.description}>
          <Switch
            checked={draft[t.key]}
            onCheckedChange={(checked) => setDraft((d) => ({ ...d, [t.key]: checked }))}
          />
        </SettingRow>
      ))}
    </SettingsCard>
  );
}
