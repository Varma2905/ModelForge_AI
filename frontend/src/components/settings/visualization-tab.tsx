import { useState } from "react";
import { toast } from "sonner";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SettingsCard, SettingRow } from "@/components/settings/settings-card";
import {
  DEFAULT_VISUALIZATION,
  ALL_REGRESSION_PLOTS,
  ALL_CLASSIFICATION_PLOTS,
  type VisualizationPreferences,
  type MaxCharts,
  type RegressionPlotKey,
  type ClassificationPlotKey,
} from "@/lib/settings-store";

const MAX_CHARTS_OPTIONS: MaxCharts[] = ["4", "6", "8", "10", "unlimited"];

function PlotCheckboxList<T extends string>({
  all,
  selected,
  onToggle,
}: {
  all: T[];
  selected: T[];
  onToggle: (key: T, checked: boolean) => void;
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {all.map((key) => (
        <label key={key} className="flex items-center gap-2 text-sm cursor-pointer rounded-md border p-2">
          <Checkbox checked={selected.includes(key)} onCheckedChange={(c) => onToggle(key, c === true)} />
          {key}
        </label>
      ))}
    </div>
  );
}

export function VisualizationTab({
  value,
  onSave,
}: {
  value: VisualizationPreferences;
  onSave: (next: VisualizationPreferences) => void;
}) {
  const [draft, setDraft] = useState(value);

  const toggleRegressionPlot = (key: RegressionPlotKey, checked: boolean) =>
    setDraft((d) => ({
      ...d,
      regressionPlots: checked ? [...d.regressionPlots, key] : d.regressionPlots.filter((k) => k !== key),
    }));

  const toggleClassificationPlot = (key: ClassificationPlotKey, checked: boolean) =>
    setDraft((d) => ({
      ...d,
      classificationPlots: checked
        ? [...d.classificationPlots, key]
        : d.classificationPlots.filter((k) => k !== key),
    }));

  const save = () => {
    onSave(draft);
    toast.success("Settings saved successfully.");
  };

  const reset = () => {
    setDraft(DEFAULT_VISUALIZATION);
    onSave(DEFAULT_VISUALIZATION);
    toast.success("Settings saved successfully.");
  };

  return (
    <SettingsCard
      title="Visualization"
      description="Only charts already implemented in this app are listed below — nothing here creates a placeholder chart."
      onSave={save}
      onReset={reset}
    >
      <SettingRow label="Auto Generate Visualizations" description="Build charts automatically once a model finishes training.">
        <Switch
          checked={draft.autoGenerate}
          onCheckedChange={(checked) => setDraft((d) => ({ ...d, autoGenerate: checked }))}
        />
      </SettingRow>

      <SettingRow label="Interactive Charts" description="Use interactive (hover/zoom) charts instead of static images where available.">
        <Switch
          checked={draft.interactiveCharts}
          onCheckedChange={(checked) => setDraft((d) => ({ ...d, interactiveCharts: checked }))}
        />
      </SettingRow>

      <SettingRow label="Maximum Charts" description="Caps how many charts are shown at once on results pages.">
        <Select value={draft.maxCharts} onValueChange={(v) => setDraft((d) => ({ ...d, maxCharts: v as MaxCharts }))}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MAX_CHARTS_OPTIONS.map((o) => (
              <SelectItem key={o} value={o}>
                {o === "unlimited" ? "Unlimited" : o}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingRow>

      <div>
        <p className="text-sm font-medium mb-2">Regression Plot Preferences</p>
        <PlotCheckboxList all={ALL_REGRESSION_PLOTS} selected={draft.regressionPlots} onToggle={toggleRegressionPlot} />
      </div>

      <div>
        <p className="text-sm font-medium mb-2">Classification Plot Preferences</p>
        <PlotCheckboxList
          all={ALL_CLASSIFICATION_PLOTS}
          selected={draft.classificationPlots}
          onToggle={toggleClassificationPlot}
        />
      </div>
    </SettingsCard>
  );
}
