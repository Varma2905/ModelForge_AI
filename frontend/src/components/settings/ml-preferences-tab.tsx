import { useState } from "react";
import { toast } from "sonner";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SettingsCard, SettingRow } from "@/components/settings/settings-card";
import {
  DEFAULT_ML_PREFERENCES,
  type MlPreferences,
  type MlType,
  type SplitRatio,
  type CrossValidation,
} from "@/lib/settings-store";

const SPLIT_OPTIONS: SplitRatio[] = ["70/30", "75/25", "80/20", "85/15", "90/10"];
const CV_OPTIONS: { value: CrossValidation; label: string }[] = [
  { value: "disabled", label: "Disabled" },
  { value: "3", label: "3 Folds" },
  { value: "5", label: "5 Folds" },
  { value: "10", label: "10 Folds" },
];

export function MlPreferencesTab({
  value,
  onSave,
}: {
  value: MlPreferences;
  onSave: (next: MlPreferences) => void;
}) {
  const [draft, setDraft] = useState(value);

  const save = () => {
    onSave(draft);
    toast.success("Settings saved successfully.");
  };

  const reset = () => {
    setDraft(DEFAULT_ML_PREFERENCES);
    onSave(DEFAULT_ML_PREFERENCES);
    toast.success("Settings saved successfully.");
  };

  return (
    <SettingsCard
      title="ML Preferences"
      description="Defaults applied when you start a new analysis. You can always override these per-analysis in the wizard."
      onSave={save}
      onReset={reset}
    >
      <SettingRow label="Default ML Type" description="Which workflow the dashboard's Quick Actions favor.">
        <RadioGroup
          value={draft.defaultMlType}
          onValueChange={(v) => setDraft((d) => ({ ...d, defaultMlType: v as MlType }))}
          className="flex gap-4"
        >
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <RadioGroupItem value="regression" /> Regression
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <RadioGroupItem value="classification" /> Classification
          </label>
        </RadioGroup>
      </SettingRow>

      <SettingRow
        label="Default Train/Test Split"
        description="Pre-selected split ratio when you reach the Split step of a new analysis."
      >
        <Select value={draft.defaultSplit} onValueChange={(v) => setDraft((d) => ({ ...d, defaultSplit: v as SplitRatio }))}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SPLIT_OPTIONS.map((s) => (
              <SelectItem key={s} value={s}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingRow>

      <SettingRow label="Random Seed" description="Controls reproducibility of model training.">
        <Input
          type="number"
          value={draft.randomSeed}
          onChange={(e) => setDraft((d) => ({ ...d, randomSeed: Number(e.target.value) || 0 }))}
          className="w-24"
        />
      </SettingRow>

      <SettingRow
        label="Cross Validation"
        description="Controls how many folds are used when evaluating models, where cross-validation is applied."
      >
        <Select
          value={draft.crossValidation}
          onValueChange={(v) => setDraft((d) => ({ ...d, crossValidation: v as CrossValidation }))}
        >
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CV_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingRow>

      <SettingRow
        label="Auto Model Selection"
        description="Suggest a recommended model automatically based on your dataset's shape."
      >
        <Switch
          checked={draft.autoModelSelection}
          onCheckedChange={(checked) => setDraft((d) => ({ ...d, autoModelSelection: checked }))}
        />
      </SettingRow>
    </SettingsCard>
  );
}
