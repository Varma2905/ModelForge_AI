import { useState } from "react";
import { toast } from "sonner";
import { ChevronDown } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { SettingsCard } from "@/components/settings/settings-card";
import { useRegressionModels, useClassificationModels } from "@/hooks/use-training";
import { DEFAULT_MODEL_PREFERENCES, type ModelPreferences } from "@/lib/settings-store";
import type { RegressionModelEntry, ClassificationModelEntry } from "@/lib/api-types";

function statusBadge(status: string) {
  if (status === "coming_soon") return <Badge variant="outline" className="text-[10px] font-normal">Coming Soon</Badge>;
  if (status === "unavailable") return <Badge variant="outline" className="text-[10px] font-normal text-muted-foreground">Unavailable</Badge>;
  return null;
}

function ModelRow({
  entry,
  enabled,
  onToggle,
}: {
  entry: { id: string; display_name: string; status: string; reason: string | null };
  enabled: boolean;
  onToggle: (checked: boolean) => void;
}) {
  const isAvailable = entry.status === "available";
  return (
    <div className="flex items-center justify-between gap-3 py-1.5 px-1">
      <div className="min-w-0 flex items-center gap-2">
        <span className={`text-sm truncate ${!isAvailable ? "text-muted-foreground" : ""}`}>{entry.display_name}</span>
        {statusBadge(entry.status)}
      </div>
      <Switch
        checked={isAvailable && enabled}
        disabled={!isAvailable}
        onCheckedChange={onToggle}
        aria-label={`Enable ${entry.display_name}`}
      />
    </div>
  );
}

function CategoryGroup({
  title,
  entries,
  prefs,
  onToggle,
}: {
  title: string;
  entries: { id: string; display_name: string; status: string; reason: string | null }[];
  prefs: Record<string, boolean>;
  onToggle: (id: string, checked: boolean) => void;
}) {
  const [open, setOpen] = useState(true);
  const enabledCount = entries.filter((e) => e.status === "available" && (prefs[e.id] ?? true)).length;

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-md border">
      <CollapsibleTrigger className="flex w-full items-center justify-between px-3 py-2 text-sm font-semibold cursor-pointer hover:bg-accent/50 transition-colors rounded-md">
        <span>
          {title} <span className="text-xs font-normal text-muted-foreground">({enabledCount}/{entries.length} enabled)</span>
        </span>
        <ChevronDown className={`h-4 w-4 transition-transform duration-150 ${open ? "rotate-180" : ""}`} />
      </CollapsibleTrigger>
      <CollapsibleContent className="px-2 pb-2 space-y-0.5">
        {entries.map((e) => (
          <ModelRow key={e.id} entry={e} enabled={prefs[e.id] ?? true} onToggle={(checked) => onToggle(e.id, checked)} />
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}

function groupByCategory(entries: RegressionModelEntry[]): Map<string, RegressionModelEntry[]> {
  const map = new Map<string, RegressionModelEntry[]>();
  for (const e of entries) {
    const list = map.get(e.category) ?? [];
    list.push(e);
    map.set(e.category, list);
  }
  return map;
}

export function ModelPreferencesTab({
  value,
  onSave,
}: {
  value: ModelPreferences;
  onSave: (next: ModelPreferences) => void;
}) {
  const [draft, setDraft] = useState(value);
  const regressionQuery = useRegressionModels();
  const classificationQuery = useClassificationModels();

  const toggleRegression = (id: string, checked: boolean) =>
    setDraft((d) => ({ ...d, regression: { ...d.regression, [id]: checked } }));
  const toggleClassification = (id: string, checked: boolean) =>
    setDraft((d) => ({ ...d, classification: { ...d.classification, [id]: checked } }));

  const save = () => {
    onSave(draft);
    toast.success("Settings saved successfully.");
  };

  const reset = () => {
    setDraft(DEFAULT_MODEL_PREFERENCES);
    onSave(DEFAULT_MODEL_PREFERENCES);
    toast.success("Settings saved successfully.");
  };

  const isLoading = regressionQuery.isLoading || classificationQuery.isLoading;
  const regressionByCategory = regressionQuery.data ? groupByCategory(regressionQuery.data) : null;

  return (
    <SettingsCard
      title="Model Preferences"
      description="Choose which real, implemented models are offered when you select an algorithm. Models the app can't currently train are always shown disabled."
      onSave={save}
      onReset={reset}
    >
      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-10 w-full rounded-md" />
          ))}
        </div>
      ) : (
        <>
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Regression</p>
            <div className="space-y-2">
              {regressionByCategory &&
                Array.from(regressionByCategory.entries()).map(([category, entries]) => (
                  <CategoryGroup
                    key={category}
                    title={category}
                    entries={entries}
                    prefs={draft.regression}
                    onToggle={toggleRegression}
                  />
                ))}
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Classification</p>
            {/* No sub-categories on this registry (unlike Regression's), so
                just a plain always-visible list — no Collapsible needed. */}
            <div className="rounded-md border px-2 py-2 space-y-0.5">
              {classificationQuery.data?.map((e: ClassificationModelEntry) => (
                <ModelRow
                  key={e.id}
                  entry={e}
                  enabled={draft.classification[e.id] ?? true}
                  onToggle={(checked) => toggleClassification(e.id, checked)}
                />
              ))}
            </div>
          </div>
        </>
      )}
    </SettingsCard>
  );
}
