import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { ScrollReveal } from "@/components/scroll-reveal";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { useClusteringModels } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { Check, PackageX } from "lucide-react";
import type { ClusteringModelEntry } from "@/lib/api-types";

export const Route = createFileRoute("/cluster/model")({
  beforeLoad: requireAuth,
  component: ModelPage,
});

function ModelCard({
  entry,
  selected,
  onSelect,
  hyperparameters,
  onParamChange,
}: {
  entry: ClusteringModelEntry;
  selected: boolean;
  onSelect: () => void;
  hyperparameters: Record<string, unknown>;
  onParamChange: (name: string, value: string | number) => void;
}) {
  const isAvailable = entry.status === "available";

  return (
    <Card
      variant="glass"
      className={
        isAvailable
          ? `card-interactive cursor-pointer ${selected ? "border-primary ring-2 ring-primary/30" : ""}`
          : "opacity-60 border-dashed"
      }
      onClick={isAvailable ? onSelect : undefined}
    >
      <CardContent className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h3 className="font-semibold">{entry.display_name}</h3>
            <Badge variant="outline" className="text-[10px] font-normal mt-1">
              {entry.algorithm_type}
            </Badge>
          </div>
          {selected && <Check className="h-4 w-4 text-primary shrink-0" />}
          {!isAvailable && (
            <Badge variant="outline" className="text-xs shrink-0 gap-1 text-destructive border-destructive/40">
              <PackageX className="h-3 w-3" /> Unavailable
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground">{entry.description}</p>
        {!isAvailable && <p className="text-xs text-muted-foreground italic">{entry.reason}</p>}

        {selected && isAvailable && entry.parameters.length > 0 && (
          <div className="space-y-2 pt-2 border-t" onClick={(e) => e.stopPropagation()}>
            {entry.parameters.map((p) => (
              <div key={p.name} className="flex items-center justify-between gap-2">
                <Label className="text-xs text-muted-foreground shrink-0">{p.label}</Label>
                {p.type === "select" ? (
                  <Select
                    value={String(hyperparameters[p.name] ?? p.default)}
                    onValueChange={(v) => onParamChange(p.name, v)}
                  >
                    <SelectTrigger className="h-7 w-28 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(p.options ?? []).map((opt) => (
                        <SelectItem key={opt} value={opt}>
                          {opt}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    type="number"
                    step={p.type === "float" ? "0.01" : "1"}
                    className="h-7 w-24 text-xs"
                    value={String(hyperparameters[p.name] ?? p.default)}
                    onChange={(e) => onParamChange(p.name, e.target.value === "" ? "" : Number(e.target.value))}
                  />
                )}
              </div>
            ))}
          </div>
        )}

        <Button
          size="sm"
          variant={selected ? "gradient" : "outline"}
          className="w-full"
          disabled={!isAvailable}
          onClick={(e) => {
            e.stopPropagation();
            if (isAvailable) onSelect();
          }}
        >
          {selected ? "Selected" : isAvailable ? "Select Algorithm" : "Unavailable"}
        </Button>
      </CardContent>
    </Card>
  );
}

function ModelPage() {
  const { state, update } = useClusteringAnalysis();
  const navigate = useNavigate();
  const modelsQuery = useClusteringModels();

  const handleSelect = (entry: ClusteringModelEntry) => {
    const defaults: Record<string, unknown> = {};
    for (const p of entry.parameters) defaults[p.name] = p.default;
    update({ model: entry.id, hyperparameters: defaults });
  };

  const handleParamChange = (name: string, value: string | number) => {
    update({ hyperparameters: { ...state.hyperparameters, [name]: value } });
  };

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />

      {modelsQuery.isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full rounded-xl" />
          ))}
        </div>
      ) : modelsQuery.isError ? (
        <Card>
          <CardContent className="p-8 text-center text-sm text-destructive">
            Couldn't load the clustering algorithm catalog. Please try again.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {(modelsQuery.data ?? []).map((entry, i) => (
            <ScrollReveal key={entry.id} delay={(i % 3) * 60}>
              <ModelCard
                entry={entry}
                selected={state.model === entry.id}
                onSelect={() => handleSelect(entry)}
                hyperparameters={state.hyperparameters}
                onParamChange={handleParamChange}
              />
            </ScrollReveal>
          ))}
        </div>
      )}

      <div className="flex justify-end mt-6">
        <Button disabled={!state.model} onClick={() => navigate({ to: "/cluster/train" })}>
          Continue
        </Button>
      </div>
    </div>
  );
}
