import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { WizardSteps } from "@/components/wizard-steps";
import { ScrollReveal } from "@/components/scroll-reveal";
import { useAnalysis } from "@/lib/analysis-store";
import { useRegressionModels } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { Check, Clock, PackageX } from "lucide-react";
import type { RegressionModelEntry } from "@/lib/api-types";

export const Route = createFileRoute("/new/model")({
  beforeLoad: requireAuth,
  component: ModelPage,
});

// Category display order — matches the spec's grouping exactly. Any
// category the backend registry doesn't know about yet still renders
// (falls in after these), so a future addition is never silently dropped.
const CATEGORY_ORDER = ["Basic Regression", "Regularization", "Tree Based", "Other", "Time Series"];

function ModelCard({ entry, selected, onSelect }: { entry: RegressionModelEntry; selected: boolean; onSelect: () => void }) {
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
          <h3 className="font-semibold">{entry.display_name}</h3>
          {selected && <Check className="h-4 w-4 text-primary shrink-0" />}
          {entry.status === "coming_soon" && (
            <Badge variant="outline" className="text-xs shrink-0 gap-1">
              <Clock className="h-3 w-3" /> Coming Soon
            </Badge>
          )}
          {entry.status === "unavailable" && (
            <Badge variant="outline" className="text-xs shrink-0 gap-1 text-destructive border-destructive/40">
              <PackageX className="h-3 w-3" /> Unavailable
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground">{entry.description}</p>
        {isAvailable ? (
          <div className="flex flex-wrap gap-1">
            {entry.pros.map((p) => (
              <Badge key={p} variant="secondary" className="text-xs">
                {p}
              </Badge>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground italic">{entry.reason}</p>
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
          {selected ? "Selected" : isAvailable ? "Select Model" : entry.status === "coming_soon" ? "Coming Soon" : "Unavailable"}
        </Button>
      </CardContent>
    </Card>
  );
}

function ModelPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const modelsQuery = useRegressionModels();

  const grouped = new Map<string, RegressionModelEntry[]>();
  for (const entry of modelsQuery.data ?? []) {
    if (!grouped.has(entry.category)) grouped.set(entry.category, []);
    grouped.get(entry.category)!.push(entry);
  }
  const categories = [
    ...CATEGORY_ORDER.filter((c) => grouped.has(c)),
    ...[...grouped.keys()].filter((c) => !CATEGORY_ORDER.includes(c)),
  ];

  return (
    <div>
      <WizardSteps />

      {modelsQuery.isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full rounded-xl" />
          ))}
        </div>
      ) : modelsQuery.isError ? (
        <Card>
          <CardContent className="p-8 text-center text-sm text-destructive">
            Couldn't load the regression model catalog. Please try again.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-8">
          {categories.map((category) => (
            <div key={category}>
              <h3 className="text-sm font-semibold text-muted-foreground mb-3">{category}</h3>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {grouped.get(category)!.map((entry, i) => (
                  <ScrollReveal key={entry.id} delay={(i % 3) * 60}>
                    <ModelCard
                      entry={entry}
                      selected={state.model === entry.id}
                      onSelect={() => update({ model: entry.id })}
                    />
                  </ScrollReveal>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="flex justify-end mt-6">
        <Button disabled={!state.model} onClick={() => navigate({ to: "/new/train" })}>
          Continue
        </Button>
      </div>
    </div>
  );
}
