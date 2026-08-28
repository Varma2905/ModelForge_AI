import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { WizardSteps, classificationSteps } from "@/components/wizard-steps";
import { ScrollReveal } from "@/components/scroll-reveal";
import { useClassificationAnalysis } from "@/lib/classification-analysis-store";
import { useClassificationModels } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import { Check, PackageX } from "lucide-react";
import type { ClassificationModelEntry } from "@/lib/api-types";

export const Route = createFileRoute("/classify/model")({
  beforeLoad: requireAuth,
  component: ModelPage,
});

function ModelCard({ entry, selected, onSelect }: { entry: ClassificationModelEntry; selected: boolean; onSelect: () => void }) {
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
          {!isAvailable && (
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
          {selected ? "Selected" : isAvailable ? "Select Model" : "Unavailable"}
        </Button>
      </CardContent>
    </Card>
  );
}

function ModelPage() {
  const { state, update } = useClassificationAnalysis();
  const navigate = useNavigate();
  const modelsQuery = useClassificationModels();

  return (
    <div>
      <WizardSteps steps={classificationSteps} />

      {modelsQuery.isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full rounded-xl" />
          ))}
        </div>
      ) : modelsQuery.isError ? (
        <Card>
          <CardContent className="p-8 text-center text-sm text-destructive">
            Couldn't load the classification model catalog. Please try again.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {(modelsQuery.data ?? []).map((entry, i) => (
            <ScrollReveal key={entry.id} delay={(i % 3) * 60}>
              <ModelCard
                entry={entry}
                selected={state.model === entry.id}
                onSelect={() => update({ model: entry.id })}
              />
            </ScrollReveal>
          ))}
        </div>
      )}

      <div className="flex justify-end mt-6">
        <Button disabled={!state.model} onClick={() => navigate({ to: "/classify/train" })}>
          Continue
        </Button>
      </div>
    </div>
  );
}
