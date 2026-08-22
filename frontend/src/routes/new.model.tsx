import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { requireAuth } from "@/lib/require-auth";
import { Check } from "lucide-react";

export const Route = createFileRoute("/new/model")({
  beforeLoad: requireAuth,
  component: ModelPage,
});

const models = [
  {
    name: "Linear Regression",
    desc: "Fits a straight line to model the relationship.",
    pros: ["Simple", "Fast", "Interpretable"],
  },
  {
    name: "Multiple Linear Regression",
    desc: "Linear model with multiple input features.",
    pros: ["Multi-feature", "Interpretable"],
  },
  {
    name: "Polynomial Regression",
    desc: "Fits nonlinear polynomial curves.",
    pros: ["Handles curves", "Flexible"],
  },
  {
    name: "Ridge Regression",
    desc: "Linear regression with L2 regularization.",
    pros: ["Reduces overfit", "Stable"],
  },
  {
    name: "Lasso Regression",
    desc: "Linear regression with L1 regularization.",
    pros: ["Feature selection", "Sparse"],
  },
  {
    name: "Elastic Net Regression",
    desc: "Combines L1 and L2 regularization.",
    pros: ["Balanced", "Robust"],
  },
  {
    name: "Decision Tree Regression",
    desc: "Tree-based nonlinear regressor.",
    pros: ["Nonlinear", "No scaling"],
  },
  {
    name: "Random Forest Regression",
    desc: "Ensemble of decision trees.",
    pros: ["High accuracy", "Robust"],
  },
  {
    name: "Support Vector Regression",
    desc: "SVM adapted for regression tasks.",
    pros: ["Kernel tricks", "Robust"],
  },
];

function ModelPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  return (
    <div>
      <WizardSteps />
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {models.map((m) => {
          const selected = state.model === m.name;
          return (
            <Card
              key={m.name}
              variant="glass"
              className={`card-interactive cursor-pointer ${selected ? "border-primary ring-2 ring-primary/30" : ""}`}
              onClick={() => update({ model: m.name })}
            >
              <CardContent className="p-5 space-y-3">
                <div className="flex items-start justify-between">
                  <h3 className="font-semibold">{m.name}</h3>
                  {selected && <Check className="h-4 w-4 text-primary" />}
                </div>
                <p className="text-sm text-muted-foreground">{m.desc}</p>
                <div className="flex flex-wrap gap-1">
                  {m.pros.map((p) => (
                    <Badge key={p} variant="secondary" className="text-xs">
                      {p}
                    </Badge>
                  ))}
                </div>
                <Button
                  size="sm"
                  variant={selected ? "gradient" : "outline"}
                  className="w-full"
                  onClick={(e) => {
                    e.stopPropagation();
                    update({ model: m.name });
                  }}
                >
                  {selected ? "Selected" : "Select Model"}
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>
      <div className="flex justify-end mt-6">
        <Button disabled={!state.model} onClick={() => navigate({ to: "/new/metrics" })}>
          Continue
        </Button>
      </div>
    </div>
  );
}
