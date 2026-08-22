import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { WizardSteps } from "@/components/wizard-steps";
import { useAnalysis } from "@/lib/analysis-store";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/new/metrics")({
  beforeLoad: requireAuth,
  component: MetricsPage,
});

const errorMetrics = ["MSE", "MAE", "RMSE"];
const statMetrics = ["R2", "Adjusted R2", "P-value", "T-test", "F-test"];

function MetricsPage() {
  const { state, update } = useAnalysis();
  const navigate = useNavigate();
  const toggle = (m: string) => {
    const has = state.metrics.includes(m);
    update({ metrics: has ? state.metrics.filter((x) => x !== m) : [...state.metrics, m] });
  };
  return (
    <div>
      <WizardSteps />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Cost / Error Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {errorMetrics.map((m) => (
              <label
                key={m}
                className="flex items-center gap-3 rounded-lg border p-3 hover:bg-accent cursor-pointer"
              >
                <Checkbox checked={state.metrics.includes(m)} onCheckedChange={() => toggle(m)} />
                <span>{m}</span>
              </label>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Regression Statistics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {statMetrics.map((m) => (
              <label
                key={m}
                className="flex items-center gap-3 rounded-lg border p-3 hover:bg-accent cursor-pointer"
              >
                <Checkbox checked={state.metrics.includes(m)} onCheckedChange={() => toggle(m)} />
                <span>{m}</span>
              </label>
            ))}
          </CardContent>
        </Card>
      </div>
      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/new/train" })}>Train Model</Button>
      </div>
    </div>
  );
}
