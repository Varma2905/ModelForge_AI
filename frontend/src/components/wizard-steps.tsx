import { Link, useRouterState } from "@tanstack/react-router";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export type WizardStep = { label: string; url: string };

const regressionSteps: WizardStep[] = [
  { label: "Dataset", url: "/new/upload" },
  { label: "Variables", url: "/new/variables" },
  { label: "Preprocessing", url: "/new/preprocessing" },
  { label: "Split", url: "/new/split" },
  { label: "Model", url: "/new/model" },
  { label: "Train", url: "/new/train" },
  { label: "Predict", url: "/new/predict" },
  { label: "Metrics", url: "/new/metrics" },
  { label: "Visualize", url: "/new/visualize" },
  { label: "AI Insights", url: "/new/explain" },
  { label: "Report", url: "/new/report" },
];

export const classificationSteps: WizardStep[] = [
  { label: "Dataset", url: "/classify/upload" },
  { label: "Variables", url: "/classify/variables" },
  { label: "Preprocessing", url: "/classify/preprocessing" },
  { label: "Split", url: "/classify/split" },
  { label: "Model", url: "/classify/model" },
  { label: "Train", url: "/classify/train" },
  { label: "Predict", url: "/classify/predict" },
  { label: "Metrics", url: "/classify/metrics" },
  { label: "Visualize", url: "/classify/visualize" },
  { label: "AI Insights", url: "/classify/explain" },
  { label: "Report", url: "/classify/report" },
];

// Clustering is unsupervised — no Target/Split/Predict steps exist (see
// cluster.*.tsx routes): Dataset -> Features -> Algorithm -> Clustering ->
// Analysis -> Insights -> Report.
export const clusteringSteps: WizardStep[] = [
  { label: "Dataset", url: "/cluster/upload" },
  { label: "Features", url: "/cluster/variables" },
  { label: "Algorithm", url: "/cluster/model" },
  { label: "Clustering", url: "/cluster/train" },
  { label: "Analysis", url: "/cluster/analysis" },
  { label: "Insights", url: "/cluster/explain" },
  { label: "Report", url: "/cluster/report" },
];

export function WizardSteps({ steps = regressionSteps }: { steps?: WizardStep[] }) {
  const pathname = useRouterState({ select: (r) => r.location.pathname });
  const currentIdx = Math.max(0, steps.findIndex((s) => s.url === pathname));
  const current = steps[currentIdx];

  return (
    <div className="mb-6">
      {/* Compact indicator — small screens only. Avoids the full horizontal
          strip (which needs its own scroll container to fit 11 steps) ever
          being the thing a narrow viewport has to deal with. */}
      <div className="md:hidden rounded-xl border bg-card p-3">
        <div className="flex items-center justify-between text-xs font-medium mb-2">
          <span className="text-muted-foreground">
            Step {currentIdx + 1} of {steps.length}
          </span>
          <span className="text-foreground">{current.label}</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
          <div
            className="h-full rounded-full bg-[image:var(--gradient-brand)] transition-[width] duration-300 ease-out"
            style={{ width: `${((currentIdx + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Full horizontal nav — scoped overflow-x-auto keeps any overflow
          contained to this strip, never the page. */}
      <div className="hidden md:block overflow-x-auto rounded-xl border bg-card p-3">
        <ol className="flex items-center gap-1 min-w-max">
          {steps.map((s, i) => {
            const done = i < currentIdx;
            const active = i === currentIdx;
            return (
              <li key={s.url} className="flex items-center gap-1">
                <Link
                  to={s.url}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-all duration-200 ease-out",
                    active &&
                      "bg-[image:var(--gradient-brand)] text-white shadow-[0_0_12px_oklch(0.606_0.219_292.717_/_0.4)]",
                    done && "text-foreground hover:bg-accent",
                    !active && !done && "text-muted-foreground hover:bg-accent",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold transition-colors duration-200",
                      active && "bg-white/25 text-white",
                      done && "bg-success text-success-foreground",
                      !active && !done && "bg-muted text-muted-foreground",
                    )}
                  >
                    {done ? <Check className="h-3 w-3" /> : i + 1}
                  </span>
                  {s.label}
                </Link>
                {i < steps.length - 1 && <span className="text-muted-foreground/40">›</span>}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
