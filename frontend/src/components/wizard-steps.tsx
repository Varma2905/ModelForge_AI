import { Link, useRouterState } from "@tanstack/react-router";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

const steps = [
  { label: "Dataset", url: "/new/upload" },
  { label: "Variables", url: "/new/variables" },
  { label: "Preprocessing", url: "/new/preprocessing" },
  { label: "Split", url: "/new/split" },
  { label: "Model", url: "/new/model" },
  { label: "Metrics", url: "/new/metrics" },
  { label: "Train", url: "/new/train" },
  { label: "Predict", url: "/new/predict" },
  { label: "Visualize", url: "/new/visualize" },
  { label: "AI Insights", url: "/new/explain" },
  { label: "Report", url: "/new/report" },
];

export function WizardSteps() {
  const pathname = useRouterState({ select: (r) => r.location.pathname });
  const currentIdx = steps.findIndex((s) => s.url === pathname);
  return (
    <div className="mb-6 overflow-x-auto rounded-xl border bg-card p-3">
      <ol className="flex items-center gap-1 min-w-max">
        {steps.map((s, i) => {
          const done = i < currentIdx;
          const active = i === currentIdx;
          return (
            <li key={s.url} className="flex items-center gap-1">
              <Link
                to={s.url}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition",
                  active &&
                    "bg-[image:var(--gradient-brand)] text-white shadow-[0_0_12px_oklch(0.606_0.219_292.717_/_0.4)]",
                  done && "text-foreground hover:bg-accent",
                  !active && !done && "text-muted-foreground hover:bg-accent",
                )}
              >
                <span
                  className={cn(
                    "flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold",
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
  );
}
