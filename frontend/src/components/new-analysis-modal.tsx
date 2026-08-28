import { createContext, useContext, useState, type ReactNode } from "react";
import { useNavigate } from "@tanstack/react-router";
import { TrendingUp, Target, Network, ArrowRight } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { GradientIcon } from "@/components/gradient-icon";
import { cn } from "@/lib/utils";

type NewAnalysisModalContextValue = { open: boolean; setOpen: (open: boolean) => void };
const NewAnalysisModalContext = createContext<NewAnalysisModalContextValue | undefined>(undefined);

export function NewAnalysisModalProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <NewAnalysisModalContext.Provider value={{ open, setOpen }}>
      {children}
      <NewAnalysisModal open={open} setOpen={setOpen} />
    </NewAnalysisModalContext.Provider>
  );
}

export function useNewAnalysisModal() {
  const ctx = useContext(NewAnalysisModalContext);
  if (!ctx) throw new Error("useNewAnalysisModal must be used within NewAnalysisModalProvider");
  return ctx;
}

const TASKS = [
  {
    id: "regression",
    title: "Regression",
    description: "Predict continuous numerical values",
    examples: "Linear Regression • Random Forest • XGBoost",
    icon: TrendingUp,
    gradient: "from-violet-500 to-purple-600",
    accentBorder: "hover:border-violet-500/50",
    accentShadow: "hover:shadow-xl hover:shadow-violet-500/20",
    accentText: "text-violet-400",
    url: "/new/upload" as const,
  },
  {
    id: "classification",
    title: "Classification",
    description: "Predict categories or classes",
    examples: "Logistic Regression • Decision Tree • Random Forest",
    icon: Target,
    gradient: "from-cyan-500 to-teal-600",
    accentBorder: "hover:border-teal-500/50",
    accentShadow: "hover:shadow-xl hover:shadow-teal-500/20",
    accentText: "text-teal-400",
    url: "/classify/upload" as const,
  },
  {
    id: "clustering",
    title: "Clustering",
    description: "Discover hidden patterns and groups",
    examples: "K-Means • DBSCAN • Hierarchical Clustering",
    icon: Network,
    gradient: "from-amber-500 to-orange-600",
    accentBorder: "hover:border-amber-500/50",
    accentShadow: "hover:shadow-xl hover:shadow-amber-500/20",
    accentText: "text-amber-400",
    url: "/cluster/upload" as const,
  },
];

function NewAnalysisModal({ open, setOpen }: { open: boolean; setOpen: (v: boolean) => void }) {
  const navigate = useNavigate();

  const select = (url: (typeof TASKS)[number]["url"]) => {
    setOpen(false);
    navigate({ to: url });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-3xl bg-card/80 backdrop-blur-2xl border-white/10 shadow-2xl shadow-black/50 p-8">
        <DialogHeader className="sm:text-center">
          <DialogTitle className="text-2xl font-bold tracking-tight">Create New Analysis</DialogTitle>
          <DialogDescription className="text-sm">
            Choose a machine learning task to get started
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 sm:grid-cols-3 mt-2">
          {TASKS.map((task) => (
            <button
              key={task.id}
              type="button"
              onClick={() => select(task.url)}
              className={cn(
                "group flex flex-col items-start text-left rounded-xl border border-border/60 bg-card/40 backdrop-blur-md p-5 transition-all duration-200 cursor-pointer hover:-translate-y-1",
                task.accentBorder,
                task.accentShadow,
              )}
            >
              <GradientIcon icon={task.icon} size="lg" className={`bg-gradient-to-br ${task.gradient}`} />
              <h3 className="mt-4 font-semibold text-base">{task.title}</h3>
              <p className="mt-1.5 text-sm text-muted-foreground leading-snug">{task.description}</p>
              <p className="mt-2 text-xs text-muted-foreground/70 leading-snug">{task.examples}</p>
              <span className={cn("mt-4 inline-flex items-center gap-1 text-xs font-semibold", task.accentText)}>
                Start Analysis
                <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
              </span>
            </button>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
