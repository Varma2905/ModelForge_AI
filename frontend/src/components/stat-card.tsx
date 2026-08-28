import { Card, CardContent } from "@/components/ui/card";
import type { LucideIcon } from "lucide-react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

export function StatCard({
  label,
  value,
  icon: Icon,
  gradient = "from-indigo-500 to-blue-500",
  /** Percent change vs. the prior period. Omit rather than fabricate one. */
  trendPct,
  /** Use for metrics (like R²) where a raw point delta reads better than a %. */
  trendAbsolute,
  trendSuffix = "this month",
}: {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  gradient?: string;
  trendPct?: number | null;
  trendAbsolute?: number | null;
  trendSuffix?: string;
}) {
  const hasPct = typeof trendPct === "number";
  const hasAbsolute = typeof trendAbsolute === "number";
  const trendValue = hasPct ? trendPct : hasAbsolute ? trendAbsolute : null;
  const isUp = typeof trendValue === "number" && trendValue >= 0;

  return (
    <Card className="transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl">
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          {Icon && (
            <div
              className={cn(
                "h-10 w-10 rounded-lg bg-gradient-to-br flex items-center justify-center flex-shrink-0 text-white shadow-md",
                gradient,
              )}
            >
              <Icon className="h-5 w-5" />
            </div>
          )}
          <div className="min-w-0 flex-1">
            <div className="text-xs text-muted-foreground">{label}</div>
            <div className="text-2xl font-bold mt-0.5 truncate">{value}</div>
          </div>
        </div>
        {trendValue !== null && (
          <div
            className={cn(
              "mt-2 flex items-center gap-1 text-xs font-medium",
              isUp ? "text-emerald-500" : "text-rose-500",
            )}
          >
            {isUp ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
            <span>
              {isUp ? "+" : ""}
              {hasPct ? `${trendValue}%` : trendValue.toFixed(2)} {trendSuffix}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
