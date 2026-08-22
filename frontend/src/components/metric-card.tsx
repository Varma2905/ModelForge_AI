import { Card, CardContent } from "@/components/ui/card";
import type { LucideIcon } from "lucide-react";

export function MetricCard({
  label,
  value,
  icon: Icon,
  gradient = "from-indigo-500 to-blue-500",
}: {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  gradient?: string;
}) {
  return (
    <Card className={`bg-gradient-to-br ${gradient} text-white border-0`}>
      <CardContent className="p-4 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs text-white/80">{label}</div>
          <div className="text-2xl font-bold mt-1 truncate">{value}</div>
        </div>
        {Icon && (
          <div className="h-10 w-10 rounded-lg bg-white/15 flex items-center justify-center flex-shrink-0">
            <Icon className="h-5 w-5" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
