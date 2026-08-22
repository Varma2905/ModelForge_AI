import { Card, CardContent } from "@/components/ui/card";
import type { LucideIcon } from "lucide-react";
import { Line, LineChart, ResponsiveContainer } from "recharts";

export function MetricCard({
  label,
  value,
  icon: Icon,
  gradient = "from-indigo-500 to-blue-500",
  sparklineData,
}: {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  gradient?: string;
  /** Real historical values only — omit rather than fabricate a trend. */
  sparklineData?: number[];
}) {
  return (
    <Card
      className={`bg-gradient-to-br ${gradient} text-white border-0 shadow-lg transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl`}
    >
      <CardContent className="p-4 flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-xs text-white/80">{label}</div>
          <div className="text-2xl font-bold mt-1 truncate">{value}</div>
          {sparklineData && sparklineData.length > 1 && (
            <div className="h-6 w-full mt-1.5">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sparklineData.map((v) => ({ v }))}>
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke="rgba(255,255,255,0.9)"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
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
