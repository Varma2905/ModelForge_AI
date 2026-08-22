import { useChartTheme } from "@/lib/chart-theme";
import type { ChartData } from "@/lib/api-types";

function correlationColor(value: number, isDark: boolean) {
  const clamped = Math.max(-1, Math.min(1, value));
  const intensity = Math.abs(clamped);
  const positive = isDark ? [52, 211, 153] : [16, 185, 129];
  const negative = isDark ? [244, 114, 182] : [236, 72, 153];
  const base = isDark ? [30, 41, 59] : [248, 250, 252];
  const target = clamped >= 0 ? positive : negative;
  const rgb = base.map((b, i) => Math.round(b + (target[i] - b) * intensity));
  return `rgb(${rgb.join(",")})`;
}

export function CorrelationMatrix({ data }: { data: ChartData["correlation_matrix"] | undefined }) {
  const t = useChartTheme();

  if (!data || data.columns.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Not enough numeric columns to compute correlations.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-xs">
        <thead>
          <tr>
            <th className="p-1" />
            {data.columns.map((c) => (
              <th key={c} className="p-1 font-medium text-muted-foreground">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.columns.map((rowLabel, i) => (
            <tr key={rowLabel}>
              <th className="p-1 text-right font-medium text-muted-foreground pr-2 whitespace-nowrap">
                {rowLabel}
              </th>
              {data.matrix[i].map((val, j) => (
                <td key={j} className="p-0.5">
                  <div
                    className="flex items-center justify-center h-10 w-16 rounded-sm"
                    style={{ backgroundColor: correlationColor(val, t.isDark) }}
                    title={`${rowLabel} vs ${data.columns[j]}: ${val.toFixed(2)}`}
                  >
                    <span
                      className="text-[11px] font-medium"
                      style={{ color: Math.abs(val) > 0.5 ? "#fff" : t.foreground }}
                    >
                      {val.toFixed(2)}
                    </span>
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
