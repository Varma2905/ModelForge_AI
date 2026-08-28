import { useChartTheme } from "@/lib/chart-theme";
import type { ClassificationChartData } from "@/lib/api-types";

function confusionCellColor(value: number, max: number, isDark: boolean) {
  const intensity = max > 0 ? value / max : 0;
  const base = isDark ? [30, 41, 59] : [248, 250, 252];
  const target = isDark ? [96, 165, 250] : [59, 130, 246];
  const rgb = base.map((b, i) => Math.round(b + (target[i] - b) * intensity));
  return `rgb(${rgb.join(",")})`;
}

export function ConfusionMatrix({ data }: { data: ClassificationChartData["confusion_matrix"] | undefined }) {
  const t = useChartTheme();
  if (!data || data.classes.length === 0) {
    return <p className="text-sm text-muted-foreground">No confusion matrix data available.</p>;
  }
  const max = Math.max(...data.matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-xs">
        <thead>
          <tr>
            <th className="p-1" />
            <th className="p-1 font-medium text-muted-foreground" colSpan={data.classes.length}>
              Predicted
            </th>
          </tr>
          <tr>
            <th className="p-1" />
            {data.classes.map((c) => (
              <th key={c} className="p-1 font-medium text-muted-foreground">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.classes.map((rowLabel, i) => (
            <tr key={rowLabel}>
              <th className="p-1 text-right font-medium text-muted-foreground pr-2 whitespace-nowrap">
                {rowLabel}
              </th>
              {data.matrix[i].map((val, j) => (
                <td key={j} className="p-0.5">
                  <div
                    className="flex items-center justify-center h-12 w-16 rounded-sm"
                    style={{ backgroundColor: confusionCellColor(val, max, t.isDark) }}
                    title={`Actual ${rowLabel}, Predicted ${data.classes[j]}: ${val}`}
                  >
                    <span
                      className="text-[12px] font-semibold"
                      style={{ color: max > 0 && val / max > 0.5 ? "#fff" : t.foreground }}
                    >
                      {val}
                    </span>
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-muted-foreground mt-2">
        Rows = actual class, columns = predicted class.
      </p>
    </div>
  );
}
