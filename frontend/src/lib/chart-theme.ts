import { useMemo } from "react";
import { useTheme } from "@/lib/theme";

export type ChartTheme = {
  isDark: boolean;
  foreground: string;
  mutedForeground: string;
  border: string;
  background: string;
  card: string;
  grid: string;
  tooltipStyle: React.CSSProperties;
  tooltipLabelStyle: React.CSSProperties;
  tooltipItemStyle: React.CSSProperties;
  legendStyle: React.CSSProperties;
  axisTick: { fill: string; fontSize: number };
  series: {
    primary: string;
    accent: string;
    success: string;
    warning: string;
    danger: string;
  };
};

function readVar(name: string, fallback: string) {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function useChartTheme(): ChartTheme {
  const { theme } = useTheme();
  return useMemo(() => {
    const isDark = theme === "dark";
    const foreground = readVar("--foreground", isDark ? "#f8fafc" : "#0f172a");
    const mutedForeground = readVar("--muted-foreground", isDark ? "#94a3b8" : "#64748b");
    const border = readVar("--border", isDark ? "#334155" : "#e2e8f0");
    const background = readVar("--background", isDark ? "#0f172a" : "#ffffff");
    const card = readVar("--card", isDark ? "#1e293b" : "#ffffff");

    // Series palette tuned to work on both light and dark backgrounds.
    const series = isDark
      ? {
          primary: "#818cf8",
          accent: "#22d3ee",
          success: "#34d399",
          warning: "#fbbf24",
          danger: "#f472b6",
        }
      : {
          primary: "#6366f1",
          accent: "#0891b2",
          success: "#10b981",
          warning: "#f59e0b",
          danger: "#ec4899",
        };

    return {
      isDark,
      foreground,
      mutedForeground,
      border,
      background,
      card,
      grid: border,
      tooltipStyle: {
        backgroundColor: card,
        border: `1px solid ${border}`,
        borderRadius: 8,
        color: foreground,
        boxShadow: isDark ? "0 4px 12px rgba(0,0,0,0.5)" : "0 4px 12px rgba(0,0,0,0.08)",
      },
      tooltipLabelStyle: { color: foreground, fontWeight: 600 },
      tooltipItemStyle: { color: foreground },
      legendStyle: { color: foreground, fontSize: 12 },
      axisTick: { fill: mutedForeground, fontSize: 11 },
      series,
    };
  }, [theme]);
}
