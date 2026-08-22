import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

const SIZE_CLASSES = {
  sm: "h-8 w-8",
  md: "h-10 w-10",
  lg: "h-12 w-12",
} as const;

const ICON_SIZE_CLASSES = {
  sm: "h-4 w-4",
  md: "h-5 w-5",
  lg: "h-6 w-6",
} as const;

export function GradientIcon({
  icon: Icon,
  size = "md",
  shape = "square",
  className,
}: {
  icon: LucideIcon;
  size?: keyof typeof SIZE_CLASSES;
  shape?: "square" | "circle";
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-center flex-shrink-0 bg-[image:var(--gradient-brand)] text-white shadow-lg shadow-primary/25",
        SIZE_CLASSES[size],
        shape === "circle" ? "rounded-full" : "rounded-lg",
        className,
      )}
    >
      <Icon className={ICON_SIZE_CLASSES[size]} />
    </div>
  );
}
