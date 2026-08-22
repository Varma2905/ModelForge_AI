import { Link } from "@tanstack/react-router";
import { AlertCircle, type LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  variant = "empty",
  className,
}: {
  icon: LucideIcon;
  title: string;
  description?: string;
  action?: { label: string; to: string };
  variant?: "empty" | "error";
  className?: string;
}) {
  const isError = variant === "error";
  const DisplayIcon = isError ? AlertCircle : Icon;

  return (
    <div className={cn("flex flex-col items-center justify-center gap-3 py-12 text-center", className)}>
      <div
        className={cn(
          "flex h-12 w-12 items-center justify-center rounded-full",
          isError ? "bg-destructive/10 text-destructive" : "bg-muted text-muted-foreground",
        )}
      >
        <DisplayIcon className="h-6 w-6" />
      </div>
      <div className="space-y-1">
        <p className={cn("text-sm font-medium", isError ? "text-destructive" : "text-foreground")}>
          {title}
        </p>
        {description && <p className="text-xs text-muted-foreground max-w-sm">{description}</p>}
      </div>
      {action && (
        <Button asChild size="sm" variant="outline" className="mt-1">
          <Link to={action.to}>{action.label}</Link>
        </Button>
      )}
    </div>
  );
}
