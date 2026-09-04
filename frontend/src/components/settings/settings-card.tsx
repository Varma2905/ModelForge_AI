import type { ReactNode } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";

// Shared "one settings section = one card, with Save Changes / Reset to
// Defaults in the footer" shell — see settings spec section 7. Subtle
// fade-in/slide-up entrance (150-250ms) per section 12; respects
// prefers-reduced-motion for free via the app-wide override in styles.css.
export function SettingsCard({
  title,
  description,
  onSave,
  onReset,
  children,
}: {
  title: string;
  description?: string;
  onSave: () => void;
  onReset?: () => void;
  children: ReactNode;
}) {
  return (
    <Card className="animate-in fade-in slide-in-from-bottom-2 duration-200">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="space-y-5">{children}</CardContent>
      <div className="flex items-center justify-end gap-2 px-6 pb-6">
        {onReset && (
          <Button variant="outline" size="sm" onClick={onReset} type="button">
            <RotateCcw className="h-3.5 w-3.5 mr-1.5" /> Reset to Defaults
          </Button>
        )}
        <Button size="sm" onClick={onSave} type="button">
          Save Changes
        </Button>
      </div>
    </Card>
  );
}

// A single "Label + description on the left, control on the right" row —
// wraps at narrow widths instead of overflowing (section 11).
export function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 flex-wrap rounded-md border p-3">
      <div className="min-w-0 max-w-md">
        <p className="text-sm font-medium">{label}</p>
        {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}
