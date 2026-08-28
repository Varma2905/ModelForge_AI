import { formatDistanceToNow } from "date-fns";
import { Bell, Database, Brain, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useDashboardSummary } from "@/hooks/use-dashboard";
import type { DashboardActivityItem } from "@/lib/api-types";

const ACTIVITY_ICON: Record<DashboardActivityItem["type"], typeof Database> = {
  dataset: Database,
  analysis: Brain,
  report: FileText,
};

export function NotificationsBell() {
  const { data } = useDashboardSummary();
  const activity = data?.recent_activity ?? [];

  // "Unread" = happened in the last 24h — a real, derivable signal rather
  // than a fabricated counter.
  const unreadCount = activity.filter(
    (a) => Date.now() - new Date(a.timestamp).getTime() < 24 * 60 * 60 * 1000,
  ).length;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className="relative" aria-label="Recent activity">
          <Bell className="h-4 w-4" />
          {unreadCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold text-primary-foreground">
              {unreadCount}
            </span>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-0">
        <div className="border-b border-border px-4 py-2.5 text-sm font-semibold">Recent Activity</div>
        {activity.length === 0 ? (
          <p className="px-4 py-6 text-center text-sm text-muted-foreground">Nothing yet.</p>
        ) : (
          <div className="max-h-80 overflow-y-auto py-1">
            {activity.slice(0, 6).map((item, i) => {
              const Icon = ACTIVITY_ICON[item.type];
              return (
                <div key={`${item.type}-${item.timestamp}-${i}`} className="flex items-start gap-2.5 px-4 py-2">
                  <Icon className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="text-xs">
                      <span className="font-medium">{item.action}</span>{" "}
                      <span className="text-muted-foreground">— {item.subtitle}</span>
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      {formatDistanceToNow(new Date(item.timestamp), { addSuffix: true })}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
