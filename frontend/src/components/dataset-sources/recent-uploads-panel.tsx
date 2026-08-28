import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Database, Eye, Trash2 } from "lucide-react";
import { useDatasetsList, useDeleteDataset } from "@/hooks/use-datasets";
import { ApiError } from "@/lib/api-service";
import type { DatasetListItem } from "@/lib/api-types";
import { toast } from "sonner";

const MAX_SHOWN = 5;

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unitIdx = 0;
  while (value >= 1024 && unitIdx < units.length - 1) {
    value /= 1024;
    unitIdx += 1;
  }
  return `${value.toFixed(1)} ${units[unitIdx]}`;
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function RecentUploadsPanel({
  onSelect,
  onViewAll,
}: {
  onSelect: (item: DatasetListItem) => void;
  onViewAll?: () => void;
}) {
  const { data, isLoading, isError } = useDatasetsList();
  const deleteMutation = useDeleteDataset();

  const items = (data ?? []).slice(0, MAX_SHOWN);

  const handleDelete = async (item: DatasetListItem) => {
    try {
      await deleteMutation.mutateAsync(item.dataset_id);
      toast.success(`Deleted ${item.name}`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to delete dataset");
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">Recent Uploads</CardTitle>
        {onViewAll && items.length > 0 && (
          <Button variant="link" size="sm" className="h-auto p-0 text-xs" onClick={onViewAll}>
            View all
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : isError ? (
          <p className="text-sm text-muted-foreground">Couldn't load recent uploads.</p>
        ) : items.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Database className="h-8 w-8 text-muted-foreground/50 mb-2" />
            <p className="text-sm text-muted-foreground">No datasets uploaded yet.</p>
          </div>
        ) : (
          <div className="space-y-1">
            {items.map((item) => (
              <div
                key={item.dataset_id}
                className="flex items-center gap-3 rounded-lg p-2 -mx-2 hover:bg-accent transition-colors"
              >
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium truncate">{item.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {item.rows.toLocaleString()} rows · {item.columns} cols · {formatBytes(item.size_bytes)}
                    {" · "}
                    {formatDate(item.created_at)}
                  </p>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    aria-label={`View ${item.name}`}
                    onClick={() => onSelect(item)}
                  >
                    <Eye className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-destructive hover:text-destructive"
                    aria-label={`Delete ${item.name}`}
                    disabled={deleteMutation.isPending}
                    onClick={() => handleDelete(item)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
