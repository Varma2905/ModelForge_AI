import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader2, Plus, Trash2 } from "lucide-react";
import { useCreateDataset } from "@/hooks/use-datasets";
import { ApiError } from "@/lib/api-service";
import type { DatasetSummary } from "@/lib/api-types";
import type { Dataset } from "@/lib/analysis-store";
import { toast } from "sonner";

export function ManualEditorPanel({ onCreated }: { onCreated: (result: DatasetSummary) => void }) {
  const [manual, setManual] = useState<Dataset>({
    name: "manual_dataset",
    columns: ["Area", "Bedrooms", "Age", "Price"],
    rows: [
      [1500, 3, 10, 5000000],
      [2400, 4, 5, 8200000],
    ],
    // Unused for this editor (its own `rows` array is always the full,
    // uncapped dataset) — present only to satisfy the shared Dataset type.
    totalRows: 2,
  });

  const createMutation = useCreateDataset();

  const saveManualDataset = async () => {
    try {
      const result = await createMutation.mutateAsync(manual);
      onCreated(result);
      toast.success("Dataset saved");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to save dataset");
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Spreadsheet Editor</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2 flex-wrap">
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              setManual((m) => ({
                ...m,
                columns: [...m.columns, `col${m.columns.length + 1}`],
                rows: m.rows.map((r) => [...r, 0]),
              }))
            }
          >
            <Plus className="h-3 w-3 mr-1" /> Column
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              setManual((m) => ({
                ...m,
                rows: [...m.rows, new Array(m.columns.length).fill(0)],
              }))
            }
          >
            <Plus className="h-3 w-3 mr-1" /> Row
          </Button>
          <Button size="sm" onClick={saveManualDataset} disabled={createMutation.isPending}>
            {createMutation.isPending ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : null}
            Save dataset
          </Button>
        </div>
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full text-sm">
            <thead className="bg-muted">
              <tr>
                {manual.columns.map((c, i) => (
                  <th key={i} className="p-1">
                    <div className="flex items-center gap-1">
                      <Input
                        value={c}
                        onChange={(e) =>
                          setManual((m) => {
                            const cols = [...m.columns];
                            cols[i] = e.target.value;
                            return { ...m, columns: cols };
                          })
                        }
                        className="h-8"
                      />
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() =>
                          setManual((m) => ({
                            ...m,
                            columns: m.columns.filter((_, x) => x !== i),
                            rows: m.rows.map((r) => r.filter((_, x) => x !== i)),
                          }))
                        }
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {manual.rows.map((r, ri) => (
                <tr key={ri} className="border-t">
                  {r.map((v, ci) => (
                    <td key={ci} className="p-1">
                      <Input
                        value={String(v)}
                        onChange={(e) =>
                          setManual((m) => {
                            const rows = m.rows.map((row) => [...row]);
                            const n = Number(e.target.value);
                            rows[ri][ci] = Number.isFinite(n) ? n : e.target.value;
                            return { ...m, rows };
                          })
                        }
                        className="h-8"
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
