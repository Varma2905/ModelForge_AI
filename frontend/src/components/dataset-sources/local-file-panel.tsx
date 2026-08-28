import { useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CloudUpload, Loader2 } from "lucide-react";
import { useUploadDataset } from "@/hooks/use-datasets";
import { ApiError } from "@/lib/api-service";
import type { DatasetSummary } from "@/lib/api-types";
import { toast } from "sonner";

const ACCEPT = ".csv,.xlsx,.xls,.json,.parquet";
const SUPPORTED_RE = /\.(csv|xlsx|xls|json|parquet)$/i;
const MAX_UPLOAD_MB = 100;

export function LocalFilePanel({ onUploaded }: { onUploaded: (result: DatasetSummary) => void }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadMutation = useUploadDataset();

  const handleFile = async (file: File) => {
    if (!SUPPORTED_RE.test(file.name)) {
      toast.error("Only CSV, Excel, JSON, and Parquet files are supported.");
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      toast.error(`File exceeds the maximum upload size of ${MAX_UPLOAD_MB}MB.`);
      return;
    }
    try {
      const result = await uploadMutation.mutateAsync(file);
      onUploaded(result);
      toast.success("Dataset uploaded");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to upload dataset");
    }
  };

  return (
    <Card className="h-full">
      <CardContent className="p-6 h-full">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
          }}
          onClick={() => !uploadMutation.isPending && inputRef.current?.click()}
          className={`flex h-full min-h-[280px] flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
            dragOver ? "border-primary bg-primary/5" : "border-muted-foreground/30"
          } ${uploadMutation.isPending ? "pointer-events-none opacity-60" : ""}`}
        >
          {uploadMutation.isPending ? (
            <Loader2 className="h-12 w-12 text-muted-foreground mb-4 animate-spin" />
          ) : (
            <CloudUpload className="h-12 w-12 text-muted-foreground mb-4" />
          )}
          <div className="text-lg font-semibold">
            {uploadMutation.isPending ? "Uploading dataset…" : "Drag & drop your file here"}
          </div>
          {!uploadMutation.isPending && (
            <p className="text-sm text-muted-foreground mt-1">or click to browse</p>
          )}
          <p className="text-xs text-muted-foreground mt-3">
            Supports: CSV, XLSX, JSON, Parquet (max {MAX_UPLOAD_MB}MB)
          </p>
          <Button
            type="button"
            variant="gradient"
            className="mt-5"
            disabled={uploadMutation.isPending}
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
          >
            Browse Files
          </Button>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            disabled={uploadMutation.isPending}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </div>
      </CardContent>
    </Card>
  );
}
