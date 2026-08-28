import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertCircle, Cloud, FileText, Loader2 } from "lucide-react";
import {
  useGoogleDriveAuthUrl,
  useGoogleDriveFiles,
  useImportGoogleDriveFile,
} from "@/hooks/use-datasets";
import { ApiError } from "@/lib/api-service";
import type { DatasetSummary } from "@/lib/api-types";
import { toast } from "sonner";

function formatBytes(bytes?: number): string {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

export function GoogleDrivePanel({
  configured,
  onImported,
}: {
  configured: boolean;
  onImported: (result: DatasetSummary) => void;
}) {
  const [session, setSession] = useState<string | null>(null);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);

  const authUrlMutation = useGoogleDriveAuthUrl();
  const filesQuery = useGoogleDriveFiles(session);
  const importMutation = useImportGoogleDriveFile();

  // Google redirects back to this page with ?gdrive_session=... (success) or
  // ?gdrive_error=... (failure) after the OAuth consent flow completes —
  // pick that up once, then strip it from the URL.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sessionParam = params.get("gdrive_session");
    const errorParam = params.get("gdrive_error");
    if (sessionParam) {
      setSession(sessionParam);
    } else if (errorParam) {
      toast.error(
        errorParam === "auth_failed"
          ? "Google Drive authorization failed. Please try again."
          : "The Google Drive connection expired. Please try again.",
      );
    }
    if (sessionParam || errorParam) {
      window.history.replaceState(null, "", window.location.pathname);
    }
  }, []);

  if (!configured) {
    return (
      <Card>
        <CardContent className="p-6 flex items-start gap-3 text-sm text-muted-foreground">
          <AlertCircle className="h-5 w-5 flex-shrink-0 text-warning" />
          <div>
            <div className="font-medium text-foreground">
              Google Drive integration requires configuration.
            </div>
            <p className="mt-1">
              The server needs a Google OAuth Client ID and Secret (GOOGLE_DRIVE_CLIENT_ID and
              GOOGLE_DRIVE_CLIENT_SECRET environment variables) to connect Drive. Ask your
              administrator to set these, then reload this page.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const handleConnect = async () => {
    try {
      const { auth_url } = await authUrlMutation.mutateAsync();
      window.location.href = auth_url;
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to start Google Drive connection");
    }
  };

  const handleImport = async () => {
    if (!session || !selectedFileId) return;
    const file = filesQuery.data?.files.find((f) => f.id === selectedFileId);
    if (!file) return;
    try {
      const result = await importMutation.mutateAsync({
        session,
        file_id: file.id,
        file_name: file.name,
      });
      onImported(result);
      toast.success("Dataset imported from Google Drive");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to import from Google Drive");
    }
  };

  if (!session) {
    return (
      <Card>
        <CardContent className="p-6 flex flex-col items-center justify-center gap-3 py-10">
          <Cloud className="h-10 w-10 text-muted-foreground" />
          <p className="text-sm text-muted-foreground text-center max-w-sm">
            Connect your Google Drive to pick a CSV, Excel, JSON, or Parquet file.
          </p>
          <Button onClick={handleConnect} disabled={authUrlMutation.isPending}>
            {authUrlMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : null}
            Connect Google Drive
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-6 space-y-3">
        {filesQuery.isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground py-6 justify-center">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading your Drive files…
          </div>
        ) : filesQuery.isError ? (
          <p className="text-sm text-destructive">
            Could not load Google Drive files. Try reconnecting.
          </p>
        ) : filesQuery.data && filesQuery.data.files.length > 0 ? (
          <>
            <div className="rounded-lg border divide-y max-h-56 overflow-y-auto">
              {filesQuery.data.files.map((f) => (
                <label
                  key={f.id}
                  className={`flex items-center justify-between gap-3 p-2.5 text-sm cursor-pointer hover:bg-muted/50 ${
                    selectedFileId === f.id ? "bg-primary/5" : ""
                  }`}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <input
                      type="radio"
                      name="gdrive-file"
                      checked={selectedFileId === f.id}
                      onChange={() => setSelectedFileId(f.id)}
                      className="flex-shrink-0"
                    />
                    <FileText className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                    <span className="truncate">{f.name}</span>
                  </span>
                  <span className="text-xs text-muted-foreground flex-shrink-0">
                    {formatBytes(f.size)}
                  </span>
                </label>
              ))}
            </div>
            <div className="flex justify-end">
              <Button onClick={handleImport} disabled={!selectedFileId || importMutation.isPending}>
                {importMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : null}
                Import selected file
              </Button>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-6">
            No supported files (CSV, Excel, JSON, Parquet) found in your Drive.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
