import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Database, FileText, Loader2, ShieldAlert } from "lucide-react";
import {
  useConnectKaggle,
  useDisconnectKaggle,
  useImportKaggleDataset,
  useKaggleStatus,
  useResolveKaggleDataset,
} from "@/hooks/use-datasets";
import { ApiError } from "@/lib/api-service";
import type { DatasetSummary, KaggleFile } from "@/lib/api-types";
import { toast } from "sonner";

function formatBytes(bytes: number): string {
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

export function KaggleImportPanel({ onImported }: { onImported: (result: DatasetSummary) => void }) {
  const status = useKaggleStatus();

  if (status.isLoading) {
    return (
      <Card>
        <CardContent className="p-6 text-sm text-muted-foreground">Checking Kaggle connection…</CardContent>
      </Card>
    );
  }

  if (!status.data?.connected) {
    return <KaggleConnectForm onConnected={() => status.refetch()} />;
  }

  return (
    <KaggleBrowsePanel
      kaggleUsername={status.data.kaggle_username}
      onImported={onImported}
      onDisconnected={() => status.refetch()}
    />
  );
}

function KaggleConnectForm({ onConnected }: { onConnected: () => void }) {
  const [username, setUsername] = useState("");
  const [key, setKey] = useState("");
  const connectMutation = useConnectKaggle();

  const handleConnect = async () => {
    if (!username.trim() || !key.trim()) return;
    try {
      await connectMutation.mutateAsync({ username: username.trim(), key: key.trim() });
      toast.success("Kaggle account connected");
      onConnected();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to connect Kaggle account");
    }
  };

  return (
    <Card>
      <CardContent className="p-6 space-y-4">
        <div className="flex items-start gap-3 text-sm text-muted-foreground">
          <ShieldAlert className="h-5 w-5 flex-shrink-0 text-warning" />
          <div>
            <div className="font-medium text-foreground">Connect your Kaggle account</div>
            <p className="mt-1">
              Kaggle doesn&apos;t offer a redirect-based sign-in for third-party apps — instead, paste your own
              personal API key from{" "}
              <a
                href="https://www.kaggle.com/settings"
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-2"
              >
                kaggle.com/settings → API → Create New Token
              </a>
              . It&apos;s encrypted and stored only against your account — never shared with other users, and
              never sent back to your browser after this step.
            </p>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="kaggle-username">Kaggle username</Label>
            <Input
              id="kaggle-username"
              placeholder="your-kaggle-username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={connectMutation.isPending}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="kaggle-key">API key</Label>
            <Input
              id="kaggle-key"
              type="password"
              placeholder="Paste your Kaggle API key"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              disabled={connectMutation.isPending}
            />
          </div>
        </div>

        <div className="flex justify-end">
          <Button onClick={handleConnect} disabled={!username.trim() || !key.trim() || connectMutation.isPending}>
            {connectMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : null}
            Connect Kaggle
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function KaggleBrowsePanel({
  kaggleUsername,
  onImported,
  onDisconnected,
}: {
  kaggleUsername: string | null;
  onImported: (result: DatasetSummary) => void;
  onDisconnected: () => void;
}) {
  const [ref, setRef] = useState("");
  const [resolvedRef, setResolvedRef] = useState<string | null>(null);
  const [files, setFiles] = useState<KaggleFile[] | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const resolveMutation = useResolveKaggleDataset();
  const importMutation = useImportKaggleDataset();
  const disconnectMutation = useDisconnectKaggle();

  const handleResolve = async () => {
    if (!ref.trim()) return;
    setFiles(null);
    setSelectedFile(null);
    try {
      const result = await resolveMutation.mutateAsync({ dataset_ref: ref.trim() });
      setResolvedRef(result.dataset_ref);
      setFiles(result.files);
      if (result.files.length === 1) setSelectedFile(result.files[0].name);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to resolve Kaggle dataset");
    }
  };

  const handleImport = async () => {
    if (!resolvedRef || !selectedFile) return;
    try {
      const result = await importMutation.mutateAsync({
        dataset_ref: resolvedRef,
        file_name: selectedFile,
      });
      onImported(result);
      toast.success("Dataset imported from Kaggle");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to import Kaggle dataset");
    }
  };

  const handleDisconnect = async () => {
    try {
      await disconnectMutation.mutateAsync();
      toast.success("Kaggle account disconnected");
      onDisconnected();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to disconnect Kaggle account");
    }
  };

  return (
    <Card>
      <CardContent className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <Badge variant="secondary" className="gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-success" />
            Connected as {kaggleUsername}
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDisconnect}
            disabled={disconnectMutation.isPending}
          >
            {disconnectMutation.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : null}
            Disconnect
          </Button>
        </div>

        <div className="space-y-2">
          <Label htmlFor="kaggle-ref">Kaggle dataset URL or identifier</Label>
          <div className="flex gap-2">
            <Input
              id="kaggle-ref"
              placeholder="https://www.kaggle.com/datasets/owner/dataset"
              value={ref}
              onChange={(e) => setRef(e.target.value)}
              disabled={resolveMutation.isPending}
            />
            <Button onClick={handleResolve} disabled={!ref.trim() || resolveMutation.isPending}>
              {resolveMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : null}
              Resolve
            </Button>
          </div>
        </div>

        {files && files.length > 0 && (
          <div className="space-y-2">
            <Label className="flex items-center gap-1.5">
              <Database className="h-3.5 w-3.5" />
              {files.length > 1 ? `Choose a file (${files.length} available)` : "File"}
            </Label>
            <div className="rounded-lg border divide-y max-h-56 overflow-y-auto">
              {files.map((f) => (
                <label
                  key={f.name}
                  className={`flex items-center justify-between gap-3 p-2.5 text-sm cursor-pointer hover:bg-muted/50 ${
                    selectedFile === f.name ? "bg-primary/5" : ""
                  }`}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <input
                      type="radio"
                      name="kaggle-file"
                      checked={selectedFile === f.name}
                      onChange={() => setSelectedFile(f.name)}
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
              <Button onClick={handleImport} disabled={!selectedFile || importMutation.isPending}>
                {importMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : null}
                Import selected file
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
