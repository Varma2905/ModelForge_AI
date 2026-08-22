import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/lib/theme";
import { getApiBaseUrl, setApiBaseUrl } from "@/lib/api-service";
import { requireAuth } from "@/lib/require-auth";
import { toast } from "sonner";

export const Route = createFileRoute("/settings")({
  beforeLoad: requireAuth,
  component: SettingsPage,
});

function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";
  const [apiUrl, setApiUrl] = useState(getApiBaseUrl());

  const saveApiUrl = () => {
    const trimmed = apiUrl.trim();
    setApiBaseUrl(trimmed);
    setApiUrl(trimmed);
    toast.success("API base URL saved");
  };

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-muted-foreground">Configure the studio to your preference.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Appearance</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isDark ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            <div>
              <Label>Dark Mode</Label>
              <p className="text-xs text-muted-foreground">
                Switch between light and dark theme. Your choice is saved on this device.
              </p>
            </div>
          </div>
          <Switch
            checked={isDark}
            onCheckedChange={(checked) => setTheme(checked ? "dark" : "light")}
            aria-label="Toggle dark mode"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Backend</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <Label>FastAPI URL</Label>
            <Input
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Only change this if your backend runs somewhere other than the default. LLM API keys
              are configured server-side and aren't set from the browser.
            </p>
          </div>
          <Button onClick={saveApiUrl}>Save</Button>
        </CardContent>
      </Card>
    </div>
  );
}
