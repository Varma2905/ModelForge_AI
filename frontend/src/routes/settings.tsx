import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Moon,
  Sun,
  Palette,
  Sparkles,
  Database,
  UserCircle,
  LogOut,
  Trash2,
  SlidersHorizontal,
  Boxes,
  Gauge,
  LineChart,
  FileText,
} from "lucide-react";
import { useTheme } from "@/lib/theme";
import { useAuth } from "@/lib/auth-context";
import { getStoredUser } from "@/lib/auth-store";
import { useDashboardSummary } from "@/hooks/use-dashboard";
import { requireAuth } from "@/lib/require-auth";
import { useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";
import { useModelForgeSettings } from "@/lib/settings-store";
import { MlPreferencesTab } from "@/components/settings/ml-preferences-tab";
import { ModelPreferencesTab } from "@/components/settings/model-preferences-tab";
import { MetricsTab } from "@/components/settings/metrics-tab";
import { VisualizationTab } from "@/components/settings/visualization-tab";
import { ReportPreferencesTab } from "@/components/settings/report-preferences-tab";

export const Route = createFileRoute("/settings")({
  beforeLoad: requireAuth,
  component: SettingsPage,
});

const TABS = [
  { value: "appearance", label: "Appearance", icon: Palette },
  { value: "ml-preferences", label: "ML Preferences", icon: SlidersHorizontal },
  { value: "model-preferences", label: "Model Preferences", icon: Boxes },
  { value: "metrics", label: "Metrics", icon: Gauge },
  { value: "visualization", label: "Visualization", icon: LineChart },
  { value: "report-preferences", label: "Report Preferences", icon: FileText },
  { value: "ai", label: "AI Settings", icon: Sparkles },
  { value: "data", label: "Data Overview", icon: Database },
  { value: "account", label: "Account", icon: UserCircle },
];

function SettingsPage() {
  const { settings, updateSection } = useModelForgeSettings();

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-muted-foreground">Configure ModelForge AI Studio to your preference.</p>
      </div>

      <Tabs defaultValue="appearance" orientation="vertical" className="flex flex-col sm:flex-row gap-6">
        <TabsList className="flex-col h-auto w-full sm:w-44 items-stretch justify-start bg-transparent p-0 gap-1 flex-shrink-0">
          {TABS.map((t) => (
            <TabsTrigger
              key={t.value}
              value={t.value}
              className="justify-start gap-2 px-3 py-2 data-[state=active]:bg-accent data-[state=active]:shadow-none"
            >
              <t.icon className="h-4 w-4" /> {t.label}
            </TabsTrigger>
          ))}
        </TabsList>

        <div className="flex-1 min-w-0">
          <TabsContent value="appearance" className="mt-0">
            <AppearanceTab />
          </TabsContent>
          <TabsContent value="ml-preferences" className="mt-0">
            <MlPreferencesTab
              value={settings.mlPreferences}
              onSave={(next) => updateSection("mlPreferences", next)}
            />
          </TabsContent>
          <TabsContent value="model-preferences" className="mt-0">
            <ModelPreferencesTab
              value={settings.modelPreferences}
              onSave={(next) => updateSection("modelPreferences", next)}
            />
          </TabsContent>
          <TabsContent value="metrics" className="mt-0">
            <MetricsTab value={settings.metrics} onSave={(next) => updateSection("metrics", next)} />
          </TabsContent>
          <TabsContent value="visualization" className="mt-0">
            <VisualizationTab
              value={settings.visualization}
              onSave={(next) => updateSection("visualization", next)}
            />
          </TabsContent>
          <TabsContent value="report-preferences" className="mt-0">
            <ReportPreferencesTab
              value={settings.reportPreferences}
              onSave={(next) => updateSection("reportPreferences", next)}
            />
          </TabsContent>
          <TabsContent value="ai" className="mt-0">
            <AiSettingsTab />
          </TabsContent>
          <TabsContent value="data" className="mt-0">
            <DataOverviewTab />
          </TabsContent>
          <TabsContent value="account" className="mt-0">
            <AccountTab />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

function AppearanceTab() {
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";
  return (
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
  );
}

function AiSettingsTab() {
  const handleClearChatHistory = () => {
    const userId = getStoredUser()?.id ?? "anon";
    localStorage.removeItem(`regression-studio:ai-chat:assistant:${userId}`);
    toast.success("AI Assistant chat history cleared");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          The AI provider and model are configured by your administrator on the backend and
          aren't editable from the browser.
        </p>
        <div className="flex items-center justify-between rounded-md border p-3">
          <div>
            <Label>Clear chat history</Label>
            <p className="text-xs text-muted-foreground">
              Removes your saved AI Assistant conversations from this device.
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={handleClearChatHistory}>
            <Trash2 className="h-3.5 w-3.5 mr-1.5" /> Clear
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function DataOverviewTab() {
  const { data, isLoading } = useDashboardSummary();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Overview</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : (
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold">{data?.total_datasets ?? 0}</p>
              <p className="text-xs text-muted-foreground">Datasets</p>
            </div>
            <div>
              <p className="text-2xl font-bold">{data?.total_analyses ?? 0}</p>
              <p className="text-xs text-muted-foreground">Analyses</p>
            </div>
            <div>
              <p className="text-2xl font-bold">{data?.total_reports ?? 0}</p>
              <p className="text-xs text-muted-foreground">Reports</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function AccountTab() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    toast.success("Logged out");
    navigate({ to: "/login" });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Account</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label className="text-xs text-muted-foreground">Name</Label>
          <p className="text-sm font-medium">{user?.name}</p>
        </div>
        <div>
          <Label className="text-xs text-muted-foreground">Email</Label>
          <p className="text-sm font-medium">{user?.email}</p>
        </div>
        <Button variant="outline" onClick={handleLogout}>
          <LogOut className="h-4 w-4 mr-2" /> Log out
        </Button>
      </CardContent>
    </Card>
  );
}
