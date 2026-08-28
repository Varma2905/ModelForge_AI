import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle2, Database, FileJson, FileSpreadsheet, FileText, Lightbulb } from "lucide-react";

const FORMATS = [
  { label: "CSV", desc: "Comma Separated Values", icon: FileText },
  { label: "XLSX", desc: "Excel Spreadsheet", icon: FileSpreadsheet },
  { label: "JSON", desc: "JavaScript Object Notation", icon: FileJson },
  { label: "Parquet", desc: "Columnar Storage Format", icon: Database },
];

const TIPS = [
  "Ensure your dataset has a header row",
  "Check for missing values before analysis",
  "Remove unnecessary columns",
  "For best results, clean your data",
];

const REGRESSION_REQUIREMENTS = [
  "Minimum 10 rows recommended",
  "At least 1 numerical column",
  "No completely empty columns",
  "Target variable should be numerical",
];

const CLASSIFICATION_REQUIREMENTS = [
  "Minimum 10 rows recommended",
  "At least 1 numerical or categorical column",
  "No completely empty columns",
  "Target should be categorical, or a low-cardinality integer label",
];

const CLUSTERING_REQUIREMENTS = [
  "Minimum 10 rows recommended",
  "At least 2 numerical or categorical columns",
  "No completely empty columns",
  "No target column needed — clustering is unsupervised",
];

export function UploadInfoCards({ taskType = "regression" }: { taskType?: "regression" | "classification" | "clustering" }) {
  const REQUIREMENTS =
    taskType === "classification"
      ? CLASSIFICATION_REQUIREMENTS
      : taskType === "clustering"
        ? CLUSTERING_REQUIREMENTS
        : REGRESSION_REQUIREMENTS;
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Supported Formats</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {FORMATS.map((f) => (
            <div key={f.label} className="flex items-center gap-3">
              <f.icon className="h-4 w-4 text-muted-foreground shrink-0" />
              <div className="min-w-0">
                <span className="text-sm font-medium">{f.label}</span>
                <span className="text-xs text-muted-foreground ml-2">{f.desc}</span>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <CardTitle className="text-base">Tips</CardTitle>
          <Lightbulb className="h-4 w-4 text-warning" />
        </CardHeader>
        <CardContent className="space-y-2">
          {TIPS.map((t) => (
            <div key={t} className="flex items-start gap-2 text-sm text-muted-foreground">
              <CheckCircle2 className="h-3.5 w-3.5 text-success shrink-0 mt-0.5" />
              {t}
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <CardTitle className="text-base">Dataset Requirements</CardTitle>
          <Database className="h-4 w-4 text-primary" />
        </CardHeader>
        <CardContent className="space-y-2">
          {REQUIREMENTS.map((r) => (
            <div key={r} className="flex items-start gap-2 text-sm text-muted-foreground">
              <CheckCircle2 className="h-3.5 w-3.5 text-success shrink-0 mt-0.5" />
              {r}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
