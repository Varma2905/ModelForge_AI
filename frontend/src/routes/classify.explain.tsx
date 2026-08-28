import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { WizardSteps, classificationSteps } from "@/components/wizard-steps";
import { useClassificationAnalysis } from "@/lib/classification-analysis-store";
import { useClassificationModelMetrics } from "@/hooks/use-training";
import { useAiExplanation } from "@/hooks/use-reports";
import { useAiChat } from "@/hooks/use-ai-chat";
import { AiChatPanel } from "@/components/ai-chat-panel";
import { requireAuth } from "@/lib/require-auth";
import { useChartTheme } from "@/lib/chart-theme";
import { Sparkles } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export const Route = createFileRoute("/classify/explain")({
  beforeLoad: requireAuth,
  component: ExplainPage,
});

function stripInlineMd(s: string) {
  return s.replace(/\*\*(.*?)\*\*/g, "$1").replace(/`(.*?)`/g, "$1");
}

function parseReportSections(markdown: string) {
  const lines = markdown.split("\n");
  const blocks: string[][] = [[]];
  for (const line of lines) {
    if (line.trim() === "---") {
      blocks.push([]);
    } else {
      blocks[blocks.length - 1].push(line);
    }
  }
  return (
    blocks
      .map((blockLines) => blockLines.map((l) => l.trim()).filter(Boolean))
      .filter((b) => b.length > 0)
      .map((body) => {
        let title = "Overview";
        let rest = body;
        const headerIdx = body.findIndex((l) => /^#{1,4}\s+/.test(l));
        if (headerIdx !== -1) {
          title = stripInlineMd(body[headerIdx].replace(/^#{1,4}\s+/, "")).trim();
          rest = body.slice(headerIdx + 1);
        }
        return { title, body: rest };
      })
      // The first block is the report's own title/KPI header — already covered
      // by the metric cards elsewhere in the wizard, so skip it here.
      .slice(1)
  );
}

function ReportSection({ title, body }: { title: string; body: string[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-muted-foreground leading-relaxed space-y-1.5">
        {body.map((line, i) => {
          if (/^#{2,4}\s+/.test(line)) {
            return (
              <p key={i} className="font-semibold text-foreground pt-2 first:pt-0">
                {stripInlineMd(line.replace(/^#{2,4}\s+/, ""))}
              </p>
            );
          }
          if (/^[-•*]\s+/.test(line)) {
            return (
              <p key={i} className="pl-3">
                • {stripInlineMd(line.replace(/^[-•*]\s+/, ""))}
              </p>
            );
          }
          return <p key={i}>{stripInlineMd(line)}</p>;
        })}
      </CardContent>
    </Card>
  );
}

function ExplainPage() {
  const { state } = useClassificationAnalysis();
  const navigate = useNavigate();
  const t = useChartTheme();
  const metricsQuery = useClassificationModelMetrics(state.modelId);
  const explainQuery = useAiExplanation(state.modelId);
  const chat = useAiChat({ contextModelId: state.modelId });

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps steps={classificationSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Train a model first to get AI insights.</p>
            <Button asChild className="mt-4">
              <Link to="/classify/train">Go to training</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const importance = metricsQuery.data?.chart_data.feature_importance ?? [];
  const sections = explainQuery.data ? parseReportSections(explainQuery.data.full_report) : [];
  // p_values keys are either "const" (binary Logit) or "const__class_{k}"
  // (flattened multiclass MNLogit intercepts) — both excluded the same way
  // map_expanded_coefficients() excludes them on the backend, since coefs
  // and p_values share the same key set/order from the same fitted model.
  const pValueList = Object.entries(metricsQuery.data?.statistical_analysis?.p_values ?? {}).filter(
    ([k]) => k !== "const" && !k.startsWith("const__"),
  );
  const coefficientRows = importance.map((entry, i) => ({
    ...entry,
    p_value: pValueList[i]?.[1] ?? null,
  }));

  return (
    <div>
      <WizardSteps steps={classificationSteps} />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          {explainQuery.isLoading ? (
            <>
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </>
          ) : explainQuery.isError ? (
            <Card>
              <CardContent className="p-6 text-sm text-muted-foreground">
                Couldn't generate the AI explanation right now. You can still review the charts
                below and continue to the report.
              </CardContent>
            </Card>
          ) : (
            sections.map((s) => <ReportSection key={s.title} title={s.title} body={s.body} />)
          )}

          <Card>
            <CardHeader>
              <CardTitle>Feature Importance</CardTitle>
            </CardHeader>
            <CardContent style={{ height: 240 }}>
              {metricsQuery.isLoading ? (
                <Skeleton className="h-full w-full" />
              ) : importance.length > 0 ? (
                <ResponsiveContainer>
                  <BarChart data={importance} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke={t.grid} opacity={0.5} />
                    <XAxis type="number" stroke={t.mutedForeground} tick={t.axisTick} />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      stroke={t.mutedForeground}
                      tick={t.axisTick}
                      width={80}
                    />
                    <Tooltip
                      contentStyle={t.tooltipStyle}
                      labelStyle={t.tooltipLabelStyle}
                      itemStyle={t.tooltipItemStyle}
                      cursor={{ fill: t.mutedForeground, opacity: 0.1 }}
                    />
                    <Bar dataKey="value" fill={t.series.primary} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-sm text-muted-foreground">No importance data available.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Coefficients (log-odds scale)</CardTitle>
              <p className="text-xs text-muted-foreground">
                Categorical features expand into one row per category, relative to a dropped
                baseline category. A multiclass target shows one row per category per class.
              </p>
            </CardHeader>
            <CardContent>
              {metricsQuery.isLoading ? (
                <Skeleton className="h-32 w-full" />
              ) : coefficientRows.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-xs text-muted-foreground">
                        <th className="py-2 pr-4 font-medium">Feature</th>
                        <th className="py-2 pr-4 font-medium">Coefficient</th>
                        <th className="py-2 font-medium">P-value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {coefficientRows.map((row) => (
                        <tr key={row.feature} className="border-b last:border-0">
                          <td className="py-2 pr-4 font-medium">{row.feature}</td>
                          <td className="py-2 pr-4 tabular-nums">
                            {typeof row.value === "number" ? row.value.toFixed(4) : "—"}
                          </td>
                          <td className="py-2 tabular-nums">
                            {typeof row.p_value === "number" ? (
                              <span className={row.p_value <= 0.05 ? "text-emerald-600 dark:text-emerald-500" : "text-muted-foreground"}>
                                {row.p_value.toFixed(4)}
                              </span>
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No coefficient data available.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Key Insights</CardTitle>
            </CardHeader>
            <CardContent>
              {explainQuery.isLoading ? (
                <Skeleton className="h-20 w-full" />
              ) : explainQuery.data?.insights.length ? (
                <ul className="space-y-2 text-sm">
                  {explainQuery.data.insights.map((insight, i) => (
                    <li key={i}>• {insight}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">No insights available yet.</p>
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="flex flex-col h-[600px] overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" /> AI Assistant
            </CardTitle>
          </CardHeader>
          <div className="flex-1 min-h-0">
            <AiChatPanel
              messages={chat.activeConversation.messages}
              onSend={chat.send}
              onStop={chat.stop}
              sending={chat.sending}
              streaming={chat.streaming}
              placeholder="Ask about your model…"
              emptyState={
                <p className="text-sm text-muted-foreground text-center py-6">
                  Ask me anything about this model's results, coefficients, or how to improve it.
                </p>
              }
            />
          </div>
        </Card>
      </div>

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/classify/report" })}>Generate Report</Button>
      </div>
    </div>
  );
}
