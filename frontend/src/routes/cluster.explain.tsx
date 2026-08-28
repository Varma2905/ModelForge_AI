import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { WizardSteps, clusteringSteps } from "@/components/wizard-steps";
import { useClusteringAnalysis } from "@/lib/clustering-analysis-store";
import { useAiExplanation } from "@/hooks/use-reports";
import { useAiChat } from "@/hooks/use-ai-chat";
import { AiChatPanel } from "@/components/ai-chat-panel";
import { requireAuth } from "@/lib/require-auth";
import { Sparkles } from "lucide-react";

export const Route = createFileRoute("/cluster/explain")({
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
  return blocks
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
    .slice(1);
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
  const { state } = useClusteringAnalysis();
  const navigate = useNavigate();
  const explainQuery = useAiExplanation(state.modelId);
  const chat = useAiChat({ contextModelId: state.modelId });

  if (!state.modelId) {
    return (
      <div>
        <WizardSteps steps={clusteringSteps} />
        <Card>
          <CardContent className="p-8 text-center">
            <p>Run clustering first to get AI insights.</p>
            <Button asChild className="mt-4">
              <Link to="/cluster/train">Go to clustering</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const sections = explainQuery.data ? parseReportSections(explainQuery.data.full_report) : [];

  return (
    <div>
      <WizardSteps steps={clusteringSteps} />
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
                Couldn't generate the AI explanation right now. You can still review the analysis
                and visualizations, and continue to the report.
              </CardContent>
            </Card>
          ) : (
            sections.map((s) => <ReportSection key={s.title} title={s.title} body={s.body} />)
          )}

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
              placeholder="Ask about your clusters…"
              emptyState={
                <p className="text-sm text-muted-foreground text-center py-6">
                  Ask me anything about these clusters, their quality, or how to improve them.
                </p>
              }
            />
          </div>
        </Card>
      </div>

      <div className="flex justify-end mt-6">
        <Button onClick={() => navigate({ to: "/cluster/report" })}>Generate Report</Button>
      </div>
    </div>
  );
}
