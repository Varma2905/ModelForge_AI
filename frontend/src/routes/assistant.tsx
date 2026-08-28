import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Plus, Trash2, MessageSquare, Eraser, FileText } from "lucide-react";
import { toast } from "sonner";
import { AiChatPanel } from "@/components/ai-chat-panel";
import { GradientIcon } from "@/components/gradient-icon";
import { ReportSelectorPanel } from "@/components/report-selector-panel";
import { useAiChat } from "@/hooks/use-ai-chat";
import { useModelsList } from "@/hooks/use-training";
import { requireAuth } from "@/lib/require-auth";
import {
  REPORT_TYPE_META,
  SUGGESTIONS_BY_TYPE,
  PLACEHOLDER_BY_TYPE,
  GENERAL_SUGGESTIONS,
  GENERAL_PLACEHOLDER,
  reportType,
  reportTitle,
} from "@/lib/report-display";

type AssistantSearch = { prefill?: string };

export const Route = createFileRoute("/assistant")({
  beforeLoad: requireAuth,
  validateSearch: (search: Record<string, unknown>): AssistantSearch => ({
    prefill: typeof search.prefill === "string" ? search.prefill : undefined,
  }),
  component: AssistantPage,
});

function AssistantPage() {
  const { prefill } = Route.useSearch();
  const reportsQuery = useModelsList();
  const chat = useAiChat({ persistKey: "assistant" });
  const reportPanelRef = useRef<HTMLDivElement>(null);

  const reports = reportsQuery.data ?? [];
  const selectedReportId = chat.activeConversation.reportId;
  const selectedReport = reports.find((r) => r.model_id === selectedReportId) ?? null;

  // If the report a conversation was analyzing has since been deleted from
  // the Reports page, clear it safely instead of silently sending a dead
  // model_id (or crashing on stale data).
  useEffect(() => {
    if (reportsQuery.isLoading || reportsQuery.isError || !selectedReportId) return;
    const stillExists = reports.some((r) => r.model_id === selectedReportId);
    if (!stillExists) {
      chat.setConversationReport(chat.activeId, null);
      toast.error("The selected report is no longer available. Please select another report.");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reportsQuery.isLoading, reportsQuery.isError, selectedReportId, chat.activeId]);

  const type = selectedReport ? reportType(selectedReport) : null;
  const meta = type ? REPORT_TYPE_META[type] : null;
  // Real-time mode (no report selected) still works — it just gets no
  // report context, per the Real-Time vs Report-Based Conversation modes.
  const suggestions = type ? SUGGESTIONS_BY_TYPE[type] : GENERAL_SUGGESTIONS;
  const placeholder = type ? PLACEHOLDER_BY_TYPE[type] : GENERAL_PLACEHOLDER;

  return (
    <div className="max-w-6xl mx-auto space-y-4">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" /> AI Assistant
        </h1>
        <p className="text-sm text-muted-foreground">
          Get expert help with your regression, classification, and clustering analysis.
        </p>
        <p className="text-[11px] text-muted-foreground/70 mt-0.5">
          Powered by Hugging Face · Qwen2.5-7B-Instruct
        </p>
      </div>

      <Card
        className="grid grid-cols-1 md:grid-cols-[220px_1fr] lg:grid-cols-[220px_1fr_280px] overflow-hidden"
        style={{ height: "72vh" }}
      >
        {/* Conversation list */}
        <div className="hidden md:flex flex-col border-r min-h-0">
          <div className="p-3 border-b">
            <Button size="sm" className="w-full" onClick={() => chat.newConversation()}>
              <Plus className="h-4 w-4" /> New Chat
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {chat.conversations.map((c) => (
              <button
                key={c.id}
                onClick={() => chat.setActiveId(c.id)}
                type="button"
                className={`group w-full flex items-center gap-2 rounded-md px-2.5 py-2 text-left text-xs transition-colors ${
                  c.id === chat.activeId ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
                }`}
              >
                <MessageSquare className="h-3.5 w-3.5 flex-shrink-0 opacity-60" />
                <span className="flex-1 truncate">{c.title}</span>
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    chat.deleteConversation(c.id);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.stopPropagation();
                      chat.deleteConversation(c.id);
                    }
                  }}
                  className="opacity-0 group-hover:opacity-100 hover:text-destructive flex-shrink-0"
                  aria-label="Delete conversation"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Active conversation */}
        <div className="flex flex-col min-h-0">
          <div className="flex items-center justify-between gap-2 border-b px-4 py-2.5">
            <div className="min-w-0 space-y-1">
              <p className="text-sm font-medium truncate">{chat.activeConversation.title}</p>
              {selectedReport && meta ? (
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[11px] text-muted-foreground">Analyzing:</span>
                  <Badge className={`${meta.badgeBg} ${meta.text} border-none text-[11px] font-normal gap-1 px-1.5`}>
                    <meta.icon className="h-3 w-3" /> {reportTitle(selectedReport)}
                  </Badge>
                </div>
              ) : (
                <p className="text-[11px] text-muted-foreground">Real-time chat — select a report to begin an analysis conversation</p>
              )}
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => chat.clearConversation(chat.activeId)}
              disabled={chat.activeConversation.messages.length === 0}
            >
              <Eraser className="h-3.5 w-3.5" /> Clear
            </Button>
          </div>

          <AiChatPanel
            messages={chat.activeConversation.messages}
            onSend={chat.send}
            onStop={chat.stop}
            sending={chat.sending}
            streaming={chat.streaming}
            placeholder={placeholder}
            suggestions={suggestions}
            initialInput={prefill}
            compact
            emptyState={
              selectedReport ? (
                <div className="h-full flex flex-col items-center justify-center text-center gap-2 py-10">
                  <GradientIcon icon={Sparkles} shape="circle" size="lg" className="mb-1" />
                  <p className="font-medium">Start a conversation</p>
                  <p className="text-sm text-muted-foreground max-w-xs">
                    Ask anything about {reportTitle(selectedReport).toLowerCase()} — metrics, results, or how to
                    improve them.
                  </p>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center gap-2 py-10">
                  <GradientIcon icon={FileText} shape="circle" size="lg" className="mb-1" />
                  <p className="font-medium">Select a Report</p>
                  <p className="text-sm text-muted-foreground max-w-xs">
                    Choose a generated report from the panel for report-specific analysis, or ask a general machine
                    learning question below.
                  </p>
                  <Button
                    size="sm"
                    variant="outline"
                    className="mt-1"
                    onClick={() => reportPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })}
                  >
                    Select Report
                  </Button>
                </div>
              )
            }
          />
        </div>

        {/* Generated Reports panel */}
        <div ref={reportPanelRef} className="hidden lg:flex flex-col border-l min-h-0">
          <ReportSelectorPanel
            reports={reports}
            isLoading={reportsQuery.isLoading}
            isError={reportsQuery.isError}
            selectedReportId={selectedReportId}
            onSelect={(id) => chat.setConversationReport(chat.activeId, id)}
          />
        </div>
      </Card>
    </div>
  );
}
