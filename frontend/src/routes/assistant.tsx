import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Sparkles, Plus, Trash2, MessageSquare, Eraser, Database } from "lucide-react";
import { AiChatPanel } from "@/components/ai-chat-panel";
import { useAiChat } from "@/hooks/use-ai-chat";
import { api } from "@/lib/api-service";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/assistant")({
  beforeLoad: requireAuth,
  component: AssistantPage,
});

const SUGGESTIONS = [
  "Which regression model should I choose?",
  "How do I interpret my R² score?",
  "What is overfitting and how do I fix it?",
  "When should I use Ridge vs Lasso?",
  "Why is my model performing badly?",
  "How can I improve this model?",
];

function AssistantPage() {
  const [contextModelId, setContextModelId] = useState<string | null>(null);
  const [contextLabel, setContextLabel] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listModels()
      .then((models) => {
        if (cancelled || models.length === 0) return;
        const latest = models[0];
        setContextModelId(latest.model_id);
        setContextLabel(`${latest.dataset_name} — ${latest.model}`);
      })
      .catch(() => {
        // No trained models yet, or request failed — chat still works, just without context.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const chat = useAiChat({ persistKey: "assistant", contextModelId });

  return (
    <div className="max-w-6xl mx-auto space-y-4">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-indigo-500" /> AI Assistant
        </h1>
        <p className="text-sm text-muted-foreground">
          Get expert help with regression analysis, model selection, and result interpretation.
        </p>
      </div>

      <Card className="grid grid-cols-1 md:grid-cols-[240px_1fr] overflow-hidden" style={{ height: "72vh" }}>
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
            <div className="min-w-0">
              <p className="text-sm font-medium truncate">{chat.activeConversation.title}</p>
              {contextLabel && (
                <p className="text-[11px] text-muted-foreground flex items-center gap-1 truncate">
                  <Database className="h-3 w-3 flex-shrink-0" /> Context: {contextLabel}
                </p>
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
            placeholder="Ask me anything about your regression analysis…"
            suggestions={SUGGESTIONS}
            emptyState={
              <div className="h-full flex flex-col items-center justify-center text-center gap-2 py-10">
                <div className="h-12 w-12 rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500 flex items-center justify-center text-white mb-1">
                  <Sparkles className="h-6 w-6" />
                </div>
                <p className="font-medium">Start a conversation</p>
                <p className="text-sm text-muted-foreground max-w-xs">
                  Ask anything about your regression analysis — model selection, R², RMSE, p-values, or how to
                  improve your results.
                </p>
              </div>
            }
          />
        </div>
      </Card>
    </div>
  );
}
