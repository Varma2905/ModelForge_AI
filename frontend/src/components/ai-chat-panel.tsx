import { useEffect, useRef, useState, type ReactNode } from "react";
import { Bot, User, Sparkles, Send, Loader2, Square, Check, Copy, ArrowDown, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatMarkdown } from "@/components/chat-markdown";
import { GradientIcon } from "@/components/gradient-icon";
import type { ChatMessage } from "@/hooks/use-ai-chat";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard
          ?.writeText(text)
          .then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
          })
          .catch(() => {});
      }}
      className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
      aria-label="Copy message"
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function errorLabel(type?: string) {
  switch (type) {
    case "auth":
      return "Authentication error";
    case "rate_limit":
      return "Rate limited";
    case "network":
      return "Connection error";
    case "config":
      return "Not configured";
    default:
      return "Error";
  }
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const isEmpty = !message.content && message.streaming && !message.error;

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : ""}`}>
      {!isUser && <GradientIcon icon={Bot} shape="circle" size="sm" className="h-9 w-9" />}
      <div
        className={`group rounded-2xl px-4 py-2.5 max-w-[80%] text-sm ${
          isUser ? "bg-primary text-primary-foreground whitespace-pre-wrap" : "bg-muted"
        }`}
      >
        {isUser ? (
          message.content
        ) : isEmpty ? (
          <div className="flex items-center gap-2 text-muted-foreground py-0.5">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>AI is thinking…</span>
          </div>
        ) : message.error ? (
          <div className="flex items-start gap-2 text-sm">
            <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0 text-amber-500" />
            <div>
              <div className="font-medium text-xs text-amber-600 dark:text-amber-400 mb-0.5">
                {errorLabel(message.error.type)}
              </div>
              <div className="whitespace-pre-wrap">{message.error.message}</div>
            </div>
          </div>
        ) : (
          <>
            <ChatMarkdown content={message.content} />
            {message.streaming && (
              <span className="inline-block w-1.5 h-4 align-text-bottom bg-foreground/60 ml-0.5 animate-pulse" />
            )}
          </>
        )}

        {!isUser && !message.streaming && message.content && !message.error && (
          <div className="mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <CopyButton text={message.content} />
          </div>
        )}
      </div>
      {isUser && (
        <div className="h-9 w-9 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
          <User className="h-5 w-5" />
        </div>
      )}
    </div>
  );
}

export function AiChatPanel({
  messages,
  onSend,
  onStop,
  sending,
  streaming,
  placeholder = "Ask a question…",
  emptyState,
  suggestions,
  headerActions,
  initialInput,
  disabled = false,
  compact = false,
}: {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  onStop: () => void;
  sending: boolean;
  streaming: boolean;
  placeholder?: string;
  emptyState?: ReactNode;
  suggestions?: string[];
  headerActions?: ReactNode;
  /** Pre-fills the input box, unsent — the user still reviews/edits before sending. */
  initialInput?: string;
  /** Disables the input + send button entirely (e.g. no report selected yet). */
  disabled?: boolean;
  /** Smaller, narrower composer (input/button/suggestions/helper text) — opt-in so
   * every other caller of this shared panel keeps its current sizing untouched. */
  compact?: boolean;
}) {
  const [input, setInput] = useState(initialInput ?? "");
  const busy = sending || streaming || disabled;

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [isNearBottom, setIsNearBottom] = useState(true);
  const [hasNewBelow, setHasNewBelow] = useState(false);

  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    bottomRef.current?.scrollIntoView({ behavior });
    setHasNewBelow(false);
  };

  useEffect(() => {
    if (isNearBottom) {
      scrollToBottom(streaming ? "auto" : "smooth");
    } else {
      setHasNewBelow(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, streaming]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const near = distanceFromBottom < 80;
    setIsNearBottom(near);
    if (near) setHasNewBelow(false);
  };

  const submit = () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    onSend(text);
    setIsNearBottom(true);
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {headerActions}

      <div className="relative flex-1 min-h-0">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="chat-scroll h-full overflow-y-auto space-y-4 p-6"
        >
          {messages.length === 0 && emptyState}

          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}

          <div ref={bottomRef} />
        </div>

        {hasNewBelow && (
          <button
            type="button"
            onClick={() => scrollToBottom()}
            className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1.5 rounded-full border bg-background shadow-md px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors"
          >
            <ArrowDown className="h-3.5 w-3.5" /> New response
          </button>
        )}
      </div>

      <div className={compact ? "border-t px-4 py-3 space-y-2 flex-shrink-0" : "border-t p-4 space-y-3 flex-shrink-0"}>
        <div className={compact ? "md:w-3/4 md:mx-auto space-y-2" : undefined}>
          {suggestions && suggestions.length > 0 && (
            <div className={compact ? "flex flex-wrap gap-1.5" : "flex flex-wrap gap-2"}>
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  disabled={busy}
                  type="button"
                  className={
                    compact
                      ? "text-[11px] rounded-full border px-2.5 py-0.5 hover:bg-accent transition-colors disabled:opacity-40"
                      : "text-xs rounded-full border px-3 py-1 hover:bg-accent transition-colors disabled:opacity-40"
                  }
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          <div className="flex gap-2 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder={placeholder}
              rows={1}
              disabled={disabled}
              className={
                compact
                  ? "chat-scroll flex-1 resize-none rounded-lg border border-input bg-background px-3.5 py-3 text-[13px] shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring max-h-40 min-h-12 disabled:opacity-50 disabled:cursor-not-allowed"
                  : "chat-scroll flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring max-h-40 min-h-9 disabled:opacity-50 disabled:cursor-not-allowed"
              }
              style={{ height: "auto" }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
              }}
            />
            {streaming ? (
              <Button
                onClick={onStop}
                variant="destructive"
                type="button"
                className={compact ? "h-12 flex-shrink-0" : undefined}
              >
                <Square className="h-4 w-4" /> Stop
              </Button>
            ) : (
              <Button
                onClick={submit}
                disabled={busy || !input.trim()}
                type="button"
                size={compact ? "icon" : undefined}
                className={compact ? "h-12 w-12 flex-shrink-0" : undefined}
              >
                {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            )}
          </div>
          <p
            className={
              compact
                ? "text-[10px] text-muted-foreground flex items-center gap-1"
                : "text-[11px] text-muted-foreground flex items-center gap-1"
            }
          >
            <Sparkles className={compact ? "h-2.5 w-2.5" : "h-3 w-3"} /> Enter to send, Shift+Enter for a new line.
          </p>
        </div>
      </div>
    </div>
  );
}
