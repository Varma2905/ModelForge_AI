import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "@/lib/api-service";
import type { AIChatErrorType, AIChatTurn } from "@/lib/api-types";
import { getStoredUser } from "@/lib/auth-store";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  error?: { message: string; type: AIChatErrorType };
  createdAt: number;
};

export type Conversation = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
};

// Client-side mirror of the backend's own history cap (see MAX_HISTORY_MESSAGES
// in ai_routes.py) — trimming here too keeps the request small and avoids ever
// tripping the backend's 400 for an oversized history.
const MAX_HISTORY_TURNS = 20;

function makeId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function titleFromText(text: string) {
  const t = text.trim().replace(/\s+/g, " ");
  if (!t) return "New conversation";
  return t.length > 48 ? `${t.slice(0, 48)}…` : t;
}

function makeConversation(): Conversation {
  return { id: makeId(), title: "New conversation", messages: [], createdAt: Date.now() };
}

function loadPersisted(storageKey: string): Conversation[] | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Conversation[];
    if (Array.isArray(parsed) && parsed.length > 0) return parsed;
  } catch {
    // ignore corrupt storage
  }
  return null;
}

export function useAiChat(options: { persistKey?: string; contextModelId?: string | null } = {}) {
  const { persistKey, contextModelId } = options;

  const storageKey = persistKey
    ? `regression-studio:ai-chat:${persistKey}:${getStoredUser()?.id ?? "anon"}`
    : null;

  const [conversations, setConversations] = useState<Conversation[]>(
    () => (storageKey && loadPersisted(storageKey)) || [makeConversation()],
  );
  const [activeId, setActiveId] = useState<string>(() => conversations[0].id);
  const [sending, setSending] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Persist whenever conversations change (multi-conversation mode only).
  useEffect(() => {
    if (!storageKey || typeof window === "undefined") return;
    try {
      const toSave = conversations.map((c) => ({
        ...c,
        messages: c.messages.map(({ streaming: _s, ...m }) => m),
      }));
      localStorage.setItem(storageKey, JSON.stringify(toSave));
    } catch {
      // storage full/unavailable — degrade to in-memory only
    }
  }, [conversations, storageKey]);

  // Keep activeId valid if the active conversation was deleted.
  useEffect(() => {
    if (!conversations.find((c) => c.id === activeId)) {
      setActiveId(conversations[0]?.id ?? "");
    }
  }, [conversations, activeId]);

  const activeConversation = conversations.find((c) => c.id === activeId) ?? conversations[0];

  const updateConversation = useCallback((id: string, updater: (c: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  }, []);

  const newConversation = useCallback(() => {
    const conv = makeConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    return conv.id;
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      return next.length > 0 ? next : [makeConversation()];
    });
  }, []);

  const clearConversation = useCallback(
    (id: string) => {
      updateConversation(id, (c) => ({ ...c, messages: [], title: "New conversation" }));
    },
    [updateConversation],
  );

  const stop = useCallback(() => {
    const controller = abortRef.current;
    if (!controller) return;
    controller.abort();
    abortRef.current = null;
    setSending(false);
    setStreaming(false);
    setConversations((prev) =>
      prev.map((c) => ({
        ...c,
        messages: c.messages.map((m) => (m.streaming ? { ...m, streaming: false } : m)),
      })),
    );
  }, []);

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || sending || streaming) return;

      const convId = activeId;
      const userMsg: ChatMessage = { id: makeId(), role: "user", content: trimmed, createdAt: Date.now() };
      const assistantMsg: ChatMessage = {
        id: makeId(),
        role: "assistant",
        content: "",
        streaming: true,
        createdAt: Date.now(),
      };

      let historyForRequest: AIChatTurn[] = [];
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const isFirst = c.messages.length === 0;
          historyForRequest = [...c.messages, userMsg]
            .slice(-MAX_HISTORY_TURNS)
            .map((m) => ({ role: m.role, content: m.content }));
          return {
            ...c,
            title: isFirst ? titleFromText(trimmed) : c.title,
            messages: [...c.messages, userMsg, assistantMsg],
          };
        }),
      );

      const controller = new AbortController();
      abortRef.current = controller;
      setSending(true);
      setStreaming(false);
      let firstToken = true;

      streamChat(
        { messages: historyForRequest, model_id: contextModelId ?? undefined },
        {
          onDelta: (delta) => {
            if (firstToken) {
              firstToken = false;
              setSending(false);
              setStreaming(true);
            }
            updateConversation(convId, (c) => ({
              ...c,
              messages: c.messages.map((m) =>
                m.id === assistantMsg.id ? { ...m, content: m.content + delta } : m,
              ),
            }));
          },
          onDone: () => {
            setSending(false);
            setStreaming(false);
            abortRef.current = null;
            updateConversation(convId, (c) => ({
              ...c,
              messages: c.messages.map((m) => (m.id === assistantMsg.id ? { ...m, streaming: false } : m)),
            }));
          },
          onError: (message, type) => {
            setSending(false);
            setStreaming(false);
            abortRef.current = null;
            updateConversation(convId, (c) => ({
              ...c,
              messages: c.messages.map((m) =>
                m.id === assistantMsg.id ? { ...m, streaming: false, error: { message, type } } : m,
              ),
            }));
          },
        },
        controller.signal,
      );
    },
    [activeId, sending, streaming, contextModelId, updateConversation],
  );

  return {
    conversations,
    activeConversation,
    activeId,
    setActiveId,
    newConversation,
    deleteConversation,
    clearConversation,
    send,
    stop,
    sending,
    streaming,
  };
}
