import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  LayoutDashboard,
  Sparkles,
  Plug,
  Boxes,
  Rocket,
  Database,
  GitCompare,
  FileText,
  Bot,
  Settings,
  Search,
} from "lucide-react";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Button } from "@/components/ui/button";
import type { ModelListItem } from "@/lib/api-types";

const NAV_ITEMS = [
  { title: "Dashboard", url: "/" as const, icon: LayoutDashboard },
  { title: "New Analysis", url: "/new/upload" as const, icon: Sparkles },
  { title: "Data Sources", url: "/data-sources" as const, icon: Plug },
  { title: "Hugging Face Models", url: "/hf-models" as const, icon: Boxes },
  { title: "Projects", url: "/baas" as const, icon: Rocket },
  { title: "Dataset History", url: "/history" as const, icon: Database },
  { title: "Compare Models", url: "/compare" as const, icon: GitCompare },
  { title: "Reports", url: "/reports" as const, icon: FileText },
  { title: "AI Assistant", url: "/assistant" as const, icon: Bot },
  { title: "Settings", url: "/settings" as const, icon: Settings },
];

type CommandMenuContextValue = { open: boolean; setOpen: (open: boolean) => void };
const CommandMenuContext = createContext<CommandMenuContextValue | undefined>(undefined);

export function CommandMenuProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  return (
    <CommandMenuContext.Provider value={{ open, setOpen }}>
      {children}
      <CommandMenu open={open} setOpen={setOpen} />
    </CommandMenuContext.Provider>
  );
}

export function useCommandMenu() {
  const ctx = useContext(CommandMenuContext);
  if (!ctx) throw new Error("useCommandMenu must be used within CommandMenuProvider");
  return ctx;
}

function CommandMenu({ open, setOpen }: { open: boolean; setOpen: (v: boolean) => void }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Only reads whatever's already in the React Query cache from pages the
  // user has visited this session — never triggers a new fetch, so this
  // stays honest rather than implying a real search index that doesn't exist.
  const recentModels = queryClient.getQueryData<ModelListItem[]>(["models"]) ?? [];

  const goToRoute = useCallback(
    (url: (typeof NAV_ITEMS)[number]["url"]) => {
      setOpen(false);
      navigate({ to: url });
    },
    [navigate, setOpen],
  );

  const goToModel = useCallback(
    (modelId: string) => {
      setOpen(false);
      navigate({ to: "/models/$modelId", params: { modelId } });
    },
    [navigate, setOpen],
  );

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Jump to a page or a recently viewed model…" />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Navigate">
          {NAV_ITEMS.map((item) => (
            <CommandItem key={item.url} value={item.title} onSelect={() => goToRoute(item.url)}>
              <item.icon />
              <span>{item.title}</span>
            </CommandItem>
          ))}
        </CommandGroup>
        {recentModels.length > 0 && (
          <CommandGroup heading="Recent models">
            {recentModels.slice(0, 5).map((m) => (
              <CommandItem
                key={m.model_id}
                value={`${m.dataset_name} ${m.model}`}
                onSelect={() => goToModel(m.model_id)}
              >
                <Sparkles />
                <span>
                  {m.dataset_name} — {m.model}
                </span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  );
}

export function CommandMenuTrigger() {
  const { setOpen } = useCommandMenu();
  return (
    <Button variant="ghost" size="icon" onClick={() => setOpen(true)} aria-label="Search (Ctrl+K)">
      <Search className="h-4 w-4" />
    </Button>
  );
}
