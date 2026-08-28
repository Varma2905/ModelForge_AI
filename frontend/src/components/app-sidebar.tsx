import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  PlusCircle,
  Database,
  Layers,
  FileText,
  Bot,
  Settings,
  BrainCircuit,
  GitCompare,
  Boxes,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { UserMenu } from "@/components/user-menu";
import { useAuth } from "@/lib/auth-context";
import { useNewAnalysisModal } from "@/components/new-analysis-modal";

const NAV_ITEMS = [{ title: "Dashboard", url: "/", icon: LayoutDashboard }];

const WORKFLOW_PREFIXES = ["/new", "/classify", "/cluster"];

const REST_ITEMS = [
  { title: "Data Sources", url: "/data-sources", icon: Database },
  { title: "Saved Models", url: "/history", icon: Layers },
  { title: "Hugging Face Models", url: "/hf-models", icon: Boxes },
  { title: "Compare Models", url: "/compare", icon: GitCompare },
  { title: "Reports", url: "/reports", icon: FileText },
  { title: "AI Assistant", url: "/assistant", icon: Bot },
  { title: "Settings", url: "/settings", icon: Settings },
];

const ACTIVE_CLASSES =
  "relative bg-[image:var(--gradient-brand)] text-white shadow-[0_0_16px_oklch(0.606_0.219_292.717_/_0.45)] hover:text-white hover:bg-[image:var(--gradient-brand)] data-[active=true]:bg-[image:var(--gradient-brand)] data-[active=true]:text-white before:absolute before:left-0 before:top-1/2 before:-translate-y-1/2 before:h-4 before:w-0.5 before:rounded-full before:bg-white";

export function AppSidebar() {
  const pathname = useRouterState({ select: (r) => r.location.pathname });
  const { user } = useAuth();
  const { setOpen } = useNewAnalysisModal();

  const isWorkflowActive = WORKFLOW_PREFIXES.some((prefix) => pathname.startsWith(prefix));

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 py-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[image:var(--gradient-brand)] text-white flex-shrink-0 shadow-lg shadow-primary/30">
            <BrainCircuit className="h-5 w-5" />
          </div>
          <div className="flex flex-col min-w-0 group-data-[collapsible=icon]:hidden">
            <span className="text-sm font-semibold truncate">ModelForge AI</span>
            <span className="text-xs text-muted-foreground truncate">Studio</span>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Workspace</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_ITEMS.map((item) => {
                const active = pathname === "/";
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={active}
                      className={active ? ACTIVE_CLASSES : undefined}
                    >
                      <Link to={item.url}>
                        <item.icon />
                        <span className="group-data-[collapsible=icon]:hidden">{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}

              <SidebarMenuItem>
                <SidebarMenuButton
                  isActive={isWorkflowActive}
                  className={isWorkflowActive ? ACTIVE_CLASSES : undefined}
                  onClick={() => setOpen(true)}
                >
                  <PlusCircle />
                  <span className="group-data-[collapsible=icon]:hidden">New Analysis</span>
                </SidebarMenuButton>
              </SidebarMenuItem>

              {REST_ITEMS.map((item) => {
                const active = pathname.startsWith(item.url.split("/").slice(0, 2).join("/"));
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={active}
                      className={active ? ACTIVE_CLASSES : undefined}
                    >
                      <Link to={item.url}>
                        <item.icon />
                        <span className="group-data-[collapsible=icon]:hidden">{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      {user && (
        <SidebarFooter>
          <UserMenu variant="sidebar" />
        </SidebarFooter>
      )}
    </Sidebar>
  );
}
