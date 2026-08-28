import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useNavigate,
  useRouter,
  useRouterState,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";
import { Loader2 } from "lucide-react";

import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";
import { AnalysisProvider } from "@/lib/analysis-store";
import { ClassificationAnalysisProvider } from "@/lib/classification-analysis-store";
import { ClusteringAnalysisProvider } from "@/lib/clustering-analysis-store";
import { AuthProvider, useAuth } from "@/lib/auth-context";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider, themeInitScript } from "@/lib/theme";
import { ThemeToggle } from "@/components/theme-toggle";
import { UserMenu } from "@/components/user-menu";
import { CommandMenuProvider, useCommandMenu } from "@/components/command-menu";
import { NewAnalysisModalProvider } from "@/components/new-analysis-modal";
import { NotificationsBell } from "@/components/notifications-bell";
import { Button } from "@/components/ui/button";
import { Search } from "lucide-react";

function HeaderSearch() {
  const { setOpen } = useCommandMenu();
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="hidden sm:flex items-center gap-2 w-full max-w-sm rounded-lg border border-input bg-muted/40 px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted/70 transition-colors"
      >
        <Search className="h-3.5 w-3.5 flex-shrink-0" />
        <span className="flex-1 text-left truncate">Search datasets, analyses, reports…</span>
        <kbd className="pointer-events-none inline-flex h-5 items-center gap-0.5 rounded border border-border bg-background px-1.5 font-mono text-[10px] text-muted-foreground">
          ⌘K
        </kbd>
      </button>
      <Button
        variant="ghost"
        size="icon"
        className="sm:hidden"
        aria-label="Search"
        onClick={() => setOpen(true)}
      >
        <Search className="h-4 w-4" />
      </Button>
    </>
  );
}

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Go home
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();
  useEffect(() => {
    reportLovableError(error, { boundary: "tanstack_root_error_component" });
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          This page didn't load
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Something went wrong on our end. You can try refreshing or head back home.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Try again
          </button>
          <a
            href="/"
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "ModelForge AI Studio — Automated Regression with Agentic AI" },
      {
        name: "description",
        content:
          "Upload data, train regression models, visualize performance, and get AI-generated explanations and PDF reports.",
      },
      { property: "og:title", content: "ModelForge AI Studio" },
      {
        property: "og:description",
        content: "Automated regression analysis with agentic AI insights.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      { rel: "icon", href: "/favicon.ico", type: "image/x-icon" },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <AnalysisProvider>
            <ClassificationAnalysisProvider>
              <ClusteringAnalysisProvider>
                <AppShell />
              </ClusteringAnalysisProvider>
            </ClassificationAnalysisProvider>
          </AnalysisProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

const AUTH_ROUTES = new Set(["/login", "/signup"]);

function AppShell() {
  const pathname = useRouterState({ select: (r) => r.location.pathname });
  const { isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();
  const isAuthRoute = AUTH_ROUTES.has(pathname);

  // The router's `requireAuth` beforeLoad guard only runs its redirect in the
  // browser (it no-ops during SSR, since localStorage isn't visible there —
  // see lib/require-auth.ts). This effect is the guard that actually covers
  // a hard navigation/refresh: once AuthProvider finishes validating (or
  // fails to find) a token, redirect away from protected pages here. No real
  // data is exposed in the interim — every data-fetching call independently
  // requires a valid token server-side.
  useEffect(() => {
    if (!isAuthRoute && !isLoading && !isAuthenticated) {
      navigate({ to: "/login", search: { redirect: pathname } });
    }
  }, [isAuthRoute, isLoading, isAuthenticated, pathname, navigate]);

  // Auth pages render full-bleed, with no sidebar chrome.
  if (isAuthRoute) {
    return (
      <>
        <Outlet />
        <Toaster richColors position="top-right" />
      </>
    );
  }

  // Briefly gate protected pages while a stored token is being validated
  // against the server, to avoid flashing the app shell right before a
  // possible redirect to /login for an expired session.
  if (!isAuthenticated) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <CommandMenuProvider>
      <NewAnalysisModalProvider>
        <SidebarProvider>
          <div className="min-h-screen flex w-full bg-background">
            <AppSidebar />
            <div className="flex-1 flex flex-col min-w-0">
              <header className="h-14 border-b border-border flex items-center gap-3 px-4 sticky top-0 bg-background/80 backdrop-blur z-10">
                <SidebarTrigger />
                <HeaderSearch />
                <div className="ml-auto flex items-center gap-1">
                  <ThemeToggle />
                  <NotificationsBell />
                  <span className="hidden sm:inline text-xs text-muted-foreground mx-1">
                    Agentic AI · v1.0
                  </span>
                  <UserMenu variant="topbar" />
                </div>
              </header>
              <main className="flex-1 p-6">
                <div key={pathname} className="page-enter">
                  <Outlet />
                </div>
              </main>
            </div>
          </div>
          <Toaster richColors position="top-right" />
        </SidebarProvider>
      </NewAnalysisModalProvider>
    </CommandMenuProvider>
  );
}
