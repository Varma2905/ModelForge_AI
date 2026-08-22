import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query";
import { createRouter } from "@tanstack/react-router";
import { routeTree } from "./routeTree.gen";
import { isUnauthorizedError } from "@/lib/api-service";
import { clearToken } from "@/lib/auth-store";

function handleQueryError(error: unknown) {
  if (isUnauthorizedError(error) && typeof window !== "undefined") {
    clearToken();
    if (!window.location.pathname.startsWith("/login")) {
      window.location.assign("/login");
    }
  }
}

export const getRouter = () => {
  const queryClient = new QueryClient({
    queryCache: new QueryCache({ onError: handleQueryError }),
    mutationCache: new MutationCache({ onError: handleQueryError }),
  });

  const router = createRouter({
    routeTree,
    context: { queryClient },
    scrollRestoration: true,
    defaultPreloadStaleTime: 0,
    // The wizard steps (WizardSteps' <Link> bar, present on every /new/*
    // page) are NOT idempotent to visit — mounting /new/train fires a real
    // POST /train-model, /new/preprocess mutates a dataset, /new/explain
    // calls the AI pipeline, etc. TanStack Router's default intent-based
    // preloading mounts a route's component (running its effects) just from
    // the user's cursor passing near a <Link>, which for this app means
    // real side effects — e.g. training a throwaway model — can fire before
    // the user ever clicks anything. Preloading has no benefit here anyway
    // (every step depends on state only the previous step produces), so
    // it's disabled outright rather than left at the framework default.
    defaultPreload: false,
  });

  return router;
};
