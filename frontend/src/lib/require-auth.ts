import { redirect } from "@tanstack/react-router";
import { getToken } from "./auth-store";

// beforeLoad guard for protected routes: `beforeLoad: requireAuth` in a
// route file's createFileRoute() options. Runs outside React (router-level),
// so it reads the token directly from storage rather than via useAuth().
//
// This also runs during SSR (the initial render of a hard navigation/refresh),
// where localStorage doesn't exist — skip the check there rather than treating
// "can't see a token" as "no token," which would bounce an already-logged-in
// user to /login on every page refresh. The client-side re-check after
// hydration (AppShell in __root.tsx, driven by AuthProvider) is the real
// guard for that case; the backend independently rejects unauthenticated API
// calls either way, so nothing is exposed by rendering the shell briefly.
export function requireAuth({ location }: { location: { href: string } }) {
  if (typeof window === "undefined") return;
  if (!getToken()) {
    throw redirect({ to: "/login", search: { redirect: location.href } });
  }
}
