// Runnable end-to-end example, doubling as this SDK's verification: exercises
// real end-user signup/login and real record CRUD against a real running
// Studio backend and a real project (created via the Studio's own UI/API).
//
// Usage:
//   PROJECT_URL=http://localhost:8001 PUBLIC_KEY=pk_live_... SECRET_KEY=sk_live_... \
//     bun run examples/smoke-test.ts
//
// Requires a table named "items" with columns {label: string, qty: number}
// to already exist in the target project (create one via the Studio UI or
// POST /baas/projects/{id}/tables).

import { createClient, BaasApiError } from "../src/client";

const projectUrl = process.env.PROJECT_URL;
const publicKey = process.env.PUBLIC_KEY;
const secretKey = process.env.SECRET_KEY;

if (!projectUrl || !publicKey || !secretKey) {
  console.error("Set PROJECT_URL, PUBLIC_KEY, and SECRET_KEY environment variables first.");
  process.exit(1);
}

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`Assertion failed: ${message}`);
}

async function main() {
  console.log("1. Client-side (public key only) — insert should be rejected client-side, no network call");
  const publicOnlyClient = createClient({ projectUrl: projectUrl!, publicKey: publicKey! });
  let rejectedLocally = false;
  try {
    await publicOnlyClient.data.insert("items", { label: "should not work", qty: 1 });
  } catch (e) {
    rejectedLocally = e instanceof Error && e.message.includes("requires a secret key");
  }
  assert(rejectedLocally, "insert without a secret key must throw the local guard error");
  console.log("   OK — insert without a secret key was rejected before any network call.\n");

  console.log("2. Server-side client (public + secret key) — full record CRUD");
  const serverClient = createClient({ projectUrl: projectUrl!, publicKey: publicKey!, secretKey: secretKey! });

  const inserted = await serverClient.data.insert("items", { label: "Widget", qty: 5 });
  assert(inserted.label === "Widget" && inserted.qty === 5, "inserted record should echo real data");
  console.log("   Inserted:", inserted);

  const fetched = await serverClient.data.get("items", inserted.id);
  assert(fetched.id === inserted.id, "get should return the same record");
  console.log("   Fetched:", fetched);

  const updated = await serverClient.data.update("items", inserted.id, { qty: 9 });
  assert(updated.qty === 9, "update should persist the new value");
  console.log("   Updated:", updated);

  const listed = await publicOnlyClient.data.list("items");
  assert(
    listed.records.some((r) => r.id === inserted.id && r.qty === 9),
    "list (public key only) should see the updated record",
  );
  console.log(`   Listed ${listed.total_count} record(s) via public-key-only read.`);

  const removed = await serverClient.data.remove("items", inserted.id);
  assert(removed.deleted === true, "remove should report deleted: true");
  console.log("   Removed — deleted:", removed.deleted, "\n");

  console.log("3. End-user signup/login (public key only)");
  const email = `smoke-test-${Date.now()}@example.com`;
  const { user, token } = await publicOnlyClient.auth.signup(email, "TestPass123!");
  assert(user.email === email, "signup should return the real created user");
  console.log("   Signed up:", user);

  const loginResult = await publicOnlyClient.auth.login(email, "TestPass123!");
  assert(loginResult.user.id === user.id, "login should resolve to the same user");
  console.log("   Logged in, got token of length", loginResult.token.length);

  const me = await publicOnlyClient.auth.me(token);
  assert(me.id === user.id, "auth.me should return the authenticated user's own identity");
  console.log("   /end-users/me:", me, "\n");

  console.log("4. A bogus record ID should surface a real BaasApiError, not throw something opaque");
  try {
    await serverClient.data.get("items", "does-not-exist");
    throw new Error("expected a BaasApiError for a nonexistent record");
  } catch (e) {
    assert(e instanceof BaasApiError && e.code === 404, "should be a 404 BaasApiError");
    console.log("   OK — got BaasApiError:", (e as BaasApiError).code, (e as BaasApiError).message);
  }

  console.log("\nAll smoke tests passed against the real backend.");
}

main().catch((err) => {
  console.error("Smoke test FAILED:", err);
  process.exit(1);
});
