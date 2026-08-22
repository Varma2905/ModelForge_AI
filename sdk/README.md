# @regression-studio/baas-sdk

Client SDK for apps built on a [Regression Analysis Studio](../README.md) backend-as-a-service
project. Create a project in the Studio, get a public/secret key pair, and use this SDK to
read/write your project's data and authenticate your app's own end users.

## Security — read this first

- **Public key**: safe to embed in browser/client-side JavaScript. It authorizes reads and
  end-user signup/login only.
- **Secret key**: server-only. **Never** put it in client-side code, a public repo, or ship it
  in a browser bundle. It authorizes writes (`insert`/`update`/`remove`). Call those methods
  from your own backend, not the browser.

The SDK enforces this at the type/runtime level: `data.insert`/`update`/`remove` throw
immediately (before any network call) if no `secretKey` was provided when creating the client.

## Install

```bash
npm install @regression-studio/baas-sdk
```

## Usage

```ts
import { createClient } from "@regression-studio/baas-sdk";

// Client-side (browser) — public key only, read + end-user auth
const client = createClient({
  projectUrl: "https://your-studio-backend.example.com",
  publicKey: "pk_live_...",
});

const { records } = await client.data.list("products");
const product = await client.data.get("products", "some-record-id");

const { user, token } = await client.auth.signup("jane@example.com", "hunter2");
const me = await client.auth.me(token);
```

```ts
// Server-side only — secret key enables writes
const serverClient = createClient({
  projectUrl: "https://your-studio-backend.example.com",
  publicKey: "pk_live_...",
  secretKey: "sk_live_...", // from your server's environment, never the browser
});

await serverClient.data.insert("products", { name: "Widget", price: 9.99 });
await serverClient.data.update("products", recordId, { price: 12.5 });
await serverClient.data.remove("products", recordId);
```

## API

- `client.data.list(table, limit?)` → `{ records, total_count }`
- `client.data.get(table, id)` → record
- `client.data.insert(table, record)` — requires secret key
- `client.data.update(table, id, record)` — requires secret key
- `client.data.remove(table, id)` — requires secret key
- `client.auth.signup(email, password)` → `{ user, token }`
- `client.auth.login(email, password)` → `{ user, token }`
- `client.auth.me(token)` → end user

All methods reject with a `BaasApiError` (`.code`, `.message`) on a real API error.

## Build

```bash
bun install
bun run build      # emits dist/ (compiled JS + .d.ts)
bun run typecheck   # tsc --noEmit
```

See `examples/smoke-test.ts` for a runnable end-to-end example against a real project.
