import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [
    tailwindcss(),
    tanstackStart({
      server: { entry: "server" },
    }),
    react(),
    tsconfigPaths(),
  ],
  server: {
    port: 3001,
    host: true,
  },
  define: {
    // Hard-pinned so a stray shell/system VITE_API_URL env var (which Vite
    // would otherwise prefer over frontend/.env) can never silently
    // override the backend URL again.
    "import.meta.env.VITE_API_URL": JSON.stringify("http://localhost:8001"),
  },
});
