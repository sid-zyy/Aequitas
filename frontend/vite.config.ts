import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to the FastAPI server; `npm run build` writes the app that FastAPI serves.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../aequitas/web/frontend_dist", emptyOutDir: true },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
