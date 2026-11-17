import { defineConfig, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const apiOrigin = new URL(process.env.MONORL_API ?? "http://127.0.0.1:8000").origin;

// Requests passed on to the API server. It refuses unsafe requests that name another origin, and a
// browser on the dev server names the dev server: a request from the dev server's own pages goes
// on under the API's origin, any other origin is passed on as it is (and refused there).
const api: ProxyOptions = {
  target: apiOrigin,
  changeOrigin: true,
  configure: (proxy) => {
    proxy.on("proxyReq", (proxyReq, req) => {
      if (req.headers.origin === `http://${req.headers.host}`) proxyReq.setHeader("origin", apiOrigin);
    });
  },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // "/?token=..." is the sign-in URL of `monorl serve`: the API server sets the session cookie and
    // redirects to "/", which Vite serves.
    proxy: { "/api": api, "^/\\?token=": api },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        // One vendor chunk: splitting recharts from react creates a chunk cycle where the charts
        // chunk evaluates before React is initialised (`forwardRef` of undefined) in production.
        manualChunks: (id: string) => (id.includes("node_modules") ? "vendor" : undefined),
      },
    },
  },
});
