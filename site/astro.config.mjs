import { defineConfig } from "astro/config";

// Cross-origin isolation lets the in-browser model use every CPU core
// (SharedArrayBuffer). The dev and preview servers send the headers here;
// GitHub Pages cannot send headers, so there public/coi-serviceworker.min.js
// adds them from a service worker instead.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  // Served at https://umargit.github.io/coffee-llm/ by .github/workflows/pages.yml.
  site: "https://umargit.github.io",
  base: "/coffee-llm",
  output: "static",
  build: { inlineStylesheets: "auto" },
  server: { headers: isolation },
  vite: {
    optimizeDeps: { exclude: ["@wllama/wllama"] },
  },
});
