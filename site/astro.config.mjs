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
  // Served at https://wrong.cuyor.com/ by .github/workflows/pages.yml (custom domain
  // set in the repo's Settings -> Pages; DNS: CNAME wrong -> umargit.github.io).
  site: "https://wrong.cuyor.com",
  base: "/",
  output: "static",
  build: { inlineStylesheets: "auto" },
  server: { headers: isolation },
  vite: {
    optimizeDeps: { exclude: ["@wllama/wllama"] },
  },
});
