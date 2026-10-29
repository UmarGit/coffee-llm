import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "qa",
  outputDir: "qa/results",
  fullyParallel: true,
  reporter: [["list"], ["html", { outputFolder: "qa/report", open: "never" }]],
  use: {
    baseURL: "http://localhost:4321/coffee-llm/",
    video: "on",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
  ],
  webServer: {
    command: "npm run build && npm run preview",
    url: "http://localhost:4321/coffee-llm/",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
