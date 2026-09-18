import { defineConfig, devices } from "@playwright/test";

// A frontend-only smoke test -- /v1/languages and /v1/predict are mocked
// via page.route() in each spec, so this never needs a live api/ process
// (that surface already has 340+ pytest tests of its own). This just
// checks the pages people actually click through still render and wire up.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:3100",
    trace: "retain-on-failure",
  },
  webServer: {
    // Not `next dev` -- Next 16's dev server refuses to start a second
    // instance in the same project directory at all (a directory-scoped
    // lock, not a port conflict), which collides with a dev server already
    // running locally for manual testing. A production build+start has no
    // such lock and is arguably the more honest thing to smoke-test anyway.
    //
    // In CI, ci.yml's own "Build" step already ran `npm run build` earlier
    // in the same job (same filesystem, same .next output) -- building
    // again here would just be a second full `next build` for zero extra
    // coverage. Locally, `npm run test:e2e` needs to be self-sufficient
    // since there's no earlier build step to rely on.
    command: process.env.CI ? "npm run start -- -p 3100" : "npm run build && npm run start -- -p 3100",
    url: "http://localhost:3100",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
