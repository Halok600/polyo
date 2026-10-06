import { expect, test } from "@playwright/test";

const LANGUAGES = [
  { id: "python", tier: 1 },
  { id: "cpp", tier: 1 },
  { id: "java", tier: 1 },
  { id: "javascript", tier: 1 },
  { id: "c", tier: 2 },
  { id: "go", tier: 2 },
];

const PREDICT_RESPONSE = {
  language_detected: "python",
  time: {
    class: "O(n)",
    rank: 2,
    confidence: 0.8,
    distribution: { "O(1)": 0.05, "O(n)": 0.8, "O(n^2)": 0.15 },
    conformal_set: ["O(n)"],
    conformal_coverage: 0.9,
    abstain: false,
  },
  space: {
    class: "O(1)",
    rank: 0,
    confidence: 0.9,
    distribution: { "O(1)": 0.9, "O(n)": 0.1 },
    conformal_set: ["O(1)"],
    conformal_coverage: 0.9,
    abstain: false,
  },
  attribution: [{ feature: "HASH_LOOKUP", contribution: 1, spans: [[1, 4, 1, 8]] }],
  curve: {
    n: [10, 100, 1000],
    time: { predicted_class: "O(n)", series: { "O(1)": [1, 1, 1], "O(n)": [10, 100, 1000] } },
    space: { predicted_class: "O(1)", series: { "O(1)": [1, 1, 1] } },
  },
  ir: { nodes: 5, edges: 4, histogram: {} },
  warnings: [],
};

test.beforeEach(async ({ page }) => {
  await page.route("**/v1/languages", (route) => route.fulfill({ json: LANGUAGES }));
  await page.route("**/v1/predict", (route) => route.fulfill({ json: PREDICT_RESPONSE }));
});

test("landing page renders with a way into the tool", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Big-O/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "RUN THE BENCH →" })).toBeVisible();
});

test("clicking through the landing page reaches the tool and runs a prediction", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await expect(page.getByRole("button", { name: "RUN ANALYSIS" })).toBeVisible();

  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect(page.getByText("CH.05 — SOURCE")).toBeVisible();
});

test("the example gallery swaps both the language and the code", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "Go — merge sort" }).click();
  await expect(page.getByRole("button", { name: "LANG" })).toContainText("go");
  await expect(page.locator(".cm-content")).toContainText("package main");
});

// What the API returns when the static engine answers: the exact expression, who answered and how
// sure it is, and the derivation. The probability bars describe the learned model and must not appear.
const SYMBOLIC_RESPONSE = {
  ...PREDICT_RESPONSE,
  engine: "symbolic",
  entry: "f",
  time: {
    ...PREDICT_RESPONSE.time,
    class: "O(n^2)",
    engine: "symbolic",
    certainty: "assumed",
    expression: "O(n * m)",
    extended_class: "O(n^2)",
    projection_lossy: true,
  },
  space: {
    ...PREDICT_RESPONSE.space,
    engine: "symbolic",
    certainty: "certain",
    expression: "O(1)",
    extended_class: "O(1)",
    projection_lossy: false,
  },
  assumptions: [{ line: 2, reason: "loop bound assumed" }],
  derivation: [
    { line: 2, kind: "loop", text: "loop runs O(n) times; the whole loop costs O(n * m)" },
    { line: 0, kind: "total", text: "time: O(n * m)" },
  ],
};

test("a static-engine answer leads with the expression, shows its derivation and hides the model's bars", async ({
  page,
}) => {
  await page.route("**/v1/predict", (route) => route.fulfill({ json: SYMBOLIC_RESPONSE }));
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();

  await expect(page.getByTestId("time-headline")).toHaveText("O(n * m)");
  // One badge per dimension, time then space.
  await expect(page.locator(".answer-badge")).toHaveText(["ASSUMED", "CERTAIN"]);
  await expect(page.getByText(/HOW THE COST WAS DERIVED — f/)).toBeVisible();
  await expect(page.getByLabel("Assumptions")).toContainText("loop bound assumed");
  await expect(page.getByRole("note", { name: /Line 2/ })).toBeVisible();

  // Probability bars are the model's distribution: absent for both symbolic dimensions.
  await expect(page.getByText("CH.04 — SOURCE")).toBeVisible();
  await expect(page.getByText(/MODEL CLASS PROBABILITY/)).toHaveCount(0);
});

// --- layout: the two columns share a top line and end on a shared bottom line --------------------

const box = async (page: import("@playwright/test").Page, selector: string, nth = 0) => {
  const rect = await page.locator(selector).nth(nth).boundingBox();
  if (!rect) throw new Error(`no box for ${selector}`);
  return { top: rect.y, bottom: rect.y + rect.height };
};

test("idle: the two columns open on one line and the playground ends with the setup column", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await expect(page.getByTestId("playground")).toBeVisible();

  const setup = await box(page, "form .bench-colhead");
  const readout = await box(page, ".bench-readout .bench-colhead");
  expect(readout.top).toBeCloseTo(setup.top, 0);

  // the first panel on the right starts where the intro text on the left starts
  const intro = await box(page, "form > p.bench-intro");
  const panel = await box(page, ".bench-readout > .bench-panel");
  expect(panel.top).toBeCloseTo(intro.top, 0);

  // and the playground finishes where the left column finishes
  const form = await box(page, "form");
  expect(panel.bottom).toBeCloseTo(form.bottom, 0);
});

test("results: panels share the top line, and the growth panel ends where the code input ends", async ({
  page,
}) => {
  await page.route("**/v1/predict", (route) => route.fulfill({ json: SYMBOLIC_RESPONSE }));
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect(page.getByTestId("time-headline")).toBeVisible();

  const intro = await box(page, "form > p.bench-intro");
  const timePanel = await box(page, ".bench-readout section > div .bench-panel", 0);
  const spacePanel = await box(page, ".bench-readout section > div .bench-panel", 1);
  expect(timePanel.top).toBeCloseTo(intro.top, 0);
  expect(spacePanel.top).toBeCloseTo(timePanel.top, 0);

  // the growth panel is stretched after layout settles, so poll instead of reading once
  await expect
    .poll(async () => {
      const growth = await box(page, ".bench-match");
      const input = await box(page, "form .bench-panel");
      return Math.abs(growth.bottom - input.bottom);
    })
    .toBeLessThan(1);
});

// --- the idle playground --------------------------------------------------------------------------

test("idle playground: the slider drives n, a lane explains itself, and clicking jumps to its breaking point", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();

  // taking the slider stops the autoplay and sets n on a log scale (500 of 1000 is n = 100)
  await page.getByRole("slider", { name: "Input size n" }).fill("500");
  await expect(page.getByTestId("playground-n")).toHaveText("n = 100");
  await expect(page.locator('.pg-lane[data-verdict="never"]')).toHaveCount(1); // only 2^n is out of reach at n = 100

  await page.getByRole("button", { name: /^O\(2\^n\),/ }).hover();
  await expect(page.getByTestId("playground-footer")).toContainText("Passes 1 second at n = 30");

  await page.getByRole("button", { name: /^O\(2\^n\),/ }).click();
  await expect(page.getByTestId("playground-n")).toHaveText("n = 30");
});

test("idle playground: runs by itself, and a result replaces it", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  const first = await page.getByTestId("playground-n").innerText();
  await expect.poll(() => page.getByTestId("playground-n").innerText()).not.toBe(first);

  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect(page.getByTestId("time-headline")).toBeVisible();
  await expect(page.getByTestId("playground")).toHaveCount(0);
});

// --- the hand-off from the playground to the first result -----------------------------------------

test("hand-off: scans while waiting, collapses to the predicted class, then shows the results", async ({ page }) => {
  await page.route("**/v1/predict", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1200)); // a slowish API
    await route.fulfill({ json: SYMBOLIC_RESPONSE });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();

  // Record what the page shows in each phase as it happens. Polling for a one-second phase between
  // separate assertions would be a race on a slow machine; an observer sees every state.
  await page.evaluate(() => {
    const seen = { phases: [] as string[], lockedWhileScanning: false, analysing: false, answerLane: "" };
    (window as unknown as { __seen: typeof seen }).__seen = seen;
    const look = () => {
      const pg = document.querySelector('[data-testid="playground"]');
      if (!pg) return;
      const phase = pg.getAttribute("data-phase") ?? "";
      if (seen.phases[seen.phases.length - 1] !== phase) seen.phases.push(phase);
      if (phase === "scanning") {
        seen.lockedWhileScanning = pg.hasAttribute("inert");
        seen.analysing = (pg.textContent ?? "").includes("ANALYSING");
      }
      const lane = document.querySelector('.pg-lane-list > li[data-answer="true"]');
      if (lane) seen.answerLane = lane.textContent ?? "";
    };
    new MutationObserver(look).observe(document.body, { subtree: true, childList: true, attributes: true });
    look();
  });

  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect(page.getByTestId("time-headline")).toHaveText("O(n * m)", { timeout: 15000 });

  const seen = await page.evaluate(
    () => (window as unknown as { __seen: { phases: string[]; lockedWhileScanning: boolean; analysing: boolean; answerLane: string } }).__seen,
  );
  // idle -> scanning -> collapsing, in that order, then the playground is gone
  expect(seen.phases).toEqual(["idle", "scanning", "collapsing"]);
  expect(seen.lockedWhileScanning).toBe(true);
  expect(seen.analysing).toBe(true);
  // the lane that stayed is the predicted time class (O(n^2)), tagged as the visitor's code
  expect(seen.answerLane).toContain("O(n^2)");
  expect(seen.answerLane).toContain("YOUR CODE");
  await expect(page.getByTestId("playground")).toHaveCount(0);
});

test("hand-off: even an instant answer is given time to be seen", async ({ page }) => {
  await page.route("**/v1/predict", (route) => route.fulfill({ json: SYMBOLIC_RESPONSE }));
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  const clickedAt = Date.now();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect(page.getByTestId("time-headline")).toBeVisible({ timeout: 15000 });
  // at least the minimum scan plus the collapse: not an instant swap. (Machine load can only make
  // this longer, so the lower bound is safe to assert; the phases themselves are checked above.)
  expect(Date.now() - clickedAt).toBeGreaterThan(1500);
});

test("hand-off: a failed request puts the playground back and says what went wrong", async ({ page }) => {
  await page.route("**/v1/predict", (route) => route.fulfill({ status: 500, json: { detail: "boom" } }));
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  // (Next adds its own role=alert route announcer, so match the app's message by its text)
  await expect(page.getByText("[!] boom")).toBeVisible();
  await expect(page.getByTestId("playground")).toHaveAttribute("data-phase", "idle");
  await expect(page.getByTestId("playground")).not.toHaveAttribute("inert", "");
});

test("hand-off: with reduced motion the results simply appear, no scan and no collapse", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.route("**/v1/predict", (route) => route.fulfill({ json: SYMBOLIC_RESPONSE }));
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  // well inside the 1.8s the animated hand-off alone takes
  await expect(page.getByTestId("time-headline")).toBeVisible({ timeout: 1200 });
  await expect(page.getByTestId("playground")).toHaveCount(0);
});

test("hand-off: results arrive in a stagger, and the layout still lines up once it settles", async ({ page }) => {
  await page.route("**/v1/predict", (route) => route.fulfill({ json: SYMBOLIC_RESPONSE }));
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect(page.getByTestId("time-headline")).toBeVisible();
  await page.waitForTimeout(1200); // the entrance animations finish

  const intro = await box(page, "form > p.bench-intro");
  const timePanel = await box(page, ".bench-readout section > div .bench-panel", 0);
  expect(timePanel.top).toBeCloseTo(intro.top, 0);
  await expect
    .poll(async () => {
      const growth = await box(page, ".bench-match");
      const input = await box(page, "form .bench-panel");
      return Math.abs(growth.bottom - input.bottom);
    })
    .toBeLessThan(1);
});

// --- the browser tab: icon and title ----------------------------------------------------------------

const iconHrefs = (page: import("@playwright/test").Page) =>
  page.evaluate(() => [...document.querySelectorAll('link[rel~="icon"]')].map((link) => link.getAttribute("href") ?? ""));

test("tab: a real title and PolyO's own icons, not the framework defaults", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle(/PolyO — know the Big-O before you run it/);
  const hrefs = await iconHrefs(page);
  expect(hrefs.some((href) => href.includes("icon.svg"))).toBe(true);
  expect(hrefs.some((href) => href.includes("favicon.ico"))).toBe(true);
  await expect(page.locator('link[rel="apple-touch-icon"]')).toHaveCount(1);
  // each of them is served and is the size it claims to be
  for (const path of ["/icon.svg", "/favicon.ico", "/apple-icon.png"]) {
    const response = await page.request.get(path);
    expect(response.ok(), path).toBe(true);
    expect((await response.body()).length, path).toBeGreaterThan(200);
  }
});

test("tab: the title and icon show the analysis running, then the answer", async ({ page }) => {
  await page.route("**/v1/predict", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    await route.fulfill({ json: SYMBOLIC_RESPONSE });
  });
  await page.goto("/");
  const idleTitle = await page.title();
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();

  // while running: the title says so and the icon is the spinning canvas frame
  await expect.poll(() => page.title()).toBe("Analysing… · PolyO");
  await expect.poll(async () => (await iconHrefs(page)).every((href) => href.startsWith("data:image/png"))).toBe(true);

  // afterwards: the answer leads the title and the real icons are back
  await expect(page.getByTestId("time-headline")).toBeVisible({ timeout: 15000 });
  await expect.poll(() => page.title()).toBe("O(n * m) · PolyO");
  const after = await iconHrefs(page);
  expect(after.some((href) => href.startsWith("data:"))).toBe(false);
  expect(after.some((href) => href.includes("icon.svg"))).toBe(true);
  expect(after.some((href) => href.includes("favicon.ico"))).toBe(true);
  expect(idleTitle).not.toBe("O(n * m) · PolyO");
});

test("tab: a failed request puts the plain title back", async ({ page }) => {
  await page.route("**/v1/predict", (route) => route.fulfill({ status: 500, json: { detail: "boom" } }));
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect(page.getByText("[!] boom")).toBeVisible();
  await expect(page).toHaveTitle(/PolyO — know the Big-O before you run it/);
  expect((await iconHrefs(page)).some((href) => href.startsWith("data:"))).toBe(false);
});

test("tab: reduced motion still updates the title but never animates the icon", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.route("**/v1/predict", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 600));
    await route.fulfill({ json: SYMBOLIC_RESPONSE });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "RUN THE BENCH →" }).click();
  await page.getByRole("button", { name: "RUN ANALYSIS" }).click();
  await expect.poll(() => page.title()).toBe("Analysing… · PolyO");
  expect((await iconHrefs(page)).some((href) => href.startsWith("data:"))).toBe(false);
  await expect.poll(() => page.title()).toBe("O(n * m) · PolyO");
});
