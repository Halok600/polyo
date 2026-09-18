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
