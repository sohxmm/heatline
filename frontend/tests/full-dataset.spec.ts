import { test, expect } from "@playwright/test";
import fs from "node:fs";

const fit = JSON.parse(
  fs.readFileSync("public/data/full_dataset_fit.json", "utf8"),
);

test("default calendar uses the full-data checkpoint and covers every May", async ({
  page,
}) => {
  await page.goto("/explorer?source=era5&year=1981&day=20");
  await expect(page.getByLabel("Model run")).toHaveValue("full");
  await expect(page.getByLabel("Replay year").locator("option")).toHaveCount(
    46,
  );
  await expect(
    page.getByRole("button", { name: "Previous May", exact: true }),
  ).toBeDisabled();
  await expect(page.getByTestId("selected-day-result")).toContainText(
    "Full-dataset fit · in-sample",
  );
  for (const id of ["era5", "nasa_power"]) {
    await page.getByLabel("Climate source").selectOption(id);
    await page.getByLabel("Replay year").selectOption("2026");
    const source = fit.sources.find((s: { id: string }) => s.id === id);
    const day = source.records.find(
      (r: { date: string; actual: number; prediction: number }) =>
        r.date.startsWith("2026") && r.actual === 0 && r.prediction === 1,
    );
    await page.locator(`[data-date="${day.date}"]`).click();
    await expect(page.getByTestId("selected-day-result")).toContainText(
      "False alarm",
    );
    await expect(page.getByTestId("decision-distance")).toContainText(
      `${((day.probability - day.threshold) * 100).toFixed(2)} percentage points`,
    );
    const details = page.locator(".day-calculation");
    if (!(await details.getAttribute("open")))
      await details.locator("summary").click();
    await expect(details.locator(".day-feature-inputs tbody tr")).toHaveCount(
      source.architecture.inputs,
    );
    await expect(page.locator(".data-note")).toContainText(fit.model_version);
  }
  await page.getByLabel("Model run").selectOption("current");
  await expect(page.locator(".view-context")).toContainText(
    "Held-out rolling evaluation",
  );
  await expect(page.getByLabel("Replay year").locator("option")).toHaveCount(
    31,
  );
});

test("results distinguish full-season fit, May fit and chronological evidence", async ({
  page,
}) => {
  await page.goto("/evidence");
  await expect(page.getByLabel("Evaluation protocol")).toHaveValue("full");
  const source = fit.sources[0];
  await expect(page.locator("table").first()).toContainText(
    `${(source.metrics[0].precision * 100).toFixed(1)}%`,
  );
  await expect(page.locator(".results-context")).toContainText(
    "1426 evaluated days",
  );
  await page.getByLabel("Score dates").selectOption("season");
  await expect(page.locator(".results-context")).toContainText(
    "4186 evaluated days",
  );
  await expect(page.locator("table").first()).toContainText(
    `${(source.all_season_metrics[0].precision * 100).toFixed(1)}%`,
  );
  await page
    .getByText("Evaluation & training protocol", { exact: true })
    .click();
  await expect(page.getByRole("table").nth(1).locator("tbody tr")).toHaveCount(
    5,
  );
  await page.getByLabel("Evaluation protocol").selectOption("rolling");
  await expect(page.locator(".results-context")).toContainText(
    "961 evaluated days",
  );
  await expect(page.getByLabel("Score dates")).toHaveCount(0);
});

test("network defaults to the actual 300-epoch all-data training curve", async ({
  page,
}) => {
  await page.goto("/learn");
  await expect(page.getByLabel("Experiment", { exact: true })).toHaveValue(
    "full",
  );
  await expect(page.getByLabel("Training run")).toHaveValue("final");
  await page.getByText("Exact epoch values", { exact: true }).click();
  const table = page.getByTestId("network-curve-values");
  await expect(table.locator("tbody tr")).toHaveCount(300);
  await expect(table.locator("thead th")).toHaveCount(2);
  const history = fit.sources[0].final_fit.members.find(
    (m: { seed: number }) => m.seed === 42,
  ).training_history;
  await expect(table.locator("tbody tr").last()).toContainText(
    history[299].train_loss.toFixed(6),
  );
  await expect(
    page.getByText("Training runs for 300 fixed epochs", { exact: false }),
  ).toBeVisible();
});
