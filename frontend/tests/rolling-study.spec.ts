import { test, expect } from "@playwright/test";
import fs from "node:fs";

const study = JSON.parse(
  fs.readFileSync("public/data/full_dataset_study.json", "utf8"),
);

test("rolling calendar shows saved held-out results and switches source", async ({
  page,
}) => {
  await page.goto("/explorer");
  for (const sourceId of ["era5", "nasa_power"]) {
    await page.getByLabel("Climate source").selectOption(sourceId);
    const source = study.sources.find((s: { id: string }) => s.id === sourceId);
    const day = source.records.find(
      (r: { date: string }) => r.date === "2026-05-20",
    );
    await expect(
      page
        .getByRole("group", { name: "May 2026 calendar" })
        .locator("[data-date]"),
    ).toHaveCount(31);
    await page.locator('[data-date="2026-05-20"]').click();
    const selected = page.getByTestId("selected-day-result");
    await expect(selected).toContainText("20 May 2026");
    await expect(selected).toContainText(day.tmax.toFixed(1));
    await expect(selected).toContainText(day.raw_score.toFixed(6));
    await expect(selected).toContainText(source.label);
    await expect(selected).toContainText("Held-out rolling evaluation");
    await expect(
      page.getByText("How the model calculated this day", { exact: true }),
    ).toHaveCount(0);
  }
});

test("rolling history has bounded years and errors are not hidden", async ({
  page,
}) => {
  await page.goto("/explorer");
  await expect(page.locator(".history-table tbody tr")).toHaveCount(31);
  await page.getByLabel("Replay year").selectOption("1996");
  await expect(
    page.getByRole("button", { name: "Previous May", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Errors only").check();
  const era5 = study.sources.find((s: { id: string }) => s.id === "era5");
  const errors = era5.records.filter(
    (r: { date: string; prediction: number; actual: number }) =>
      r.date.startsWith("1996") && r.actual !== r.prediction,
  );
  await expect(page.locator(".history-table tbody tr")).toHaveCount(
    errors.length,
  );
  await page.getByLabel("Replay year").selectOption("2026");
  await expect(
    page.getByRole("button", { name: "Next May", exact: true }),
  ).toBeDisabled();
});

test("missing rolling artifact leaves a usable original replay", async ({
  page,
}) => {
  await page.route("**/data/full_dataset_study.json", (route) =>
    route.fulfill({ status: 503, body: "Unavailable" }),
  );
  await page.route("http://127.0.0.1:8999/**", (route) => route.abort());
  await page.goto("/explorer");
  await expect(
    page.getByText(
      "Rolling results are unavailable. The original replay remains accessible above.",
    ),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Original 26-input replay", exact: true })
    .click();
  await expect(page.locator('[data-date="2026-05-20"]')).toBeEnabled();
});
