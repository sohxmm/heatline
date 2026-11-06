import { test, expect } from "@playwright/test";
import fs from "node:fs";

const study = JSON.parse(
  fs.readFileSync("public/data/full_dataset_study.json", "utf8"),
);

test("rolling calendar shows saved held-out results and switches source", async ({
  page,
}) => {
  await page.goto("/explorer?run=current");
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
    ).toBeVisible();
  }
});

test("rolling history has bounded years and errors are not hidden", async ({
  page,
}) => {
  await page.goto("/explorer?run=current");
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
  await page.goto("/explorer?run=current");
  await expect(
    page.getByText(
      "Rolling results are unavailable. The original replay remains accessible above.",
    ),
  ).toBeVisible();
  await page.getByLabel("Model run").selectOption("archive");
  await expect(page.locator('[data-date="2026-05-20"]')).toBeEnabled();
});

test("false alarms expose actual outcomes, score margin and source-specific computation", async ({
  page,
}) => {
  await page.goto("/explorer?run=current");
  for (const sourceId of ["era5", "nasa_power"]) {
    const source = study.sources.find((s: { id: string }) => s.id === sourceId);
    const day = source.records.find(
      (r: { date: string; actual: number; prediction: number }) =>
        r.date.startsWith("2026") && r.actual === 0 && r.prediction === 1,
    );
    expect(day).toBeTruthy();
    await page.getByLabel("Climate source").selectOption(sourceId);
    await page.locator(`[data-date="${day.date}"]`).click();
    const selected = page.getByTestId("selected-day-result");
    await expect(selected).toContainText("False alarm");
    await expect(selected).toContainText("No heatwave proxy");
    await expect(selected).toContainText(`${day.tmax.toFixed(1)}°C`);
    await expect(page.getByTestId("decision-distance")).toContainText(
      `${((day.probability - day.threshold) * 100).toFixed(2)} percentage points`,
    );
    await expect(page.getByTestId("observed-criteria")).toContainText(
      `${Math.abs(day.tmax - 45).toFixed(2)}°C`,
    );
    const details = page.locator(".day-calculation");
    if (!(await details.getAttribute("open")))
      await details.locator("summary").click();
    await expect(details.locator(".day-feature-inputs tbody tr")).toHaveCount(
      sourceId === "era5" ? 26 : 52,
    );
    const trace = JSON.parse(
      fs.readFileSync(`public/data/study/${sourceId}/2026.json`, "utf8"),
    );
    const actualTrace = trace.records.find(
      (r: { date: string }) => r.date === day.date,
    );
    await expect(details.locator(".day-trace-values")).toContainText(
      actualTrace.output_logit.toFixed(6),
    );
    await expect(details).toContainText(
      `${source.architecture.inputs} → 16 → 1`,
    );
  }
});

test("direct date links choose the requested source and date", async ({
  page,
}) => {
  await page.goto("/explorer?run=current&source=nasa_power&year=2019&day=17");
  await expect(page.getByLabel("Climate source")).toHaveValue("nasa_power");
  await expect(page.getByLabel("Replay year")).toHaveValue("2019");
  await expect(page.getByTestId("selected-day-result")).toContainText(
    "17 May 2019",
  );
});

test("missing current computation preserves result and distance", async ({
  page,
}) => {
  await page.route("**/data/study/era5/2026.json", (route) =>
    route.fulfill({ status: 404, body: "Unavailable" }),
  );
  await page.route("http://127.0.0.1:8999/**", (route) => route.abort());
  await page.goto("/explorer?run=current&day=20");
  await expect(page.getByTestId("decision-distance")).toContainText(
    "percentage points",
  );
  await page
    .getByText("How the model calculated this day", { exact: true })
    .click();
  await expect(page.locator(".day-calculation")).toContainText(
    "input trace is unavailable",
  );
  await expect(page.getByTestId("selected-day-result")).toContainText(
    "20 May 2026",
  );
});
