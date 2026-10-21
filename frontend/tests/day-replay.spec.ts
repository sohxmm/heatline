import { test, expect } from "@playwright/test";
import fs from "node:fs";
import AxeBuilder from "@axe-core/playwright";
const trace = JSON.parse(
  fs.readFileSync("public/data/replay/2026.json", "utf8"),
);

test("calendar selection shows the real daily prediction and forward computation", async ({
  page,
}) => {
  await page.goto("/explorer");
  await expect(
    page
      .getByRole("group", { name: "May 2026 calendar" })
      .locator("[data-date]"),
  ).toHaveCount(31);
  await page.locator('[data-date="2026-05-20"]').click();
  const day = trace.records.find(
    (r: { date: string }) => r.date === "2026-05-20",
  );
  const selected = page.getByTestId("selected-day-result");
  await expect(selected).toContainText("20 May 2026");
  await expect(selected).toContainText("Missed event");
  await expect(selected).toContainText("19 May");
  await expect(selected).toContainText(day.score.toFixed(6));
  await expect(selected).toContainText(
    `${(day.probability * 100).toFixed(1)}%`,
  );
  await expect(page.locator('[data-date="2026-05-20"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page
    .getByText("How the model calculated this day", { exact: true })
    .click();
  await expect(page.locator(".day-calculation tbody tr")).toHaveCount(26);
  await expect(page.locator(".day-trace-values")).toContainText(
    day.output_logit.toFixed(6),
  );
  const first = page.locator(".day-calculation tbody tr").first();
  await expect(first).toContainText(day.inputs.tmax_lag1.toFixed(4));
  await expect(first).toContainText(day.normalized_inputs.tmax_lag1.toFixed(4));
  const accessibility = await new AxeBuilder({ page })
    .include("#day-replay")
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(accessibility.violations.map((v) => v.id)).toEqual([]);
  await page.locator('[data-date="2026-05-20"]').press("ArrowRight");
  await expect(selected).toContainText("21 May 2026");
  await expect(selected).toContainText(
    trace.records
      .find((r: { date: string }) => r.date === "2026-05-21")
      .score.toFixed(6),
  );
});

test("changing year cannot display a previous year input trace", async ({
  page,
}) => {
  await page.goto("/explorer");
  await expect(page.locator('[data-date="2026-05-20"]')).toBeEnabled();
  await page.locator('[data-date="2026-05-20"]').click();
  await page.getByLabel("Replay year").selectOption("2019");
  await expect(page.getByTestId("selected-day-result")).toContainText(
    "20 May 2019",
  );
  await expect(page.locator('[data-date="2019-05-20"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.getByLabel("Replay year").selectOption("1981");
  await expect(
    page.getByRole("button", { name: "Previous May", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Replay year").selectOption("2026");
  await expect(
    page.getByRole("button", { name: "Next May", exact: true }),
  ).toBeDisabled();
});

test("missing detail files keep published daily results and show unavailability", async ({
  page,
}) => {
  await page.route("**/data/replay/2026.json", (route) =>
    route.fulfill({ status: 404, body: "Unavailable" }),
  );
  await page.goto("/explorer");
  await expect(page.locator('[data-date="2026-05-20"]')).toBeEnabled();
  await page.locator('[data-date="2026-05-20"]').click();
  await expect(page.getByTestId("selected-day-result")).toContainText(
    "Missed event",
  );
  await page
    .getByText("How the model calculated this day", { exact: true })
    .click();
  await expect(page.locator(".day-calculation")).toContainText(
    "input trace is unavailable",
  );
});
