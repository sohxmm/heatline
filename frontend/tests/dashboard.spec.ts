import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
const history = JSON.parse(fs.readFileSync("public/data/history.json", "utf8"));
const outlook = JSON.parse(fs.readFileSync("public/data/outlook.json", "utf8"));
const study = JSON.parse(
  fs.readFileSync("public/data/full_dataset_fit.json", "utf8"),
);

test("all five views load and remain accessible", async ({ page }) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const path of ["/", "/outlook", "/explorer", "/learn", "/evidence"]) {
    await page.goto(path);
    await expect(page.locator("h1")).toBeVisible();
    await expect(page.locator(".data-note").first()).toContainText(
      /Open-Meteo|Heatline|ERA5|NASA POWER/,
    );
    const result = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      result.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.target),
      })),
    ).toEqual([]);
  }
  expect(errors).toEqual([]);
});

test("replay numbers and errors equal published records", async ({ page }) => {
  await page.goto("/explorer");
  await page.getByLabel("Model run").selectOption("archive");
  await expect(page.locator(".history-table tbody tr")).toHaveCount(31);
  const expected = history.records.filter((r: { date: string }) =>
    r.date.startsWith("2026"),
  );
  expect(
    await page.locator(".history-table tbody tr").first().innerText(),
  ).toContain(expected[0].tmax.toFixed(1));
  expect(
    await page.locator(".history-table tbody tr").first().innerText(),
  ).toContain(expected[0].score.toFixed(3));
  await page.getByLabel("Replay year").selectOption("2019");
  await page.getByLabel("Errors only").check();
  const errors = history.records.filter(
    (r: { date: string; actual: number; prediction: number }) =>
      r.date.startsWith("2019") && r.actual !== r.prediction,
  );
  await expect(page.locator(".history-table tbody tr")).toHaveCount(
    errors.length,
  );
  await expect(
    page
      .locator(".view-context")
      .getByText("Historical replay", { exact: true }),
  ).toBeVisible();
});

test("provider weather is distinct, with matching values and city controls", async ({
  page,
}) => {
  await page.goto("/outlook");
  await expect(page.locator(".day-selector button")).toHaveCount(7);
  await page
    .getByRole("button", { name: "Mumbai", exact: true })
    .first()
    .click();
  const city = outlook.cities.find((c: { id: string }) => c.id === "mumbai");
  await expect(page.locator(".metric").first()).toContainText(
    city.daily[0].tmax.toFixed(1),
  );
  await page.locator(".day-selector button").nth(2).click();
  await page.getByText("View hourly measurements", { exact: true }).click();
  await expect(page.locator("table").first().locator("tbody tr")).toHaveCount(
    24,
  );
  await expect(page.locator("table").last().locator("tbody tr")).toHaveCount(6);
  await expect(
    page.getByText("Weather-provider forecast", { exact: true }),
  ).toBeVisible();
});

test("learning fixture and evidence match published data", async ({ page }) => {
  await page.goto("/learn");
  await page
    .getByText("Lab exercise · 2–2–1 backpropagation", { exact: true })
    .click();
  await page.getByRole("tab", { name: "2 Forward pass" }).click();
  await expect(page.getByRole("tabpanel")).toContainText("0.581031948");
  await page.getByRole("tab", { name: "3 Error signals" }).click();
  await expect(page.getByRole("tabpanel")).toContainText("0.101990994");
  await page.goto("/evidence");
  const era5 = study.sources.find((s: { id: string }) => s.id === "era5");
  await expect(page.locator("table").first().locator("tbody tr")).toHaveCount(
    era5.metrics.length,
  );
  await expect(page.locator("table").first()).toContainText(
    era5.metrics[0].pr_auc.toFixed(3),
  );
});

test("city deep links and comparison controls change actual chart data", async ({
  page,
}) => {
  await page.goto("/outlook?city=mumbai");
  await expect(
    page.getByRole("button", { name: "Mumbai", exact: true }).first(),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByLabel("Comparison measurement").selectOption("apparent_tmax");
  await expect(page.getByLabel("Comparison measurement")).toHaveValue(
    "apparent_tmax",
  );
  for (const name of ["Nagpur", "Mumbai", "Pune"])
    await page
      .getByRole("group", { name: "Cities to compare" })
      .getByLabel(name, { exact: true })
      .uncheck();
  await expect(
    page.getByText("Select at least one city to draw a comparison."),
  ).toBeVisible();
  await page
    .getByRole("group", { name: "Cities to compare" })
    .getByLabel("Akola", { exact: true })
    .check();
  await expect(page.getByRole("img", { name: /Akola chart/ })).toBeVisible();
});

test("dashboard city and May-day controls open the selected records", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Forecast city").selectOption("mumbai");
  const mumbai = outlook.cities.find((c: { id: string }) => c.id === "mumbai");
  await expect(page.locator(".dashboard-temperature")).toContainText(
    mumbai.daily[0].tmax.toFixed(1),
  );
  await page.getByLabel("Year", { exact: true }).selectOption("2019");
  await page.getByLabel("Day in May").selectOption("17");
  await page.getByRole("link", { name: "Open day", exact: true }).click();
  await expect(page.getByTestId("selected-day-result")).toContainText(
    "17 May 2019",
  );
});

test("results switch source-specific architectures and offer real data exports", async ({
  page,
}) => {
  await page.goto("/evidence");
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "Model results",
  );
  await expect(
    page.getByText("29.4% was precision.", { exact: true }),
  ).toHaveCount(0);
  for (const sourceId of ["era5", "nasa_power"]) {
    await page.getByLabel("Climate source").selectOption(sourceId);
    const source = study.sources.find((s: { id: string }) => s.id === sourceId);
    await expect(page.locator(".results-context")).toContainText(
      `${source.architecture.inputs} → ${source.architecture.hidden} → 1`,
    );
    await expect(page.locator("table").first()).toContainText(
      `${(source.metrics[0].precision * 100).toFixed(1)}%`,
    );
  }
  const response = await page.request.get("/data/full_dataset_study.json");
  expect(response.ok()).toBeTruthy();
  await page.getByText("Dataset & target definition", { exact: true }).click();
  await expect(
    page.getByRole("link", { name: "Station comparison", exact: true }),
  ).toHaveAttribute("download", "");
});

test("old snapshots and an unavailable API retain explicit freshness", async ({
  page,
}) => {
  const old = {
    ...outlook,
    issued_at: "2000-01-01T00:00:00Z",
    availability: "stale",
    cities: outlook.cities.map((c: object) => ({
      ...c,
      issued_at: "2000-01-01T00:00:00Z",
      availability: "stale",
    })),
  };
  await page.route("**/data/outlook.json", (route) =>
    route.fulfill({ json: old }),
  );
  await page.route("http://127.0.0.1:8999/**", (route) => route.abort());
  await page.goto("/outlook");
  await expect(page.locator(".data-note")).toContainText("Stale snapshot");
  await expect(page.locator(".data-note")).toContainText("saved snapshot");
  await expect(page.locator(".data-note")).toContainText(
    "API is waking or unavailable",
  );
  await expect(page.locator(".data-note")).toContainText("2000");
});

test("mobile layout, navigation and reduced motion", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const path of ["/", "/outlook", "/explorer", "/learn", "/evidence"]) {
    await page.goto(path);
    await expect(page.locator("h1")).toBeVisible();
    const overflow = await page.evaluate(() =>
      Array.from(document.querySelectorAll("body *"))
        .filter((el) => el.getBoundingClientRect().right > innerWidth + 1)
        .map((el) => ({
          tag: el.tagName,
          cls: el.className,
          right: el.getBoundingClientRect().right,
        }))
        .slice(0, 12),
    );
    await expect
      .poll(
        () =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth,
          ),
        { message: `${path}: ${JSON.stringify(overflow)}` },
      )
      .toBeTruthy();
  }
  await page.goto("/");
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page
    .getByRole("navigation")
    .getByRole("link", { name: "Regional outlook" })
    .click();
  await expect(page).toHaveURL(/outlook/);
  await page.goto("/");
  expect(
    await page
      .locator(".dashboard-page")
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe("none");
});
