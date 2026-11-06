import { test, expect } from "@playwright/test";
import fs from "node:fs";

const full = JSON.parse(
  fs.readFileSync("public/data/full_dataset_fit.json", "utf8"),
);
const study = JSON.parse(
  fs.readFileSync("public/data/full_dataset_study.json", "utf8"),
);

test("network source and seed controls display their own architecture and initialization", async ({
  page,
}) => {
  await page.goto("/learn");
  for (const id of ["era5", "nasa_power"]) {
    const source = full.sources.find((s: { id: string }) => s.id === id);
    await page.getByLabel("Network source", { exact: true }).selectOption(id);
    await expect(
      page.getByTestId("network-architecture").locator("b").first(),
    ).toHaveText(String(source.architecture.inputs));
    await expect(page.getByTestId("network-settings")).toContainText(
      source.final_fit.positive_weight.toFixed(3),
    );
    await page.getByLabel("Network member", { exact: true }).selectOption("7");
    await page
      .getByText("Seed 7 · inspect initial connections", { exact: true })
      .click();
    const initial = source.initialization.members.find(
      (m: { seed: number }) => m.seed === 7,
    );
    await expect(
      page.getByTestId("network-initial-values").locator("tbody tr").first(),
    ).toContainText(initial.first_input_weights[0].toFixed(6));
    await expect(
      page.getByTestId("network-initial-values").locator("tbody tr").first(),
    ).toContainText(initial.output_weights[0].toFixed(6));
    await page
      .getByText("Seed 7 · inspect initial connections", { exact: true })
      .click();
  }
});

test("run and member controls select recorded curves, while full-data fit has no stopping curve", async ({
  page,
}) => {
  await page.goto("/learn");
  await page.getByLabel("Experiment", { exact: true }).selectOption("rolling");
  const source = study.sources.find((s: { id: string }) => s.id === "era5");
  const fold = source.folds[0];
  await page.getByLabel("Training run", { exact: true }).selectOption(fold.id);
  await page.getByLabel("Network member", { exact: true }).selectOption("21");
  await page.getByText("Exact epoch values", { exact: true }).click();
  const history = fold.members.find(
    (m: { seed: number }) => m.seed === 21,
  ).training_history;
  const table = page.getByTestId("network-curve-values");
  await expect(table.locator("tbody tr")).toHaveCount(history.length);
  await expect(table.locator("tbody tr").first()).toContainText(
    history[0].train_loss.toFixed(6),
  );
  await expect(table.locator("tbody tr").first()).toContainText(
    history[0].val_loss.toFixed(6),
  );
  await page.getByLabel("Network member", { exact: true }).selectOption("mean");
  await expect(table.locator("tbody tr")).toHaveCount(
    fold.training_history.length,
  );
  await expect(table.locator("tbody tr").first()).toContainText(
    fold.training_history[0].train_loss.toFixed(6),
  );
  await page.getByLabel("Training run", { exact: true }).selectOption("final");
  await expect(table.locator("thead th")).toHaveCount(2);
  await expect(table.locator("tbody tr")).toHaveCount(
    source.final_fit.training_history.length,
  );
  await expect(table.locator("tbody tr").first()).toContainText(
    source.final_fit.training_history[0].train_loss.toFixed(6),
  );
  await expect(
    page.getByText("Its curve is training fit only", { exact: false }),
  ).toBeVisible();
});

test("missing model metadata never substitutes an older curve and leaves the teaching fixture usable", async ({
  page,
}) => {
  await page.route("**/data/full_dataset_fit.json", (route) =>
    route.fulfill({ status: 503, body: "Unavailable" }),
  );
  await page.route("http://127.0.0.1:8999/**", (route) => route.abort());
  await page.goto("/learn");
  await expect(
    page.getByText("Current network metadata is unavailable.", {
      exact: false,
    }),
  ).toBeVisible();
  await expect(page.getByTestId("network-curve-values")).toHaveCount(0);
  await page
    .getByText("Lab exercise · 2–2–1 backpropagation", { exact: true })
    .click();
  await expect(
    page.getByText(
      "separate teaching fixture, not the trained weather network",
      { exact: false },
    ),
  ).toBeVisible();
  await page.getByRole("tab", { name: "2 Forward pass" }).click();
  await expect(page.getByRole("tabpanel")).toContainText("0.581031948");
  await page.getByRole("tab", { name: "3 Error signals" }).click();
  await expect(page.getByRole("tabpanel")).toContainText("0.101990994");
});
