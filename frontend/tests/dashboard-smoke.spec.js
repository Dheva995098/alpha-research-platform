import { expect, test } from "@playwright/test";

test("dashboard renders main operational controls", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Alpha Research Platform" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Generate" }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Accounts", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Queue" }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Results" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Vault" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Rank" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Worker" })).toBeVisible();
  await page.getByRole("button", { name: "Accounts", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Add Account" })).toBeVisible();
  await page.getByRole("button", { name: "Results" }).click();
  await expect(page.getByRole("heading", { name: "Results" })).toBeVisible();
  await page.getByRole("button", { name: "Vault" }).click();
  await expect(page.getByRole("heading", { name: "Good Live Alphas" })).toBeVisible();
  await expect(pageErrors).toEqual([]);
});
