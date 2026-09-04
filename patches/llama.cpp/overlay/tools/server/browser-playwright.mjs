#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const MAX_ACTIONS = 20;
const MAX_DIAGNOSTICS = 20;
const MAX_LINKS = 50;
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));

async function readStdin() {
  let data = '';
  for await (const chunk of process.stdin) data += chunk;
  return JSON.parse(data || '{}');
}

function playwrightModuleUrl() {
  const configured = process.env.LLAMA_PLAYWRIGHT_MODULE;
  const target = configured
    ? path.resolve(configured)
    : path.resolve(SCRIPT_DIR, '../ui/node_modules/playwright/index.mjs');
  return pathToFileURL(target).href;
}

async function checkedUrl(value) {
  const target = String(value || '').trim();
  let url;

  try {
    url = new URL(target);
  } catch {
    url = new URL(pathToFileURL(path.resolve(target)).href);
  }

  if (!['http:', 'https:', 'data:', 'file:'].includes(url.protocol)) {
    throw new Error(`unsupported URL protocol: ${url.protocol}`);
  }

  if (url.protocol === 'file:') {
    const localPath = fileURLToPath(url);
    let stat;

    try {
      stat = await fs.stat(localPath);
    } catch {
      throw new Error(`local browser target does not exist: ${localPath}`);
    }

    if (!stat.isFile()) throw new Error(`local browser target is not a file: ${localPath}`);
  }

  return url.href;
}

function positiveInt(value, fallback, maximum) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(1, Math.min(parsed, maximum)) : fallback;
}

async function performAction(page, action) {
  const type = String(action?.type || '');
  const timeout = positiveInt(action?.timeout, 10_000, 60_000);

  switch (type) {
    case 'click':
      await page.locator(String(action.selector || '')).first().click({ timeout });
      break;
    case 'fill':
      await page.locator(String(action.selector || '')).first().fill(String(action.text ?? ''), { timeout });
      break;
    case 'type':
      await page.locator(String(action.selector || '')).first().pressSequentially(String(action.text ?? ''), {
        delay: positiveInt(action.delay, 25, 1000), timeout,
      });
      break;
    case 'press':
      if (action.selector) {
        await page.locator(String(action.selector)).first().press(String(action.key || 'Enter'), { timeout });
      } else {
        await page.keyboard.press(String(action.key || 'Enter'));
      }
      break;
    case 'wait_for_selector':
      await page.locator(String(action.selector || '')).first().waitFor({
        state: action.state || 'visible', timeout,
      });
      break;
    case 'wait_for_text':
      await page.getByText(String(action.text ?? ''), { exact: Boolean(action.exact) }).first().waitFor({ timeout });
      break;
    case 'scroll':
      await page.mouse.wheel(Number(action.x || 0), Number(action.y || 700));
      break;
    default:
      throw new Error(`unsupported browser action: ${type || '<empty>'}`);
  }

  return { type, url: page.url() };
}

async function main() {
  const input = await readStdin();
  const url = await checkedUrl(input.url);
  const timeout = positiveInt(input.timeout, 30_000, 120_000);
  const maxTextChars = positiveInt(input.max_text_chars, 12_000, 64_000);
  const actions = Array.isArray(input.actions) ? input.actions.slice(0, MAX_ACTIONS) : [];
  const waitUntil = ['commit', 'domcontentloaded', 'load', 'networkidle'].includes(input.wait_until)
    ? input.wait_until
    : 'domcontentloaded';

  const { chromium } = await import(playwrightModuleUrl());
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });

  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      locale: input.locale || 'pt-BR',
    });
    const page = await context.newPage();
    const consoleMessages = [];
    const pageErrors = [];
    const failedRequests = [];

    page.on('console', (message) => {
      if (consoleMessages.length >= MAX_DIAGNOSTICS) return;
      consoleMessages.push({ type: message.type(), text: message.text().slice(0, 1000) });
    });
    page.on('pageerror', (error) => {
      if (pageErrors.length >= MAX_DIAGNOSTICS) return;
      pageErrors.push(String(error?.message || error).slice(0, 2000));
    });
    page.on('requestfailed', (request) => {
      if (failedRequests.length >= MAX_DIAGNOSTICS) return;
      failedRequests.push({
        error: String(request.failure()?.errorText || 'request failed').slice(0, 1000),
        method: request.method(),
        url: request.url().slice(0, 2000),
      });
    });
    page.setDefaultTimeout(timeout);
    page.setDefaultNavigationTimeout(timeout);
    await page.goto(url, { waitUntil, timeout });

    const actionResults = [];
    for (const action of actions) actionResults.push(await performAction(page, action));

    const result = await page.evaluate(({ maxTextChars, maxLinks }) => ({
      title: document.title,
      text: (document.body?.innerText || document.body?.textContent || '').trim().slice(0, maxTextChars),
      links: Array.from(document.querySelectorAll('a[href]')).slice(0, maxLinks).map((link) => ({
        text: (link.innerText || link.textContent || '').trim().slice(0, 200),
        url: link.href,
      })).filter((link) => link.url),
    }), { maxTextChars, maxLinks: MAX_LINKS });

    let screenshotPath = null;
    if (input.screenshot_path) {
      screenshotPath = path.resolve(String(input.screenshot_path));
      await fs.mkdir(path.dirname(screenshotPath), { recursive: true });
      await page.screenshot({ path: screenshotPath, fullPage: Boolean(input.full_page) });
    }

    process.stdout.write(JSON.stringify({
      ok: true,
      engine: 'playwright',
      browser_version: await browser.version(),
      url: page.url(),
      title: result.title,
      text: result.text,
      links: result.links,
      console: consoleMessages,
      page_errors: pageErrors,
      failed_requests: failedRequests,
      actions: actionResults,
      screenshot_path: screenshotPath,
    }));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stdout.write(JSON.stringify({ ok: false, error: error?.message || String(error) }));
  process.exitCode = 1;
});
