/** Isolated browser smoke test. All API calls use fixtures; no real account or data. */
import assert from 'node:assert/strict';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.NAV_TEST_PLAYWRIGHT || 'playwright');
const baseURL = process.env.NAV_TEST_URL || 'http://127.0.0.1:5000';
if (!['127.0.0.1', 'localhost'].includes(new URL(baseURL).hostname)) throw new Error('Use a local test server only');
const output = await mkdtemp(join(tmpdir(), 'navigation-ui-'));
const browser = await chromium.launch({ headless: true, ...(process.env.NAV_TEST_BROWSER ? { executablePath: process.env.NAV_TEST_BROWSER } : {}) });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
let role = 'admin';
let statsUnavailable = false;
const errors = [];
try {
  await context.addCookies([{ name: 'access_token', value: 'local-ui-fixture-only', url: baseURL }]);
  await context.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const fixtures = {
      '/api/auth/me': { user: { id: 1, username: 'UI Preview', role } },
      '/api/config': { config: { language: 'zh-CN' } },
      '/api/stats': { todayReplies: 128, avgResponseTime: 1.4, activeSessions: 6, modelLoad: 24, cpuUsage: 18, memoryUsage: { used: 8, total: 32 } },
      '/api/loras': { loras: [] },
      '/api/stats/services': { services: [{ name: '对话服务', status: 'running' }] },
      '/api/stats/activity': { activity: [] },
      '/api/characters': { characters: [] },
    };
    await route.fulfill({ status: path === '/api/stats' && statsUnavailable ? 503 : 200, contentType: 'application/json', body: JSON.stringify(fixtures[path] || {}) });
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(baseURL);
  await page.getByRole('heading', { name: '工作台', exact: true }).waitFor();
  const nav = page.getByRole('navigation', { name: '主导航' });
  assert.equal(await nav.locator('a:visible').count(), 4);
  assert.equal(await nav.getByRole('button', { name: '模型与训练' }).getAttribute('aria-expanded'), 'false');
  assert.equal(await page.getByRole('button', { name: '运行详情与趋势' }).getAttribute('aria-expanded'), 'false');
  await page.screenshot({ path: join(output, 'desktop.png'), fullPage: true, animations: 'disabled' });
  await nav.getByRole('button', { name: '模型与训练' }).click();
  assert.equal(await nav.locator('a:visible').count(), 8);
  await page.getByRole('textbox', { name: '查找功能…' }).fill('RAG');
  assert.equal(await nav.locator('a:visible').count(), 1);
  assert.equal(await nav.locator('a:visible').getAttribute('href'), '/knowledge');
  await page.getByRole('button', { name: '清空搜索' }).click();
  await page.goto(`${baseURL}/lora`);
  await nav.locator('a[aria-current="page"]').waitFor();
  assert.equal(await nav.locator('a[aria-current="page"]').getAttribute('href'), '/lora');
  assert.equal(await nav.getByRole('button', { name: '模型与训练' }).getAttribute('aria-expanded'), 'true');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(baseURL);
  await page.getByRole('heading', { name: '工作台', exact: true }).waitFor();
  await page.getByRole('button', { name: '打开导航' }).click();
  const drawer = page.getByRole('dialog');
  await drawer.waitFor();
  assert.equal(await drawer.getByRole('navigation').locator('a:visible').count(), 4);
  await page.screenshot({ path: join(output, 'mobile-navigation.png'), fullPage: true, animations: 'disabled' });
  await page.keyboard.press('Escape');
  await drawer.waitFor({ state: 'hidden' });
  await page.screenshot({ path: join(output, 'mobile.png'), fullPage: true, animations: 'disabled' });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.getByRole('button', { name: '打开导航' }).click();
  await page.getByRole('dialog').getByRole('link', { name: '角色与记忆' }).click();
  await page.waitForURL('**/characters');
  await page.getByRole('dialog').waitFor({ state: 'hidden' });

  role = 'user';
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(baseURL);
  await page.getByRole('heading', { name: '无权访问' }).waitFor();
  await nav.getByRole('button', { name: '评测与实验' }).click();
  assert.equal(await nav.locator('a:visible').count(), 1);
  assert.equal(await nav.locator('a:visible').getAttribute('href'), '/narrative');
  await page.getByRole('textbox', { name: '查找功能…' }).fill('训练');
  assert.equal(await nav.locator('a:visible').count(), 0);

  role = 'admin'; statsUnavailable = true;
  await page.goto(baseURL);
  await page.getByText(/运行数据暂不可用，不影响打开其他功能/).waitFor();
  assert.ok(await page.getByRole('link', { name: '维护知识库' }).isVisible());
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, checks: ['desktop groups', 'search', 'active route', 'mobile drawer', 'Escape', 'navigation closes drawer', 'no mobile overflow', 'role filtering', 'API failure keeps actions'], screenshots: output }, null, 2));
} finally {
  await browser.close();
}
