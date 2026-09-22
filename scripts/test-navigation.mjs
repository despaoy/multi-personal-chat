import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { test } from 'node:test';
import { navigationItems, navigationGroups, visibleNavigation, isNavigationActive } from '../src/lib/navigation.ts';

test('Every existing page remains reachable without duplicated links', () => {
  const paths = navigationItems.map(item => item.href);
  assert.equal(paths.length, new Set(paths).size);
  for (const item of navigationItems) {
    assert.ok(existsSync(new URL(`../src/app${item.href === '/' ? '' : item.href}/page.tsx`, import.meta.url)), item.href);
    assert.ok(navigationGroups.some(group => group.id === item.group));
  }
  for (const dir of readdirSync(new URL('../src/app', import.meta.url), { withFileTypes: true })) {
    if (dir.isDirectory() && dir.name !== 'login' && existsSync(new URL(`../src/app/${dir.name}/page.tsx`, import.meta.url))) {
      assert.ok(paths.includes(`/${dir.name}`), dir.name);
    }
  }
});

test('Everyday navigation is limited to four entries', () => {
  assert.deepEqual(visibleNavigation(true).filter(item => item.group === 'workspace').map(item => item.href), ['/', '/characters', '/history', '/knowledge']);
});

test('Search never reveals administrator-only entries to ordinary users', () => {
  assert.deepEqual(visibleNavigation(false).map(item => item.href), ['/narrative']);
  assert.deepEqual(visibleNavigation(false, 'training'), []);
  assert.deepEqual(visibleNavigation(false, '分支').map(item => item.href), ['/narrative']);
});

test('Search supports synonyms, routes, multiple words and translated labels', () => {
  assert.equal(visibleNavigation(true, 'RAG')[0].href, '/knowledge');
  assert.equal(visibleNavigation(true, '关系 记忆')[0].href, '/characters');
  assert.equal(visibleNavigation(true, '/intent-training')[0].href, '/intent-training');
  assert.equal(visibleNavigation(true, 'personalities', (key, fallback) => key === 'navigation.characters' ? 'Personalities' : fallback)[0].href, '/characters');
  assert.deepEqual(visibleNavigation(true, 'not-a-real-feature'), []);
});

test('Nested routes highlight their owner, not similarly named pages or home', () => {
  assert.ok(isNavigationActive('/training/jobs/12', '/training'));
  assert.ok(isNavigationActive('/', '/'));
  assert.ok(!isNavigationActive('/training', '/'));
  assert.ok(!isNavigationActive('/training-extra', '/training'));
});
