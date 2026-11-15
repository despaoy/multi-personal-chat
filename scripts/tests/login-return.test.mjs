import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../../src/lib/login-return.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { loginReturnPath } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

test('login preserves character and sharing destinations', () => {
  assert.equal(loginReturnPath('/chat?character=sample#turn-1'), '/chat?character=sample#turn-1');
  assert.equal(loginReturnPath('/character-library?share=sample'), '/character-library?share=sample');
});

test('login does not accept external, executable or looping destinations', () => {
  for (const value of [null, '', 'https://example.com', '//example.com', '/\\example.com', 'javascript:alert(1)', '/login?redirect=/login', '/api/auth/logout', '/\n/example.com']) {
    assert.equal(loginReturnPath(value), '/', String(value));
  }
});
