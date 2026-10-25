import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

// Execute the actual dependency-free API client with controlled HTTP responses.
const source = await readFile(new URL('../../src/lib/api.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { api, ApiRequestError } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const request = { character_description: '完整合成角色：仅办理预约查询', num_dialogues: 2 };

test('failed generation preserves structured partial results without adding content to the error message', async (t) => {
  const partial = {
    dialogues: [{ conversations: [{ from: 'human', value: 'PRIVATE_SYNTHETIC_CONTENT' }, { from: 'gpt', value: '完整合成回复' }] }],
    total: 1,
    cost_time: 0.25,
  };
  const detail = { error: 'dialogue_batch_invalid', message: '第二批格式错误', partial_result: partial };
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    calls++;
    return Response.json({ detail }, { status: 502 });
  });
  await assert.rejects(api.generateDialogues(request), (error) => {
    assert.ok(error instanceof ApiRequestError);
    assert.equal(error.status, 502);
    assert.deepEqual(error.detail.partial_result, partial);
    assert.match(error.message, /第二批格式错误/);
    assert.doesNotMatch(error.message, /PRIVATE_SYNTHETIC_CONTENT|partial_result/);
    return true;
  });
  assert.equal(calls, 1);
});

test('ordinary string and validation errors retain their status and details', async (t) => {
  for (const [status, detail] of [[500, '模型服务调用失败'], [422, [{ loc: ['body', 'num_dialogues'], msg: '必须大于零' }]]]) {
    t.mock.method(globalThis, 'fetch', async () => Response.json({ detail }, { status }));
    await assert.rejects(api.generateDialogues(request), (error) => {
      assert.ok(error instanceof ApiRequestError);
      assert.equal(error.status, status);
      assert.deepEqual(error.detail, detail);
      assert.match(error.message, new RegExp(String(status)));
      return true;
    });
    t.mock.restoreAll();
  }
});

test('non-JSON failure remains an error without exposing an upstream HTML body', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('<html>PRIVATE_UPSTREAM_BODY</html>', {
    status: 502, statusText: 'Bad Gateway',
  }));
  await assert.rejects(api.generateDialogues(request), (error) => {
    assert.equal(error.status, 502);
    assert.equal(error.detail, 'Bad Gateway');
    assert.doesNotMatch(error.message, /PRIVATE_UPSTREAM_BODY/);
    return true;
  });
});

test('successful generation still returns the original result', async (t) => {
  const result = { success: true, dialogues: [], total: 0, cost_time: 0.25 };
  t.mock.method(globalThis, 'fetch', async () => Response.json(result));
  assert.deepEqual(await api.generateDialogues(request), result);
});

test('auth requests share the client but keep failed credentials in the current form', async (t) => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options });
    return Response.json({ detail: 'Invalid credentials' }, { status: 401 });
  });
  // Auth errors must not start a logout or redirect, even outside /login.
  globalThis.window = { location: { pathname: '/', href: '/' } };
  t.after(() => { delete globalThis.window; });
  for (const request of [() => api.login({ username: 'sample', password: 'synthetic' }),
    () => api.register({ username: 'sample', password: 'synthetic' }), () => api.getCurrentUser()]) {
    await assert.rejects(request, error => error.status === 401);
  }
  assert.deepEqual(calls.map(call => call.url), ['/api/auth/login', '/api/auth/register', '/api/auth/me']);
  assert.equal(window.location.href, '/');
  assert.ok(calls.every(call => call.options.credentials === 'include'));
  assert.equal(calls[0].options.headers.get('Content-Type'), 'application/json');
});

test('request headers honor case-insensitive overrides and FormData boundaries', async (t) => {
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    seen.push(options.headers);
    return Response.json({ success: true });
  });
  await api.request('/synthetic', { body: '{}', headers: { 'content-type': 'application/custom' } });
  await api.request('/synthetic', { body: new FormData() });
  assert.equal(seen[0].get('Content-Type'), 'application/custom');
  assert.equal(seen[1].has('Content-Type'), false);
});
