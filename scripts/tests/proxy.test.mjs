import assert from 'node:assert/strict';
import { after, test } from 'node:test';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';
import ts from 'typescript';

const observations = new Map();
const server = createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  const key = url.searchParams.get('id');
  const observation = { closed: false };
  if (key) observations.set(key, observation);
  response.on('close', () => { observation.closed = true; });
  if (url.pathname.endsWith('/empty')) {
    response.writeHead(204);
    response.end();
    return;
  }
  if (url.pathname.endsWith('/no-headers')) return;
  if (url.pathname.endsWith('/stalled-json')) {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.write('{"incomplete":');
    return;
  }
  if (url.pathname.endsWith('/stream')) {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.write('data: first\n\n');
    return;
  }
  if (url.pathname.endsWith('/network-error')) {
    response.destroy();
    return;
  }
  if (url.pathname.endsWith('/compressed')) {
    const payload = gzipSync(JSON.stringify({ value: '完整合成回复' }));
    response.writeHead(200, { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip', 'Content-Length': payload.length });
    response.end(payload);
    return;
  }
  if (url.pathname.endsWith('/auth/logout')) {
    response.writeHead(503, { 'Content-Type': 'text/plain' });
    response.end('logout unavailable');
    return;
  }
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = Buffer.concat(chunks).toString();
  if (url.pathname.endsWith('/invalid')) {
    try { JSON.parse(body); } catch {
      response.writeHead(422, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ detail: 'invalid JSON', received: body }));
      return;
    }
  }
  const failure = url.pathname.endsWith('/failure');
  response.writeHead(failure ? 503 : 200, {
    'Content-Type': 'application/json', 'X-Upstream': 'preserved',
    'Set-Cookie': ['access_token=synthetic; HttpOnly; Path=/', 'synthetic_second=1; Path=/'],
  });
  response.end(JSON.stringify(failure ? { detail: { kind: 'dependency_error', message: '完整合成错误' } } : {
    method: request.method, body, authorization: request.headers.authorization,
    contentType: request.headers['content-type'],
  }));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
process.env.BACKEND_URL = `http://127.0.0.1:${server.address().port}`;
after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });

async function moduleUrl(path, imports = {}) {
  let source = await readFile(path, 'utf8');
  for (const [from, to] of Object.entries(imports)) source = source.replaceAll(`from '${from}'`, `from '${to}'`);
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  });
  return `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`;
}
const proxyUrl = await moduleUrl('src/lib/proxy.ts');
const { proxyRequest } = await import(proxyUrl);
const routes = await import(await moduleUrl('src/app/api/[[...path]]/route.ts', { '@/lib/proxy': proxyUrl }));
const require = createRequire(pathToFileURL(`${process.cwd()}/package.json`));
const logout = await import(await moduleUrl('src/app/api/auth/logout/route.ts', {
  '@/lib/proxy': proxyUrl, 'next/server': pathToFileURL(require.resolve('next/server')).href,
}));
const localRequest = (path, options = {}) => new Request(`http://localhost${path}`, options);
const context = path => ({ params: Promise.resolve({ path: path.split('/').filter(Boolean) }) });

async function bounded(promise, milliseconds = 1000) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('observation deadline exceeded')), milliseconds);
    })]);
  } finally { clearTimeout(timer); }
}
async function waitFor(predicate) {
  const deadline = Date.now() + 1000;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error('observation deadline exceeded');
    await new Promise(resolve => setTimeout(resolve, 5));
  }
}

test('catch-all forwards the method, exact JSON bytes, identity and response headers without a content-length hint', async () => {
  const body = '{ "message": "完整事实：研究图像分割，不研究语音。", "count": 2 }';
  const response = await routes.POST(localRequest('/api/generate/echo', {
    method: 'POST', body, headers: { 'Content-Type': 'application/json', Cookie: 'access_token=synthetic-owner', Authorization: 'Bearer ignored' },
  }), context('generate/echo'));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { method: 'POST', body, authorization: 'Bearer synthetic-owner', contentType: 'application/json' });
  assert.equal(response.headers.get('x-upstream'), 'preserved');
  assert.equal(response.headers.getSetCookie().length, 2);
});

test('malformed deletion input reaches backend validation unchanged instead of becoming an empty operation', async () => {
  const response = await routes.DELETE(localRequest('/api/messages/invalid', {
    method: 'DELETE', body: '{broken', headers: { 'Content-Type': 'application/json', 'Content-Length': '7' },
  }), context('messages/invalid'));
  assert.equal(response.status, 422);
  assert.deepEqual(await response.json(), { detail: 'invalid JSON', received: '{broken' });
});

test('multipart uploads retain their boundary and original content', async () => {
  const form = new FormData();
  form.set('file', new File(['完整合成文档：开放时间为周二至周日九点。'], 'facts.txt', { type: 'text/plain' }));
  const request = localRequest('/api/knowledge/upload', { method: 'POST', body: form });
  const expected = await request.clone().text();
  const response = await routes.POST(request, context('knowledge/upload'));
  const received = await response.json();
  assert.equal(received.method, 'POST');
  assert.equal(received.body, expected);
  assert.equal(received.contentType, request.headers.get('Content-Type'));
});

test('client cancellation before headers cancels the upstream wait and retains the cancellation reason', async () => {
  const controller = new AbortController();
  const request = localRequest('/api/generate/no-headers?id=client', { signal: controller.signal });
  const pending = proxyRequest(request, '/api/generate/no-headers?id=client', { timeout: 2000 });
  const outcome = pending.then(value => ({ value }), error => ({ error }));
  await waitFor(() => observations.has('client'));
  const reason = new DOMException('synthetic cancellation', 'AbortError');
  controller.abort(reason);
  try {
    const result = await bounded(outcome, 300);
    assert.equal(result.error, reason);
    await waitFor(() => observations.get('client').closed);
  } finally {
    if (!observations.get('client').closed) server.closeAllConnections();
    await outcome;
  }
});

test('a deadline before headers returns 504 and closes the upstream connection', async () => {
  const response = await proxyRequest(localRequest('/api/generate/no-headers?id=timeout'), '/api/generate/no-headers?id=timeout', { timeout: 100 });
  assert.equal(response.status, 504);
  assert.deepEqual(await response.json(), { detail: '请求超时' });
  await waitFor(() => observations.get('timeout')?.closed);
});

test('the same deadline covers a JSON body stalled after successful headers', async () => {
  const response = await proxyRequest(localRequest('/api/generate/stalled-json?id=json'), '/api/generate/stalled-json?id=json', { timeout: 100 });
  try {
    assert.equal(response.status, 504);
    assert.deepEqual(await response.json(), { detail: '请求超时' });
    await waitFor(() => observations.get('json')?.closed);
  } finally { if (!response.bodyUsed) await response.body.cancel(); }
});

test('a streaming deadline interrupts the body instead of reporting a completed stream', async () => {
  const response = await proxyRequest(localRequest('/api/generate/stream?id=stream-timeout'), '/api/generate/stream?id=stream-timeout', { timeout: 150 });
  assert.equal(response.headers.get('content-type'), 'text/event-stream');
  const reader = response.body.getReader();
  try {
    assert.match(new TextDecoder().decode((await reader.read()).value), /data: first/);
    await assert.rejects(bounded(reader.read(), 500), error => error.name === 'TimeoutError');
    await waitFor(() => observations.get('stream-timeout').closed);
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
});

test('cancelling a response reader releases the upstream stream', async () => {
  const response = await proxyRequest(localRequest('/api/generate/stream?id=reader'), '/api/generate/stream?id=reader', { timeout: 2000 });
  const reader = response.body.getReader();
  await reader.read();
  await reader.cancel('synthetic consumer stopped');
  reader.releaseLock();
  await waitFor(() => observations.get('reader').closed);
});

test('backend failure details and status are forwarded without invented progress or a replacement result', async () => {
  const response = await routes.GET(localRequest('/api/generate/failure'), context('generate/failure'));
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), { detail: { kind: 'dependency_error', message: '完整合成错误' } });
  assert.equal(response.headers.get('x-upstream'), 'preserved');
});

test('decoded compressed bodies do not retain stale encoding or length headers', async () => {
  const response = await proxyRequest(localRequest('/api/generate/compressed'), '/api/generate/compressed');
  assert.deepEqual(await response.json(), { value: '完整合成回复' });
  assert.equal(response.headers.get('content-encoding'), null);
  assert.equal(response.headers.get('content-length'), null);
});

test('a network failure is an explicit 502', async () => {
  const response = await proxyRequest(localRequest('/api/generate/network-error'), '/api/generate/network-error');
  assert.equal(response.status, 502);
  assert.deepEqual(await response.json(), { detail: '后端服务不可用' });
});

test('a valid response without a body remains 204', async () => {
  const response = await proxyRequest(localRequest('/api/generate/empty'), '/api/generate/empty');
  assert.equal(response.status, 204);
  assert.equal(response.body, null);
});

test('logout retains an upstream failure body and clears the local cookie', async () => {
  const response = await logout.POST(localRequest('/api/auth/logout', { method: 'POST' }));
  assert.equal(response.status, 503);
  assert.equal(await response.text(), 'logout unavailable');
  assert.match(response.headers.get('set-cookie'), /access_token=;/);
  assert.match(response.headers.get('set-cookie'), /Max-Age=0/i);
});

test('same-origin and permitted-path checks still reject before forwarding', async () => {
  for (const [path, options] of [['/api/not-allowed', {}], ['/api/generate', { method: 'POST', headers: { Origin: 'https://unrelated.invalid' } }]]) {
    const response = await proxyRequest(localRequest(path, options), path);
    assert.equal(response.status, 403);
    await response.body.cancel();
  }
});
