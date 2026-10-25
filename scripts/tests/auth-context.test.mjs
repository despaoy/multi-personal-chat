import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

// Execute the provider's real callbacks/effect without a DOM or persistent storage.
const source = (await readFile(new URL('../../src/contexts/AuthContext.tsx', import.meta.url), 'utf8'))
  .replace(/^import .*;$/gm, '').replaceAll('export ', '');
const { outputText } = ts.transpileModule(`export function harness(api) {
  const state = [];
  let effect;
  const useState = initial => {
    const index = state.length;
    state.push(initial);
    return [initial, value => state[index] = value];
  };
  const useEffect = callback => effect = callback;
  const useCallback = callback => callback;
  const createContext = () => ({ Provider: 'provider' });
  const React = { createElement: (_, props) => props };
  ${source}
  const { value } = AuthProvider({ children: null });
  return { state, value, mount: () => effect() };
}`, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React } });
const { harness } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const flush = () => new Promise(resolve => setImmediate(resolve));
const user = { id: 3, username: 'synthetic', created_at: '', role: 'admin' };

test('server validation hydrates authentication and failure stays unauthenticated', async () => {
  for (const success of [true, false]) {
    const provider = harness({ getCurrentUser: async () => {
      if (!success) throw new Error('offline');
      return { user };
    } });
    assert.deepEqual(provider.state, [null, true]);
    provider.mount();
    await flush();
    assert.deepEqual(provider.state, [success ? user : null, false]);
  }
});

test('unmounted authentication requests cannot update state', async () => {
  let resolve;
  const provider = harness({ getCurrentUser: () => new Promise(done => resolve = done) });
  provider.mount()();
  resolve({ user });
  await flush();
  assert.deepEqual(provider.state, [null, true]);
});

for (const action of ['login', 'register']) {
  test(`${action} independently authenticates through the API without localStorage`, async () => {
    const calls = [];
    const provider = harness({ [action]: async credentials => {
      calls.push(credentials);
      return { user };
    } });
    await provider.value[action]('synthetic', 'not-a-real-password');
    assert.equal(provider.state[0], user);
    assert.deepEqual(calls, [{ username: 'synthetic', password: 'not-a-real-password' }]);
  });

  test(`${action} rejects failed authentication without changing the current user`, async () => {
    const failure = new Error('Authentication rejected');
    const provider = harness({ [action]: async () => { throw failure; } });
    await assert.rejects(provider.value[action]('synthetic', 'not-a-real-password'), failure);
    assert.equal(provider.state[0], null);
  });
}
