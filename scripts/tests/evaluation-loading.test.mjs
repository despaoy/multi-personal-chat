import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../../src/hooks/useEvaluation.ts', import.meta.url), 'utf8');
const tree = ts.createSourceFile('hook.ts', source, ts.ScriptTarget.Latest, true);
let fetchDeclaration;
function visit(node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === 'fetchAll') fetchDeclaration = node;
  ts.forEachChild(node, visit);
}
visit(tree);
assert.ok(fetchDeclaration);
const declaration = ts.createPrinter().printNode(ts.EmitHint.Unspecified, fetchDeclaration, tree);
const { outputText } = ts.transpileModule(`export function harness(api, state, enabled = true) {
  const useCallback = callback => callback;
  const setLoading = value => state.loading = value;
  const setError = value => state.error = value;
  const setGoldSet = value => state.gold = value;
  const setRuns = value => state.runs = value;
  const setRunsTotal = value => state.runsTotal = value;
  const setFeedbacks = value => state.feedbacks = value;
  const setFeedbackTotal = value => state.feedbackTotal = value;
  const ${declaration};
  return fetchAll;
}`, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { harness } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const api = {
  getGoldSet: async () => ({ total: 1, prompts: ['sample'], category_breakdown: { sample: 1 } }),
  getEvaluationRuns: async () => ({ runs: ['run'], total: 7 }),
  getFeedback: async () => ({ feedbacks: ['feedback'], total: 9 }),
};

test('evaluation loads authoritative totals and records together', async () => {
  const state = {};
  await harness(api, state)();
  assert.equal(state.error, null);
  assert.equal(state.loading, false);
  assert.deepEqual(state.feedbacks, ['feedback']);
  assert.equal(state.feedbackTotal, 9);
  assert.equal(state.runsTotal, 7);
});

test('failed feedback is visible and never replaces existing records with fake empty success', async (t) => {
  t.mock.method(console, 'error', () => {});
  const state = { feedbacks: ['previous'], feedbackTotal: 3 };
  await harness({ ...api, getFeedback: async () => { throw new Error('Feedback unavailable'); } }, state)();
  assert.equal(state.error, 'Feedback unavailable');
  assert.equal(state.loading, false);
  assert.deepEqual(state.feedbacks, ['previous']);
  assert.equal(state.feedbackTotal, 3);
});
