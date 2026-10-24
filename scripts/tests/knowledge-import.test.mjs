import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

// Execute the real client, hook methods and page handlers with controlled HTTP/UI bindings.
const apiSource = await readFile(new URL('../../src/lib/api.ts', import.meta.url), 'utf8');
const hookSource = await readFile(new URL('../../src/hooks/useKnowledge.ts', import.meta.url), 'utf8');
const pageSource = await readFile(new URL('../../src/app/knowledge/page.tsx', import.meta.url), 'utf8');
const compilerOptions = { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 };
const load = async (source) => import(`data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, { compilerOptions }).outputText).toString('base64')}`);
const { api } = await load(apiSource);
const printer = ts.createPrinter();
function declaration(source, name) {
  const tree = ts.createSourceFile('fixture.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const found = [];
  function visit(node) {
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text === name) {
      found.push('const ' + printer.printNode(ts.EmitHint.Unspecified, node, tree) + ';');
    }
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) {
      found.push(printer.printNode(ts.EmitHint.Unspecified, node, tree));
    }
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.equal(found.length, 1, `Expected actual declaration ${name}`);
  return found[0];
}
const { harness } = await load(`export function harness(bindings) {
  const { api, toast, fetchFolders, fetchDocuments, fetchBases, setError,
    setUploading, setImporting, activeBaseId, fileInputRef, textEncoding } = bindings;
  ${declaration(hookSource, 'uploadZip')}
  ${declaration(hookSource, 'importDir')}
  ${declaration(pageSource, 'notifyKnowledgeImport')}
  ${declaration(pageSource, 'handleZipUpload')}
  ${declaration(pageSource, 'handleImportDir')}
  return { handleZipUpload, handleImportDir };
}`);

function fixture() {
  const notices = [], refreshes = [], errors = [], busy = [];
  const toast = Object.fromEntries(['success', 'info', 'warning', 'error'].map(kind => [kind, message => notices.push({ kind, message })]));
  const fileInputRef = { current: { value: 'selected' } };
  const handlers = harness({ api, toast, activeBaseId: 17, fileInputRef, textEncoding: 'unicode',
    fetchFolders: async () => refreshes.push('folders'),
    fetchDocuments: async () => refreshes.push('documents'),
    fetchBases: async () => refreshes.push('bases'),
    setError: value => errors.push(value), setUploading: value => busy.push(value), setImporting: value => busy.push(value) });
  return { handlers, notices, refreshes, errors, busy, fileInputRef };
}
async function run(rig, route) {
  if (route === 'zip') {
    // A valid empty ZIP is sufficient for this transport/UI unit; Python tests parse real document ZIPs.
    const file = new File([Buffer.from('UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA==', 'base64')], 'knowledge.zip', { type: 'application/zip' });
    await rig.handlers.handleZipUpload({ target: { files: [file] } });
  } else {
    await rig.handlers.handleImportDir('complete-synthetic-source');
  }
}
const outcomes = [
  { status: 'complete', success: true, createdDocs: 2, errors: [], kind: 'success' },
  { status: 'complete', success: true, createdDocs: 0, errors: [], kind: 'info' },
  { status: 'partial', success: false, createdDocs: 1, errors: ['02-broken.txt: 保存未完成'], kind: 'warning' },
  { status: 'failed', success: false, createdDocs: 0, errors: ['01-broken.txt: 保存未完成'], kind: 'error' },
  { status: 'unconfirmed', success: false, createdDocs: 0, errors: ['操作超时，请刷新核对'], kind: 'warning' },
];
for (const route of ['zip', 'directory']) {
  for (const outcome of outcomes) {
    test(`${route} displays ${outcome.status}/${outcome.createdDocs} and refreshes confirmed state`, async t => {
      const { kind, ...fields } = outcome;
      const result = { ...fields, createdFolders: ['适用条件'], message: `完整合成结果：${outcome.status}，已确认 ${outcome.createdDocs} 个文档` };
      let calls = 0;
      t.mock.method(globalThis, 'fetch', async (url, options) => {
        calls++;
        const requestUrl = new URL(url, 'http://localhost');
        assert.match(requestUrl.pathname, route === 'zip' ? /\/knowledge\/bases\/17\/upload-zip$/ : /\/knowledge\/scan\/import$/);
        assert.equal(requestUrl.searchParams.get('text_encoding'), 'unicode');
        assert.equal(options.credentials, 'include');
        return Response.json(result);
      });
      const rig = fixture();
      await run(rig, route);
      assert.equal(calls, 1);
      assert.deepEqual(rig.notices[0], { kind, message: result.message });
      assert.deepEqual(rig.notices.slice(1), result.errors.map(message => ({ kind: 'warning', message })));
      assert.deepEqual(rig.refreshes, route === 'zip' ? ['folders', 'documents'] : ['bases', 'folders', 'documents']);
      assert.deepEqual(rig.errors, []);
      assert.deepEqual(rig.busy, route === 'zip' ? [true, false] : ['complete-synthetic-source', null]);
      if (route === 'zip') assert.equal(rig.fileInputRef.current.value, '');
      if (!result.success) assert.equal(rig.notices.some(notice => notice.kind === 'success'), false);
    });
  }
  test(`${route} reports an HTTP failure once and releases the busy state`, async t => {
    t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: '导入服务暂时不可用' }, { status: 503 }));
    const rig = fixture();
    await run(rig, route);
    assert.equal(rig.notices.length, 1);
    assert.equal(rig.notices[0].kind, 'error');
    assert.match(rig.notices[0].message, /503/);
    assert.equal(rig.errors.length, 1);
    assert.deepEqual(rig.refreshes, []);
    assert.deepEqual(rig.busy, route === 'zip' ? [true, false] : ['complete-synthetic-source', null]);
  });
}
