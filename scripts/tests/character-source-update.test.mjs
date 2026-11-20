import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import test from 'node:test';
import ts from 'typescript';

// Real React hooks/components, controlled HTTP. Optional isolated renderer installation
// keeps production dependencies unchanged (see CHARACTER_SOURCE_UPDATE_UI.md).
const testRequire = createRequire(process.env.CHARACTER_UI_TEST_MODULES || import.meta.url);
const React = testRequire('react');
const { create, act } = testRequire('react-test-renderer');
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const root = resolve(import.meta.dirname, '../..');
const doc = (id, content = id) => ({ id, title: id, content });
const settings = { name: '旅人', persona: '人物', greeting: '你好', scenario: '', relationships: '', examples: [], reference_text: '' };
const character = { id: 'copy', revision: 1, settings_revision: 1, settings, source_character_id: 'author', source_revision: 1 };
const basePlan = () => ({ available: true, revision: 1, source_revision: 2, publication_id: 'pub2', local_corpus_id: 'local1',
  baseline_publication_id: 'pub1', changes: {}, current_sources: [doc('a', '本地'), doc('b', '本地b')],
  incoming_sources: [doc('a', '作者')], sources: [doc('a', '作者')], source_merge: { status: 'conflict', reason: 'explicit_resolution_required',
    actions: [], conflicts: [{ id: 'a', kind: 'both_modified' }, { id: 'b', kind: 'local_modify_author_delete' }], merged_sources: [] } });
const text = value => typeof value === 'string' ? value : Array.isArray(value) ? value.map(text).join('') : value?.children ? text(value.children) : '';
const buttons = renderer => renderer.root.findAllByType('button');
const button = (renderer, label) => {
  const found = buttons(renderer).find(b => text(b).includes(label));
  assert.ok(found, `button missing: ${label}`); return found;
};
const click = async (renderer, label) => { const b = button(renderer, label); assert.ok(!b.props.disabled, `${label} disabled`); await act(async () => b.props.onClick?.()); };
const change = async (node, value) => act(async () => node.props.onChange({ target: { value } }));
function loader(api) {
  const cache = new Map();
  function load(file) {
    const path = resolve(root, file);
    if (cache.has(path)) return cache.get(path);
    const source = readFileSync(path, 'utf8');
    const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
    const loadedModule = { exports: {} }; cache.set(path, loadedModule.exports);
    const req = id => {
      if (id === 'react' || id === 'react/jsx-runtime') return testRequire(id);
      if (id === '@/lib/api' && api) return { api, ApiRequestError: ErrorClass };
      if (id.startsWith('@/components/ui/')) {
        const tag = id.endsWith('button') ? 'button' : id.endsWith('input') ? 'input' : 'label';
        const name = tag[0].toUpperCase() + tag.slice(1);
        return { [name]: props => { const hostProps = { ...props }; delete hostProps.asChild; return React.createElement(tag, hostProps); } };
      }
      if (id === 'next/link') return { __esModule: true, default: props => React.createElement('a', props) };
      if (id.includes('/layout/')) return { AppLayout: ({ children }) => children, AuthGuard: ({ children }) => children };
      const location = id.startsWith('@/') ? resolve(root, 'src', id.slice(2)) : resolve(dirname(path), id);
      for (const suffix of ['.tsx', '.ts']) {
        try { return load(location + suffix); } catch (error) { if (error.code !== 'ENOENT') throw error; }
      }
      throw new Error(`Unexpected require ${id}`);
    };
    new Function('require', 'module', 'exports', compiled)(req, loadedModule, loadedModule.exports);
    cache.set(path, loadedModule.exports); return loadedModule.exports;
  }
  return load;
}
class ErrorClass extends Error { constructor(status) { super(`HTTP ${status}`); this.status = status; } }
function harness(overrides = {}) {
  const calls = { get: 0, preview: [], confirm: [], save: 0, publish: 0 };
  const api = {
    getCharacterSourceUpdate: async () => { calls.get++; return structuredClone(basePlan()); },
    previewCharacterSourceUpdate: async (_id, request) => {
      calls.preview.push(structuredClone(request));
      return { ...request, preview_id: 'exact', baseline_publication_id: 'pub1', settings, sources: [doc('a', '本地')],
        source_merge: { status: 'ready', actions: [], conflicts: [], resolved_conflicts: [{ id: 'a', choice: 'local' }, { id: 'b', choice: 'author' }] } };
    },
    confirmCharacterSourceUpdate: async (_id, preview) => { calls.confirm.push(preview); return { ...character, revision: 2 }; },
    listCharacterLibrary: async () => ({ characters: [character] }),
    listCharacterSources: async () => ({ corpora: [{ id: 'edition', created_at: '2026-11-19', index_status: 'completed', index_attempt: 1 }] }),
    getCharacterSources: async () => ({ sources: [doc('a')] }),
    publishCharacterSources: async () => { calls.publish++; return { id: 'new', index_status: 'completed', created_at: '2026-11-19' }; },
    saveLibraryCharacter: async () => { calls.save++; return { ...character, revision: 2 }; },
    ...overrides,
  };
  return { api, calls, load: loader(api) };
}
async function panel(h = harness()) {
  let locked = false, begins = 0, imported = 0;
  const Component = h.load('src/components/chat/CharacterSourceUpdatePanel.tsx').CharacterSourceUpdatePanel;
  let renderer;
  await act(async () => { renderer = create(React.createElement(Component, { character, disabled: false,
    begin: () => { begins++; if (locked) return false; locked = true; return true; }, end: () => { locked = false; },
    onImported: async () => { imported++; } })); });
  return { ...h, renderer, locked: () => locked, begins: () => begins, imported: () => imported };
}
async function choices(renderer) {
  const radios = renderer.root.findAllByType('input').filter(n => n.props.type === 'radio');
  assert.equal(radios.length, 4);
  await act(async () => radios[0].props.onChange());
  await act(async () => radios[3].props.onChange());
}
async function unmount(renderer) { await act(async () => renderer.unmount()); }

test('continuation binds a new source edition only after user selection', async () => {
  const originalWindow = globalThis.window;
  globalThis.window = { location: { search: '' } };
  const source = { id: 'old', character_id: character.id, title: '旧会话', revision: 3, mode: 'initial' };
  const result = {
    id: 'new-session', character, baseline: { history: [] },
    state: { relationship: { stage: 'stranger' }, memories: [] },
  };
  const submissions = [];
  const load = loader({
    listCharacterLibrary: async () => ({ characters: [character] }),
    listCharacterChats: async () => ({ sessions: [source] }),
    listSharedChatMemories: async () => ({ memories: [] }),
    getContinuationPreview: async () => ({
      source_session_id: source.id, source_revision: source.revision, character_revision: 1, previous_character_revision: 1,
      source_corpus_id: 'old-source', latest_source_corpus: { id: 'new-source', index_status: 'completed', created_at: '2026-11-19' },
      changes: {}, relationship: { stage: 'stranger' }, memories: [],
    }),
    createCharacterChat: async body => {
      submissions.push(body);
      return submissions.length === 1 ? Promise.reject(new Error('模拟刷新前的失败')) : result;
    },
    getCharacterChat: async () => ({ session: result, turns: [] }),
  });
  let renderer;
  try {
    await act(async () => { renderer = create(React.createElement(load('src/app/chat/page.tsx').default)); });
    await change(renderer.root.findByProps({ id: 'chat-source' }), source.id);
    const submit = () => act(async () => renderer.root.findByType('form').props.onSubmit({ preventDefault() {} }));
    await submit();
    assert.equal(submissions[0].source_session_id, 'old');
    assert.equal(submissions[0].source_corpus_id, undefined);
    const option = renderer.root.findAllByType('label').find(node => text(node).includes('在新会话中使用最新资料版本'));
    assert.ok(option);
    assert.equal(option.findByType('input').props.checked, false);
    await act(async () => option.findByType('input').props.onChange({ target: { checked: true } }));
    await submit();
    assert.equal(submissions[1].source_corpus_id, 'new-source');
  } finally {
    if (renderer) await unmount(renderer);
    globalThis.window = originalWindow;
  }
});

test('session deletion previews related copies and submits only explicit selections', async () => {
  const originalWindow = globalThis.window;
  globalThis.window = { location: { search: '' } };
  const root = { id: 'original', character_id: character.id, title: '原会话', revision: 3, mode: 'initial' };
  const branch = { ...root, id: 'branch', title: '关联分支', revision: 6, mode: 'branch' };
  const copy = { ...root, id: 'copy', character_id: 'other', title: '跨角色副本', revision: 2 };
  const previews = [], deletions = [];
  const load = loader({
    listCharacterLibrary: async () => ({ characters: [character] }),
    listCharacterChats: async () => ({ sessions: [root, branch] }),
    listSharedChatMemories: async () => ({ memories: [] }),
    previewCharacterChatDeletion: async selected => {
      previews.push(selected);
      return { selected: [root], related: [branch, copy] };
    },
    deleteCharacterChats: async selected => {
      deletions.push(selected);
      return { deleted_session_ids: selected.map(item => item.id) };
    },
  });
  let renderer;
  try {
    await act(async () => { renderer = create(React.createElement(load('src/app/chat/page.tsx').default)); });
    const check = async label => {
      const item = renderer.root.findAllByType('label').find(node => text(node).includes(label));
      assert.ok(item, label);
      await act(async () => item.findByType('input').props.onChange({ target: { checked: true } }));
    };
    await check('原会话');
    await click(renderer, '预览删除范围');
    assert.deepEqual(previews, [[{ id: 'original', revision: 3 }]]);
    assert.deepEqual(deletions, []);
    await check('跨角色副本');
    await click(renderer, '永久删除 2 个会话');
    assert.deepEqual(deletions, [[{ id: 'original', revision: 3 }, { id: 'copy', revision: 2 }]]);
  } finally {
    if (renderer) await unmount(renderer);
    globalThis.window = originalWindow;
  }
});

test('shared preview and copy work when only getRandomValues is available', async t => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: { getRandomValues: bytes => bytes.fill(42) },
  });
  t.after(() => Object.defineProperty(globalThis, 'crypto', previous));
  const requests = [];
  const SharedCharacterPreview = loader({
    getSharedCharacter: async () => ({ publication_id: 'published', settings, sources: [] }),
    copySharedCharacter: async (_token, _publication, request) => {
      requests.push(request);
      return character;
    },
  })('src/components/chat/CharacterSharing.tsx').SharedCharacterPreview;
  let renderer;
  await act(async () => {
    renderer = create(React.createElement(SharedCharacterPreview, { token: 'token', onCopied: async () => {} }));
  });
  try {
    await click(renderer, '预览或刷新分享内容');
    await click(renderer, '预览或刷新分享内容');
    await click(renderer, '确认复制此发布内容');
    assert.deepEqual(requests, ['2a'.repeat(16)]);
  } finally {
    await unmount(renderer);
  }
});

test('mixed conflict/deletion choices have no defaults; exact final result and one confirmation', async () => {
  const h = await panel(); await click(h.renderer, '检查并预览');
  assert.ok(button(h.renderer, '生成最终预览').props.disabled);
  assert.ok(h.renderer.root.findAllByType('input').every(n => !n.props.checked));
  assert.match(text(h.renderer.toJSON()), /作者已删除/);
  await choices(h.renderer); await click(h.renderer, '生成最终预览');
  assert.deepEqual(h.calls.preview[0].choices, { a: 'local', b: 'author' });
  assert.match(text(h.renderer.toJSON()), /删除（新版不包含此篇）/);
  const confirm = button(h.renderer, '确认保存此最终版本');
  await act(async () => { confirm.props.onClick(); confirm.props.onClick(); });
  assert.equal(h.calls.confirm.length, 1); assert.equal(h.calls.confirm[0].preview_id, 'exact');
  assert.equal(h.imported(), 1); assert.equal(h.locked(), false);
  await unmount(h.renderer);
});

test('expired confirmation drops choices and requires a fresh plan; cancel releases lock', async () => {
  const h = await panel(harness({ confirmCharacterSourceUpdate: async () => { throw new ErrorClass(409); } }));
  await click(h.renderer, '检查并预览'); await choices(h.renderer); await click(h.renderer, '生成最终预览');
  await click(h.renderer, '确认保存'); assert.match(text(h.renderer.toJSON()), /预览已失效/);
  assert.ok(!buttons(h.renderer).some(b => text(b).includes('确认保存')));
  await click(h.renderer, '重新检查');
  assert.ok(h.renderer.root.findAllByType('input').every(n => !n.props.checked));
  await click(h.renderer, '取消更新'); assert.equal(h.locked(), false); assert.equal(h.imported(), 0);
  await unmount(h.renderer);
});

test('unknown confirmation outcome retains exact preview for idempotent retry and blocks cancel', async () => {
  let attempts = 0; const received = [];
  const h = await panel(harness({ confirmCharacterSourceUpdate: async (_id, value) => {
    received.push(value); if (++attempts === 1) throw new Error('network'); return { ...character, revision: 2 };
  } }));
  await click(h.renderer, '检查并预览'); await choices(h.renderer); await click(h.renderer, '生成最终预览');
  await click(h.renderer, '确认保存'); assert.equal(h.locked(), true);
  assert.ok(button(h.renderer, '取消更新').props.disabled); assert.ok(button(h.renderer, '重新检查').props.disabled);
  await click(h.renderer, '重试同一确认'); assert.deepEqual(received[0], received[1]); assert.equal(h.locked(), false);
  await unmount(h.renderer);
});

for (const reason of ['baseline', 'collision', 'unknown-kind']) test(`read-only blocked plan: ${reason}`, async () => {
  const value = basePlan();
  if (reason === 'baseline') value.source_merge = { ...value.source_merge, status: 'blocked', reason: 'baseline_unavailable', conflicts: [] };
  else value.source_merge.conflicts[0].kind = reason === 'collision' ? 'addition_id_collision' : 'future_conflict';
  const h = await panel(harness({ getCharacterSourceUpdate: async () => value }));
  await click(h.renderer, '检查并预览');
  assert.ok(button(h.renderer, '生成最终预览').props.disabled);
  assert.equal(h.renderer.root.findAllByType('input').length, 0);
  await click(h.renderer, '取消更新'); await unmount(h.renderer);
});

test('conflict-free plan requests final preview with explicit empty choices', async () => {
  const value = basePlan(); value.source_merge = { status: 'ready', reason: null, conflicts: [], actions: [{ id: 'a', action: 'keep_local_modification' }], merged_sources: [doc('a')] };
  const h = await panel(harness({ getCharacterSourceUpdate: async () => value }));
  await click(h.renderer, '检查并预览'); assert.match(text(h.renderer.toJSON()), /保留本地修改/);
  await click(h.renderer, '生成最终预览'); assert.deepEqual(h.calls.preview[0].choices, {});
  await click(h.renderer, '取消更新'); await unmount(h.renderer);
});

test('preview errors cannot leave old choices or enable confirmation', async () => {
  const h = await panel(harness({ previewCharacterSourceUpdate: async () => { throw new ErrorClass(409); } }));
  await click(h.renderer, '检查并预览'); await choices(h.renderer); await click(h.renderer, '生成最终预览');
  assert.match(text(h.renderer.toJSON()), /重新检查并重新选择/);
  assert.ok(!buttons(h.renderer).some(b => text(b).includes('确认保存')));
  await click(h.renderer, '重新检查'); assert.ok(h.renderer.root.findAllByType('input').every(n => !n.props.checked));
  await click(h.renderer, '取消更新'); await unmount(h.renderer);
});

async function page(overrides) {
  const h = harness(overrides); let confirmations = 0, answer = false;
  globalThis.window = { location: { search: '', origin: 'http://test' }, confirm: () => { confirmations++; return answer; }, addEventListener() {}, removeEventListener() {} };
  globalThis.document = { addEventListener() {}, removeEventListener() {} };
  let renderer;
  const Component = h.load('src/app/character-library/page.tsx').default;
  await act(async () => { renderer = create(React.createElement(Component)); });
  await click(renderer, '旅人');
  return { ...h, renderer, confirmations: () => confirmations, answer: value => { answer = value; } };
}

test('read-only loaded sources are clean; cancel update preserves loaded content', async () => {
  const h = await page(); await click(h.renderer, '查看或作为新版起点');
  await click(h.renderer, '检查并预览'); assert.equal(h.confirmations(), 0);
  await click(h.renderer, '取消更新');
  assert.equal(h.renderer.root.findByProps({ id: 'source-body-a' }).props.value, 'a');
  await unmount(h.renderer);
});

test('dirty source edits survive cancellation, failed save, role save and selection attempts; saved source reloads plan', async () => {
  let fail = true;
  const h = await page({ publishCharacterSources: async () => { if (fail) throw new Error('save failed'); return { id: 'new', index_status: 'completed' }; } });
  await click(h.renderer, '查看或作为新版起点'); await change(h.renderer.root.findByProps({ id: 'source-body-a' }), 'keep me');
  const before = h.calls.get;
  await click(h.renderer, '检查并预览'); assert.equal(h.calls.get, before);
  await click(h.renderer, '创建角色'); assert.equal(h.renderer.root.findByProps({ id: 'source-body-a' }).props.value, 'keep me');
  const forms = h.renderer.root.findAllByType('form');
  await act(async () => forms[0].props.onSubmit({ preventDefault() {} })); assert.equal(h.calls.save, 0);
  await act(async () => forms[1].props.onSubmit({ preventDefault() {} }));
  assert.equal(h.renderer.root.findByProps({ id: 'source-body-a' }).props.value, 'keep me');
  fail = false; await act(async () => forms[1].props.onSubmit({ preventDefault() {} }));
  assert.ok(h.calls.get > before); const prompts = h.confirmations();
  await click(h.renderer, '检查并预览'); assert.equal(h.confirmations(), prompts);
  assert.ok(h.renderer.root.findAllByType('fieldset').some(f => f.props.disabled));
  assert.ok(button(h.renderer, '创建角色').props.disabled);
  await click(h.renderer, '取消更新'); await unmount(h.renderer);
});

test('settings edits are retained when discarding is cancelled, and only explicit discard clears them', async () => {
  const h = await page(); await change(h.renderer.root.findByProps({ id: 'character-name' }), 'unsaved');
  await click(h.renderer, '检查并预览'); assert.equal(h.renderer.root.findByProps({ id: 'character-name' }).props.value, 'unsaved');
  h.answer(true); await click(h.renderer, '检查并预览');
  assert.equal(h.renderer.root.findByProps({ id: 'character-name' }).props.value, settings.name);
  await click(h.renderer, '取消更新'); await unmount(h.renderer);
});

test('client sends only identity, choices and exact preview id, never server-generated body', async () => {
  const { api } = loader().call(null, 'src/lib/api.ts'); const requests = [];
  api.request = async (path, options) => { requests.push({ path, body: JSON.parse(options.body) }); return {}; };
  const value = { revision: 1, source_revision: 2, publication_id: 'pub', local_corpus_id: null, choices: { a: 'author' }, preview_id: 'exact', settings, sources: [doc('a')] };
  await api.confirmCharacterSourceUpdate('copy', value);
  assert.deepEqual(requests[0].body, { revision: 1, source_revision: 2, publication_id: 'pub', local_corpus_id: null, choices: { a: 'author' }, preview_id: 'exact' });
  await api.importCharacterSourceUpdate('copy', 1, 2, 'pub', null);
  assert.equal(requests[1].body.choices, undefined); assert.equal(requests[1].body.preview_id, undefined);
});

test('in-flight source save excludes duplicate saves and competing update/revision changes', async () => {
  let finish; let writes = 0;
  const pending = new Promise(resolve => { finish = resolve; });
  const h = await page({ publishCharacterSources: async () => { writes++; return pending; } });
  await click(h.renderer, '查看或作为新版起点'); await change(h.renderer.root.findByProps({ id: 'source-body-a' }), 'pending edit');
  const sourceForm = h.renderer.root.findAllByType('form')[1];
  await act(async () => { sourceForm.props.onSubmit({ preventDefault() {} }); sourceForm.props.onSubmit({ preventDefault() {} }); });
  assert.equal(writes, 1);
  assert.ok(button(h.renderer, '检查并预览').props.disabled);
  assert.ok(button(h.renderer, '创建角色').props.disabled);
  await act(async () => finish({ id: 'saved', index_status: 'completed' }));
  assert.equal(h.renderer.root.findByProps({ id: 'source-body-a' }).props.value, 'pending edit');
  assert.ok(!button(h.renderer, '检查并预览').props.disabled);
  await unmount(h.renderer);
});

test('removing every draft document is still dirty and cancelled update does not restore it', async () => {
  const h = await page(); await click(h.renderer, '查看或作为新版起点'); await click(h.renderer, '从草稿移除此篇');
  await click(h.renderer, '检查并预览'); assert.equal(h.confirmations(), 1);
  assert.equal(h.renderer.root.findAllByProps({ id: 'source-body-a' }).length, 0);
  assert.match(text(h.renderer.toJSON()), /有未保存的原作资料编辑/);
  await unmount(h.renderer);
});

test('failed settings save keeps draft; failed source loading after discard prompt keeps old draft', async () => {
  let loads = 0;
  const h = await page({ saveLibraryCharacter: async () => { throw new Error('settings failure'); },
    getCharacterSources: async () => { if (++loads > 1) throw new Error('load failure'); return { sources: [doc('a')] }; } });
  await change(h.renderer.root.findByProps({ id: 'character-name' }), 'unsaved name');
  await act(async () => h.renderer.root.findAllByType('form')[0].props.onSubmit({ preventDefault() {} }));
  assert.equal(h.renderer.root.findByProps({ id: 'character-name' }).props.value, 'unsaved name');
  await click(h.renderer, '查看或作为新版起点'); await change(h.renderer.root.findByProps({ id: 'source-body-a' }), 'unsaved source');
  h.answer(true); await click(h.renderer, '查看或作为新版起点');
  assert.equal(h.renderer.root.findByProps({ id: 'source-body-a' }).props.value, 'unsaved source');
  await unmount(h.renderer);
});

test('double preview clicks send one request and lock competing sharing controls', async () => {
  const h = await page();
  const start = button(h.renderer, '检查并预览');
  await act(async () => { start.props.onClick(); start.props.onClick(); });
  assert.equal(h.calls.get, 2); // mount plus one explicit check
  const shareSection = h.renderer.root.findByProps({ 'aria-label': '发布角色分享' });
  assert.ok(shareSection.findByType('fieldset').props.disabled);
  await click(h.renderer, '取消更新'); await unmount(h.renderer);
});
