'use client';

import { useRef, useState } from 'react';
import { api } from '@/lib/api';
import { createClientId } from '@/lib/client-id';
import { Button } from '@/components/ui/button';
import type { CharacterLibraryRecord, CharacterOriginalSource, CharacterPublicationPreview, CharacterSourceEdition, CharacterSourceEditionSummary, SharedCharacterPublication } from '@/lib/api-contracts';

export function SourcePreview({ sources }: { sources: CharacterOriginalSource[] }) {
  return <div className="space-y-2">
    {sources.length === 0 && <p className="text-sm">不包含独立原作资料。</p>}
    {sources.map(source => <details key={source.id} className="rounded border p-2">
      <summary className="cursor-pointer">{source.title}</summary>
      <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-sm">{source.content}</pre>
    </details>)}
  </div>;
}

export function SourceChanges({ current, incoming }: { current: CharacterOriginalSource[]; incoming: CharacterOriginalSource[] }) {
  const before = new Map(current.map(source => [source.id, source]));
  const after = new Map(incoming.map(source => [source.id, source]));
  return <div className="space-y-2">
    <p className="font-medium">独立原作资料变化</p>
    {incoming.map(source => <p key={source.id} className="text-sm">
      {before.has(source.id) ? before.get(source.id)?.title === source.title && before.get(source.id)?.content === source.content ? '保留' : '替换' : '新增'}：{source.title}
    </p>)}
    {current.filter(source => !after.has(source.id)).map(source => <p key={source.id} className="text-sm">新版不再携带：{source.title}</p>)}
    <details><summary className="cursor-pointer">查看当前资料全文</summary><SourcePreview sources={current} /></details>
    <details><summary className="cursor-pointer">查看更新资料全文</summary><SourcePreview sources={incoming} /></details>
  </div>;
}

export function CharacterSharing({ character, onPublished, disabled = false, acquire, release }: {
  disabled?: boolean; acquire?: () => boolean; release?: () => void;
  character: CharacterLibraryRecord;
  onPublished: () => Promise<void>;
}) {
  const [editions, setEditions] = useState<CharacterSourceEditionSummary[]>([]);
  const [more, setMore] = useState(false);
  const [edition, setEdition] = useState<CharacterSourceEdition>();
  const [selected, setSelected] = useState<string[]>([]);
  const [preview, setPreview] = useState<CharacterPublicationPreview>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const flight = useRef(false);

  async function run(operation: () => Promise<void>) {
    if (flight.current || (acquire && !acquire())) return;
    flight.current = true; setBusy(true); setError(''); setNotice('');
    try { await operation(); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { flight.current = false; setBusy(false); release?.(); }
  }

  async function load(offset = 0) {
    const result = await api.listCharacterSources(character.id, character.revision, offset);
    setEditions(previous => offset ? [...previous, ...result.corpora] : result.corpora);
    setMore(result.corpora.length === 50);
  }

  return <section aria-label="发布角色分享" className="space-y-3 rounded-xl border p-5">
    <h2 className="text-lg font-medium">发布角色分享</h2>
    <p className="text-sm text-muted-foreground">仅发布已保存的设定。独立资料默认不选；可从当前角色版本的一份资料集中逐篇选择。不会携带私人对话、记忆、聊天关系或账号配置。</p>
    {error && <p role="alert" className="text-destructive">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    <fieldset disabled={busy || disabled} className="space-y-3">
      <Button type="button" variant="outline" onClick={() => void run(() => load())}>加载或刷新可选资料版本</Button>
      <label className="block space-y-1">资料版本
        <select aria-label="分享资料版本" className="block w-full rounded border bg-background p-2" value={edition?.id || ''}
          onChange={event => {
            const id = event.target.value;
            setEdition(undefined); setSelected([]); setPreview(undefined);
            if (id) void run(async () => { setEdition(await api.getCharacterSources(character.id, id)); });
          }}>
          <option value="">不选择独立资料</option>
          {editions.map(item => <option key={item.id} value={item.id}>{new Date(item.created_at).toLocaleString()} · {item.id.slice(0, 8)}</option>)}
        </select>
      </label>
      {more && <Button type="button" variant="outline" onClick={() => void run(() => load(editions.length))}>加载更多可选版本</Button>}
      {edition?.sources.map(source => <div key={source.id} className="space-y-1">
        <label className="flex items-center gap-2"><input type="checkbox" checked={selected.includes(source.id)} onChange={event => {
          setSelected(previous => event.target.checked ? [...previous, source.id] : previous.filter(id => id !== source.id));
          setPreview(undefined);
        }} />分享：{source.title}</label>
        <details><summary className="cursor-pointer text-sm">查看原文</summary><p className="max-h-64 overflow-auto whitespace-pre-wrap text-sm">{source.content}</p></details>
      </div>)}
      <Button type="button" onClick={() => void run(async () => {
        setPreview(await api.previewCharacterPublication(character, selected.length ? edition!.id : null, selected));
      })}>预览将分享的内容</Button>
      {preview && <div className="space-y-3 rounded border p-3" aria-label="分享发布预览">
        <h3 className="font-medium">{preview.settings.name} · 设定版本 {preview.revision} · {preview.sources.length} 篇独立资料</h3>
        <details><summary className="cursor-pointer">查看将分享的完整设定</summary><pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-sm">{JSON.stringify(preview.settings, null, 2)}</pre></details>
        <SourcePreview sources={preview.sources} />
        <p className="text-sm">发布后，持有链接并登录的人可预览并复制这些内容。关闭分享阻止新复制，已有独立副本保留。</p>
        <Button type="button" onClick={() => void run(async () => {
          await api.publishLibraryCharacter(character, true, preview);
          setPreview(undefined); setNotice('已发布预览中的内容。'); await onPublished();
        })}>确认发布这些内容</Button>
        <Button type="button" variant="outline" onClick={() => setPreview(undefined)}>取消发布</Button>
      </div>}
    </fieldset>
  </section>;
}

export function SharedCharacterPreview({ token, onCopied, disabled = false, acquire, release }: { token: string; onCopied: (value: CharacterLibraryRecord) => Promise<void>; disabled?: boolean; acquire?: () => boolean; release?: () => void }) {
  const [preview, setPreview] = useState<SharedCharacterPublication>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const request = useRef('');

  const flight = useRef(false);

  async function run(operation: () => Promise<void>) {
    if (flight.current || (acquire && !acquire())) return;
    flight.current = true; setBusy(true); setError('');
    try { await operation(); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { flight.current = false; setBusy(false); release?.(); }
  }

  return <section className="space-y-3 rounded-lg border p-4" aria-label="收到角色分享">
    <p>收到角色分享，请先预览。确认后创建你的独立角色和资料副本。</p>
    {error && <p role="alert" className="text-destructive">{error}</p>}
    <Button disabled={busy || disabled} onClick={() => void run(async () => {
      const result = await api.getSharedCharacter(token);
      if (result.publication_id !== preview?.publication_id) request.current = createClientId();
      setPreview(result);
    })}>预览或刷新分享内容</Button>
    {preview && <>
      <h2 className="font-medium">{preview.settings.name} · {preview.sources.length} 篇独立资料</h2>
      <details><summary className="cursor-pointer">查看完整角色设定</summary><pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-sm">{JSON.stringify(preview.settings, null, 2)}</pre></details>
      <SourcePreview sources={preview.sources} />
      <p className="text-sm">副本可自行修改，作者更新不会自动覆盖。复制后的资料需要独立处理，可在资料管理中查看状态和重试。</p>
      <Button disabled={busy || disabled} onClick={() => void run(async () => {
        await onCopied(await api.copySharedCharacter(token, preview.publication_id, request.current));
      })}>确认复制此发布内容</Button>
    </>}
  </section>;
}
