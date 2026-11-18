'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { api } from '@/lib/api';
import type { CharacterLibraryRecord, CharacterOriginalSource, CharacterSourceEditionSummary } from '@/lib/api-contracts';

const processingLabels = { pending: '等待处理', submitted: '处理中', completed: '处理完成', failed: '处理失败' };

export function CharacterSourceManager({ character }: { character: CharacterLibraryRecord }) {
  const [editions, setEditions] = useState<CharacterSourceEditionSummary[]>([]);
  const [sources, setSources] = useState<CharacterOriginalSource[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [more, setMore] = useState(false);
  const [statusError, setStatusError] = useState('');
  const hasPending = editions.some(edition => edition.index_status === 'pending' || edition.index_status === 'submitted');

  useEffect(() => {
    if (busy || !hasPending) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const updated: CharacterSourceEditionSummary[] = [];
        let hasMore = false;
        for (let offset = 0; offset < Math.max(editions.length, 50); offset += 50) {
          const result = await api.listCharacterSources(character.id, character.revision, offset);
          if (!active) return;
          updated.push(...result.corpora);
          hasMore = result.corpora.length === 50;
          if (!hasMore) break;
        }
        if (active) {
          setEditions([...new Map(updated.map(item => [item.id, item])).values()]);
          setMore(hasMore); setStatusError('');
        }
      } catch {
        if (active) setStatusError('暂时无法刷新处理状态，稍后会自动重试。');
      } finally {
        if (active) timer = setTimeout(() => void refresh(), 5000);
      }
    }
    timer = setTimeout(() => void refresh(), 5000);
    return () => { active = false; clearTimeout(timer); };
  }, [busy, hasPending, editions.length, character.id, character.revision]);

  useEffect(() => {
    let active = true;
    api.listCharacterSources(character.id, character.revision).then(result => {
      if (active) { setEditions(result.corpora); setMore(result.corpora.length === 50); }
    }).catch(error => { if (active) setError(String(error)); });
    return () => { active = false; };
  }, [character.id, character.revision]);

  async function run(operation: () => Promise<void>) {
    setBusy(true); setError(''); setNotice('');
    try { await operation(); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  }

  function edit(index: number, field: 'title' | 'content', value: string) {
    setSources(previous => previous.map((source, i) => i === index ? { ...source, [field]: value } : source));
  }

  return <section className="space-y-4 rounded-xl border p-5" aria-label="角色原作资料">
    <h2 className="text-lg font-medium">{character.settings.name}的原作资料</h2>
    <p className="text-sm text-muted-foreground">资料属于已保存的人物设定版本 {character.revision}。每次发布保存完整资料集；旧会话保留原版，新会话使用新版。资料需完成处理后才能用于原作问答。</p>
    {error && <p role="alert" className="text-destructive">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    {statusError && <p role="status" className="text-sm text-muted-foreground">{statusError}</p>}
    <details>
      <summary className="cursor-pointer">查看已发布资料（已加载 {editions.length} 版）</summary>
      <ul className="mt-3 space-y-2">{editions.map(edition => <li key={edition.id} className="flex flex-wrap items-center gap-2 rounded border p-2">
        <span className="text-sm">{new Date(edition.created_at).toLocaleString()} · {edition.id.slice(0, 8)}</span>
        <span className="text-sm" role="status">{processingLabels[edition.index_status]} · 第 {edition.index_attempt} 次处理</span>
        {edition.index_status === 'failed' && <Button type="button" variant="outline" disabled={busy} onClick={() => void run(async () => {
          const result = await api.retryCharacterSources(character.id, edition.id, edition.index_attempt);
          setEditions(previous => previous.map(item => item.id === result.id ? result : item));
          setNotice('已重新提交处理，完成后可用于原作问答。');
        })}>重试处理</Button>}
        <Button type="button" variant="outline" disabled={busy} onClick={() => void run(async () => {
          if (sources.length && !window.confirm('载入该版本会替换下方未发布的草稿，是否继续？')) return;
          const result = await api.getCharacterSources(character.id, edition.id);
          setSources(result.sources); setNotice('已载入原文，可查看或编辑后另发新版。');
        })}>查看或作为新版起点</Button>
        <Button type="button" variant="outline" disabled={busy} onClick={() => {
          if (!window.confirm('撤销此资料版本后，绑定它的会话将无法再检索该版原文。包含它的分享也将停止新预览和复制；已有独立副本、聊天和出处快照不会因此删除。是否继续？')) return;
          void run(async () => { await api.revokeCharacterSources(character.id, edition.id); setEditions(previous => previous.filter(item => item.id !== edition.id)); setNotice('资料版本已撤销。'); });
        }}>撤销</Button>
      </li>)}</ul>
      {more && <Button type="button" variant="outline" disabled={busy} onClick={() => void run(async () => {
        const result = await api.listCharacterSources(character.id, character.revision, editions.length);
        setEditions(previous => [...previous, ...result.corpora]); setMore(result.corpora.length === 50);
      })}>加载更多版本</Button>}
    </details>
    <form className="space-y-4" onSubmit={event => { event.preventDefault(); void run(async () => {
      const result = await api.publishCharacterSources(character.id, character.revision, sources);
      setEditions(previous => [result, ...previous]); setSources([]); setNotice('新版资料已保存。现有会话没有切换版本。');
    }); }}>
      <fieldset disabled={busy} className="space-y-4">
        {sources.map((source, index) => <div key={source.id} className="space-y-2 rounded border p-3">
          <Label htmlFor={`source-title-${source.id}`}>资料 {index + 1} 标题</Label>
          <Input id={`source-title-${source.id}`} required maxLength={300} value={source.title} onChange={event => edit(index, 'title', event.target.value)} />
          <Label htmlFor={`source-body-${source.id}`}>原文（按段落换行）</Label>
          <textarea id={`source-body-${source.id}`} required rows={8} maxLength={200000} value={source.content} onChange={event => edit(index, 'content', event.target.value)} className="w-full rounded-md border bg-background p-3 text-sm" />
          <Button type="button" variant="outline" onClick={() => setSources(previous => previous.filter((_, i) => i !== index))}>从草稿移除此篇</Button>
        </div>)}
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" disabled={sources.length >= 100} onClick={() => setSources(previous => [...previous, { id: crypto.randomUUID(), title: '', content: '' }])}>添加一篇资料</Button>
          <Button type="submit" disabled={!sources.length}>发布完整资料集为新版</Button>
        </div>
      </fieldset>
    </form>
  </section>;
}
