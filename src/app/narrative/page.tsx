'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { api, type GenerateResponse } from '@/lib/api';

type Branch = { id: string; character_id: string; title: string; initial_hypothesis: string; status: string; revision: number };
type Assertion = { id: string; subject: string; predicate: string; object: string; assertion_kind: string; status: string; source_type: string };
type Chat = { role: 'user' | 'assistant'; content: string; sources?: GenerateResponse['evidenceSources'] };
type Character = { character_id: string; display_name: string };

export default function NarrativePage() {
  return <AuthGuard><AppLayout><Workspace /></AppLayout></AuthGuard>;
}

function Workspace() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [character, setCharacter] = useState('');
  const [branches, setBranches] = useState<Branch[]>([]);
  const [selected, setSelected] = useState('');
  const [branch, setBranch] = useState<Branch | null>(null);
  const [assertions, setAssertions] = useState<Assertion[]>([]);
  const [messages, setMessages] = useState<Chat[]>([]);
  const [title, setTitle] = useState('');
  const [hypothesis, setHypothesis] = useState('');
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const generation = useRef(0);
  const retry = useRef<{ key: string; trace: string } | null>(null);

  const refreshList = useCallback(async () => {
    const all: Branch[] = [];
    for (let offset = 0; ; offset += 100) {
      const result = await api.narrative<{ branches: Branch[] }>(`?limit=100&offset=${offset}`);
      all.push(...result.branches);
      if (result.branches.length < 100) break;
    }
    setBranches(all);
  }, []);

  useEffect(() => {
    let live = true;
    api.narrative<{ characters: Character[] }>('/characters').then(result => {
      if (!live) return;
      setCharacters(result.characters);
      setCharacter(result.characters[0]?.character_id || '');
      return refreshList();
    }).catch(e => { if (live) setError(String(e)); });
    return () => { live = false; };
  }, [refreshList]);

  const loadBranch = useCallback(async (id: string, ticket: number) => {
    const [detail, active, pending, history] = await Promise.all([
      api.narrative<Branch>(`/${id}`),
      api.narrative<{ assertions: Assertion[] }>(`/${id}/assertions?status=active&limit=100`),
      api.narrative<{ assertions: Assertion[] }>(`/${id}/assertions?status=pending&limit=100`),
      api.narrative<{ messages: Array<{ message: string; reply: string }> }>(`/${id}/messages`),
    ]);
    if (ticket !== generation.current) return;
    setBranch(detail);
    setAssertions([...active.assertions, ...pending.assertions]);
    setMessages(history.messages.flatMap(m => [
      { role: 'user' as const, content: m.message }, { role: 'assistant' as const, content: m.reply },
    ]));
  }, []);

  useEffect(() => {
    const ticket = ++generation.current;
    setMessages([]); setAssertions([]); setBranch(null); setError(''); setInput('');
    retry.current = null;
    if (!selected) {
      if (!character) { setLoading(false); return; }
      setLoading(true);
      api.narrative<{ messages: Array<{ message: string; reply: string }> }>(`/canonical/messages?character_id=${encodeURIComponent(character)}`)
        .then(result => {
          if (ticket === generation.current) setMessages(result.messages.flatMap(m => [
            { role: 'user' as const, content: m.message }, { role: 'assistant' as const, content: m.reply },
          ]));
        }).catch(e => { if (ticket === generation.current) setError(String(e)); })
        .finally(() => { if (ticket === generation.current) setLoading(false); });
      return;
    }
    setLoading(true);
    loadBranch(selected, ticket).catch(e => {
      if (ticket === generation.current) setError(String(e));
    }).finally(() => { if (ticket === generation.current) setLoading(false); });
  }, [selected, character, loadBranch]);

  async function create() {
    setBusy(true); setError('');
    try {
      const result = await api.narrative<Branch>('', {
        character_id: character, title, initial_hypothesis: hypothesis,
      });
      await refreshList(); setSelected(result.id); setTitle(''); setHypothesis('');
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy || loading) return;
    setBusy(true); setError('');
    const key = JSON.stringify([selected, character, text]);
    if (retry.current?.key !== key) retry.current = { key, trace: crypto.randomUUID() };
    try {
      const response = selected
        ? await api.generateReply({ message: text, branchId: selected, platform: 'web', traceId: retry.current.trace })
        : await api.narrative<GenerateResponse>('/canonical/generate', {
          character_id: character, message: text, traceId: retry.current.trace,
        });
      if (selected) {
        await loadBranch(selected, generation.current);
        setMessages(items => items.map((m, i) => i === items.length - 1
          ? { ...m, sources: response.evidenceSources } : m));
      } else {
        setMessages(items => [...items, { role: 'user', content: text }, { role: 'assistant', content: response.reply }]);
      }
      setInput(''); retry.current = null;
    } catch (e) {
      setError(`${String(e)}。可重试；若提示版本冲突，请刷新分支。`);
    } finally { setBusy(false); }
  }

  async function decide(item: Assertion, action: 'confirm' | 'reject') {
    if (!branch) return;
    const normalize = (s: string) => s.trim().replace(/\s+/g, ' ').toLowerCase();
    const conflicts = action === 'confirm' ? assertions.filter(a => a.status === 'active' && a.assertion_kind !== 'premise'
      && normalize(a.subject) === normalize(item.subject) && normalize(a.predicate) === normalize(item.predicate)
      && normalize(a.object) !== normalize(item.object)) : [];
    if (conflicts.length && !window.confirm(`将替代以下事实及其依赖事实：\n${conflicts.map(a => a.object).join('\n')}\n是否继续？`)) return;
    setBusy(true); setError('');
    try {
      await api.narrative(`/${branch.id}/assertions/${item.id}/${action}`, {
        revision: branch.revision, replace_ids: conflicts.map(a => a.id),
      });
      await loadBranch(branch.id, generation.current);
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }

  async function archive() {
    if (!branch) return;
    setBusy(true); setError('');
    try {
      await api.narrative(`/${branch.id}/archive`, { revision: branch.revision });
      await refreshList(); await loadBranch(branch.id, generation.current);
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }

  const disabled = busy || loading;
  return <div className="space-y-5 max-w-6xl mx-auto">
    <div><h1 className="text-2xl font-semibold">假想分支实验室</h1>
      <p className="text-muted-foreground mt-2">保持角色，改变故事前提。每个分支的历史与事实独立保存。</p></div>
    {error && <p role="alert" className="rounded border border-destructive p-3 text-sm">{error}</p>}
    <div className="flex flex-wrap items-center gap-3">
      <select aria-label="角色" className="border rounded p-2 bg-background" disabled={disabled}
        value={character} onChange={e => { setSelected(''); setCharacter(e.target.value); }}>
        {characters.map(c => <option key={c.character_id} value={c.character_id}>{c.display_name}</option>)}
      </select>
      <select aria-label="世界模式" className="border rounded p-2 bg-background" disabled={disabled}
        value={selected} onChange={e => setSelected(e.target.value)}>
        <option value="">正史模式</option>
        {branches.filter(b => b.character_id === character).map(b => <option key={b.id} value={b.id}>
          {b.title}{b.status === 'archived' ? '（已归档）' : ''}</option>)}
      </select>
      {selected && <Button variant="outline" disabled={disabled} onClick={() => setSelected('')}>返回正史</Button>}
      {branch && <Button variant="outline" disabled={disabled} onClick={() => {
        setLoading(true); loadBranch(branch.id, generation.current).catch(e => setError(String(e))).finally(() => setLoading(false));
      }}>刷新分支</Button>}
      {branch?.status === 'active' && <Button variant="outline" disabled={disabled} onClick={archive}>归档分支</Button>}
    </div>
    <div className="grid gap-5 lg:grid-cols-[1fr_340px]">
      <section className="border rounded-xl p-4 space-y-4">
        <h2 className="font-medium">{selected ? `假想分支 · ${branch?.title || '加载中'}` : '正史模式'}</h2>
        {branch && <p className="text-sm bg-muted rounded p-3">当前假设：{branch.initial_hypothesis}</p>}
        <div className="h-[430px] overflow-y-auto space-y-4" aria-live="polite">
          {!messages.length && <p className="text-muted-foreground text-sm">{loading ? '正在加载…' : '发送消息开始对话。分支回复中的推演需确认后才成为有效事实。'}</p>}
          {messages.map((m, i) => <div key={i} className={`rounded-lg p-3 ${m.role === 'user' ? 'bg-primary/10' : 'bg-muted'}`}>
            <p className="text-xs text-muted-foreground mb-1">{m.role === 'user' ? '你' : '角色'}</p>
            <p className="whitespace-pre-wrap break-words">{m.content}</p>
            {!!m.sources?.length && <details className="text-xs mt-3"><summary>本轮可用证据（不代表逐句语义核验）</summary>
              {m.sources.map((s, j) => <p key={j}>{s.type === 'current_hypothesis' ? '当前假设' : s.type === 'canonical_evidence' ? '原作证据' : '已确认分支事实'} · {s.id || JSON.stringify(s.citation)}</p>)}
            </details>}
          </div>)}
        </div>
        <Textarea aria-label="消息" maxLength={8000} value={input} onChange={e => setInput(e.target.value)} disabled={disabled || branch?.status === 'archived'} />
        <Button onClick={send} disabled={disabled || !character || !input.trim() || branch?.status === 'archived'}>{busy ? '处理中…' : '发送'}</Button>
      </section>
      <aside className="space-y-5">
        <section className="border rounded-xl p-4 space-y-3">
          <h2 className="font-medium">创建独立分支</h2>
          <Input aria-label="分支标题" placeholder="例如：从未相遇" maxLength={100} value={title} onChange={e => setTitle(e.target.value)} disabled={disabled} />
          <Textarea aria-label="初始假设" placeholder="明确写下与原作不同的前提…" maxLength={2000} value={hypothesis} onChange={e => setHypothesis(e.target.value)} disabled={disabled} />
          <Button onClick={create} disabled={disabled || !character || !title.trim() || !hypothesis.trim()}>创建假想分支</Button>
        </section>
        {branch && <section className="border rounded-xl p-4 space-y-3">
          <h2 className="font-medium">分支事实与待确认提议</h2>
          <p className="text-xs text-muted-foreground">核对语义与初始假设后再确认；系统只自动检查直接字段冲突。待确认项最多展示100条。</p>
          {assertions.map(a => <div key={a.id} className="border rounded p-3 text-sm space-y-2">
            <p className="text-xs text-muted-foreground">{a.status === 'pending' ? '待确认推演' : a.assertion_kind === 'premise' ? '初始假设 · 未结构化' : '已确认事实'}</p>
            <p>{a.subject} {a.predicate} {a.object}</p>
            {a.status === 'pending' && branch.status === 'active' && <div className="flex gap-2">
              <Button size="sm" disabled={disabled} onClick={() => decide(a, 'confirm')}>确认</Button>
              <Button size="sm" variant="outline" disabled={disabled} onClick={() => decide(a, 'reject')}>拒绝</Button>
            </div>}
          </div>)}
        </section>}
      </aside>
    </div>
  </div>;
}
