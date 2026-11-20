'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { api, ApiRequestError } from '@/lib/api';
import { createClientId } from '@/lib/client-id';
import { GenerationDraft } from '@/components/chat/GenerationDraft';
import type { CharacterChatDeletionPreview, CharacterChatGeneration } from '@/lib/api-contracts';
import type { CharacterChatSession, CharacterChatSummary, CharacterChatTurn, CharacterLibraryRecord, SendCharacterChatMessage, ContinuationPreview, CharacterSharedMemory, CharacterChatEvidence } from '@/lib/api-contracts';

function AnswerSources({ evidence }: { evidence?: CharacterChatEvidence }) {
  if (!evidence) return null;
  return <div className="space-y-2 text-sm">
    {evidence.retrieval_status === 'character_abstention' && <p className="text-muted-foreground">本轮未取得足够的资料依据。</p>}
    {!!evidence.sources?.length && <details className="rounded-lg border p-3">
      <summary className="cursor-pointer">本条回答的资料出处（{evidence.sources.length}）</summary>
      <p className="my-2 text-xs text-muted-foreground">以下保存的是回答生成时的资料片段。出处可核对，不代表模型对原文的理解一定正确。</p>
      {evidence.sources.map((source, index) => <section key={`${source.id}-${index}`} className="mt-3 space-y-1">
        <p className="font-medium">{source.title}</p>
        {source.line_start != null && <p className="text-xs text-muted-foreground">原文第 {source.line_start}{source.line_end != null ? `—${source.line_end}` : ''} 行</p>}
        <blockquote className="whitespace-pre-wrap break-words border-l-2 pl-3">{source.excerpt || '本条记录未保存原文片段。'}</blockquote>
        {source.truncated && <p className="text-xs text-muted-foreground">片段较长，此处保存了部分内容。</p>}
      </section>)}
    </details>}
  </div>;
}

export default function ChatPage() {
  return <AuthGuard><AppLayout><ChatWorkspace /></AppLayout></AuthGuard>;
}

function ChatWorkspace() {
  const [characters, setCharacters] = useState<CharacterLibraryRecord[]>([]);
  const [characterOffset, setCharacterOffset] = useState(0);
  const [moreCharacters, setMoreCharacters] = useState(false);
  const [characterId, setCharacterId] = useState('');
  const [sessions, setSessions] = useState<CharacterChatSummary[]>([]);
  const [sessionOffset, setSessionOffset] = useState(0);
  const [moreSessions, setMoreSessions] = useState(false);
  const [deleteSelection, setDeleteSelection] = useState<string[]>([]);
  const [deletionPreview, setDeletionPreview] = useState<CharacterChatDeletionPreview>();
  const [relatedDeletionIds, setRelatedDeletionIds] = useState<string[]>([]);
  const [sourceExcerpt, setSourceExcerpt] = useState<{ message: string; reply: string; title: string }>();
  const [session, setSession] = useState<CharacterChatSession>();
  const [turns, setTurns] = useState<CharacterChatTurn[]>([]);
  const [generation, setGeneration] = useState<CharacterChatGeneration | null>(null);
  const [enrichmentPending, setEnrichmentPending] = useState(false);
  const [enrichmentRetryable, setEnrichmentRetryable] = useState(false);
  const [streamRequest, setStreamRequest] = useState<{ sessionId: string; requestId: string } | null>(null);
  const [source, setSource] = useState('');
  const [title, setTitle] = useState('新的相遇');
  const [acceptChange, setAcceptChange] = useState(false);
  const [preview, setPreview] = useState<ContinuationPreview>();
  const [excludedMemories, setExcludedMemories] = useState<string[]>([]);
  const [resetRelationship, setResetRelationship] = useState(false);
  const [inheritHistory, setInheritHistory] = useState(true);
  const [sharedMemories, setSharedMemories] = useState<CharacterSharedMemory[]>([]);
  const [selectedShared, setSelectedShared] = useState<string[]>([]);
  const [moreShared, setMoreShared] = useState(false);
  const [draft, setDraft] = useState('');
  const [editing, setEditing] = useState<string>();
  const [branchTurn, setBranchTurn] = useState<string>();
  const [hypothesis, setHypothesis] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [newSession, setNewSession] = useState(true);
  const [localOnly, setLocalOnly] = useState(false);
  const [memoryDraft, setMemoryDraft] = useState('');
  const [memoryEditing, setMemoryEditing] = useState<string>();
  const [relationshipStage, setRelationshipStage] = useState<CharacterChatSession['state']['relationship']['stage']>('stranger');
  const [relationshipSummary, setRelationshipSummary] = useState('');
  const pending = useRef<{ sessionId: string; body: SendCharacterChatMessage } | null>(null);
  const selectedCharacter = characters.find(item => item.id === characterId);
  const sourceSession = sessions.find(item => item.id === source);
  const settingsChanged = Boolean(preview && preview.previous_character_revision !== preview.character_revision);
  const loadVersion = useRef(0);

  const refreshShared = useCallback(async () => {
    const result = await api.listSharedChatMemories();
    setSharedMemories(result.memories); setMoreShared(result.memories.length === 100); setSelectedShared([]);
  }, []);

  useEffect(() => { refreshShared().catch(error => setError(String(error))); }, [refreshShared]);

  useEffect(() => {
    let active = true;
    setPreview(undefined);
    setSelectedShared([]);
    setAcceptChange(false);
    setExcludedMemories([]);
    setResetRelationship(false);
    setInheritHistory(true);
    if (source) api.getContinuationPreview(source).then(result => { if (active) setPreview(result); })
      .catch(error => { if (active) setError(String(error)); });
    return () => { active = false; };
  }, [source]);

  const refreshSessions = useCallback(async (id: string) => {
    const response = await api.listCharacterChats(id);
    setSessions(response.sessions);
    setSessionOffset(response.sessions.length); setMoreSessions(response.sessions.length === 100);
  }, []);

  useEffect(() => {
    let active = true;
    api.listCharacterLibrary().then(async response => {
      const requested = new URLSearchParams(window.location.search).get('character');
      let requestedCharacter: CharacterLibraryRecord | undefined;
      if (requested && !response.characters.some(item => item.id === requested)) {
        try { requestedCharacter = await api.getLibraryCharacter(requested); }
        catch (error) { if (active) setError(error instanceof Error ? error.message : String(error)); }
      }
      if (!active) return;
      setCharacters(requestedCharacter ? [requestedCharacter, ...response.characters] : response.characters);
      setCharacterOffset(response.characters.length); setMoreCharacters(response.characters.length === 100);
      setCharacterId(requestedCharacter?.id || response.characters.find(item => item.id === requested)?.id || response.characters[0]?.id || '');
    }).catch(error => { if (active) setError(String(error)); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const version = ++loadVersion.current;
    setSessions([]);
    setDeleteSelection([]);
    setDeletionPreview(undefined);
    setRelatedDeletionIds([]);
    setSessionOffset(0); setMoreSessions(false); setSourceExcerpt(undefined);
    setSelectedShared([]);
    setSession(undefined);
    setGeneration(null);
    setEnrichmentPending(false);
    setEnrichmentRetryable(false);
    setTurns([]);
    setSource('');
    setAcceptChange(false);
    setNewSession(true);
    setDraft('');
    setEditing(undefined);
    setBranchTurn(undefined);
    pending.current = null;
    if (!characterId) return;
    api.listCharacterChats(characterId).then(response => {
      if (version === loadVersion.current) {
        setSessions(response.sessions); setSessionOffset(response.sessions.length); setMoreSessions(response.sessions.length === 100);
      }
    }).catch(error => { if (version === loadVersion.current) setError(String(error)); });
  }, [characterId]);

  async function run(operation: () => Promise<void>) {
    setBusy(true);
    setError('');
    try { await operation(); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  }

  function clearDeletionPreview() {
    setDeletionPreview(undefined);
    setRelatedDeletionIds([]);
  }

  async function reviewDeletion() {
    clearDeletionPreview();
    const selected = sessions.filter(item => deleteSelection.includes(item.id));
    try {
      setDeletionPreview(await api.previewCharacterChatDeletion(selected.map(({ id, revision }) => ({ id, revision }))));
    } catch (error) {
      if (error instanceof ApiRequestError && [404, 409].includes(error.status)) await refreshSessions(characterId);
      throw error;
    }
  }

  async function confirmDeletion() {
    if (!deletionPreview) return;
    const selected = [...deletionPreview.selected, ...deletionPreview.related.filter(item => relatedDeletionIds.includes(item.id))];
    try {
      const result = await api.deleteCharacterChats(selected.map(({ id, revision }) => ({ id, revision })));
      setDeleteSelection([]); setSource(''); setSourceExcerpt(undefined);
      if (session && result.deleted_session_ids.includes(session.id)) {
        setSession(undefined); setTurns([]); setNewSession(true); setDraft(''); setEditing(undefined); setBranchTurn(undefined); pending.current = null;
      }
      await refreshSessions(characterId); await refreshShared();
    } catch (error) {
      if (error instanceof ApiRequestError && [404, 409].includes(error.status)) await refreshSessions(characterId);
      throw error;
    } finally {
      clearDeletionPreview();
    }
  }

  async function open(id: string) {
    const response = await api.getCharacterChat(id);
    setSession(response.session);
    setGeneration(response.generation);
    setEnrichmentPending(response.enrichment_pending);
    setEnrichmentRetryable(response.enrichment_retryable);
    setSessions(previous => previous.map(item => item.id === id ? { ...item, revision: response.session.revision } : item));
    setRelationshipStage(response.session.state.relationship.stage);
    setRelationshipSummary(response.session.state.relationship.summary || '');
    setSourceExcerpt(undefined);
    setMemoryDraft('');
    setMemoryEditing(undefined);
    setTurns(response.turns);
    setNewSession(false);
    setBranchTurn(undefined);
    return response;
  }

  async function send() {
    if (!session || !draft.trim() || generation?.status === 'running' || (enrichmentPending && !editing)) return;
    const existing = pending.current;
    const body: SendCharacterChatMessage = existing?.sessionId === session.id && existing.body.message === draft && (existing.body.replace_turn_id || undefined) === editing
      ? existing.body : { revision: session.revision, request_id: createClientId(), message: draft, ...(editing ? { replace_turn_id: editing } : {}) };
    pending.current = { sessionId: session.id, body };
    setStreamRequest({ sessionId: session.id, requestId: body.request_id });
    try {
      await api.sendCharacterChat(session.id, body);
      setDraft('');
      setEditing(undefined);
      pending.current = null;
      await open(session.id);
      await refreshSessions(characterId);
    } catch (error) {
      // A timeout may follow a committed response. Read back before offering a retry.
      try {
        const response = await open(session.id);
        if (response.turns.some(turn => turn.request_id === body.request_id)) {
          pending.current = null;
          setDraft('');
          setEditing(undefined);
          return;
        }
        if (response.session.revision !== body.revision) pending.current = null;
      } catch { /* Keep the original failure and exact retry identity. */ }
      throw error;
    } finally {
      setStreamRequest(null);
    }
  }

  const activeSessionId = session?.id;
  const watchedRequest = streamRequest && streamRequest.sessionId === activeSessionId ? streamRequest.requestId
    : generation?.status === 'running' ? generation.request_id : undefined;
  useEffect(() => {
    if (!activeSessionId || (generation?.status !== 'running' && !enrichmentPending) || busy || newSession) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const response = await api.getCharacterChat(activeSessionId);
        if (!active) return;
        setSession(response.session);
        setTurns(response.turns);
        setGeneration(response.generation);
        setEnrichmentPending(response.enrichment_pending);
        setEnrichmentRetryable(response.enrichment_retryable);
        const waiting = pending.current;
        if (waiting?.sessionId === activeSessionId && response.turns.some(turn => turn.request_id === waiting.body.request_id)) {
          pending.current = null;
          setDraft(current => current === waiting.body.message ? '' : current);
          setEditing(undefined);
        }
        setSessions(previous => previous.map(item => item.id === activeSessionId ? { ...item, revision: response.session.revision } : item));
        if (response.generation?.status !== 'running' && !response.enrichment_pending) return;
      } catch { /* Keep polling after a transient disconnect; manual refresh remains available. */ }
      if (active) timer = setTimeout(poll, 3000);
    };
    timer = setTimeout(poll, 1500);
    return () => { active = false; clearTimeout(timer); };
  }, [activeSessionId, generation?.status, enrichmentPending, busy, newSession]);

  return <div className="mx-auto max-w-6xl space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h1 className="text-2xl font-semibold">角色聊天</h1>
      <Button asChild variant="outline"><Link href="/character-library">管理角色</Link></Button>
    </div>
    {error && <p role="alert" className="rounded-lg border border-destructive p-3 text-destructive">{error}</p>}
    <div className="grid gap-5 md:grid-cols-[240px_1fr]">
      <aside className="space-y-3">
        <Label htmlFor="chat-character">角色</Label>
        <select id="chat-character" disabled={busy} className="w-full rounded-md border bg-background p-2" value={characterId} onChange={event => setCharacterId(event.target.value)}>
          <option value="" disabled>选择角色</option>
          {characters.map(character => <option key={character.id} value={character.id}>{character.settings.name}</option>)}
        </select>
        {moreCharacters && <Button variant="outline" disabled={busy} className="w-full" onClick={() => void run(async () => {
          const response = await api.listCharacterLibrary(characterOffset);
          setCharacters(previous => [...new Map([...previous, ...response.characters].map(item => [item.id, item])).values()]);
          setCharacterOffset(previous => previous + response.characters.length); setMoreCharacters(response.characters.length === 100);
        })}>加载更多角色</Button>}
        {!characters.length && <p className="text-sm text-muted-foreground">先在“我的角色”中创建或导入角色。</p>}
        <Button disabled={busy || !characterId} className="w-full" onClick={() => { setNewSession(true); setSource(''); setAcceptChange(false); setSelectedShared([]); }}>新会话</Button>
        <nav aria-label="会话列表" className="space-y-2">
          {sessions.map(item => <button key={item.id} disabled={busy} onClick={() => void run(async () => {
            await open(item.id); setDraft(''); setEditing(undefined); pending.current = null;
          })} className={`w-full rounded-lg border p-3 text-left ${session?.id === item.id && !newSession ? 'bg-muted' : ''}`}>
            <span className="block">{item.title}</span>
            <span className="text-xs text-muted-foreground">{item.mode === 'branch' ? '假想分支' : item.mode === 'continuation' ? '延续关系' : '初识'} · 角色版本 {item.character_revision}</span>
          </button>)}
          {moreSessions && <Button variant="outline" disabled={busy} className="w-full" onClick={() => void run(async () => {
            const response = await api.listCharacterChats(characterId, sessionOffset);
            setSessions(previous => [...new Map([...previous, ...response.sessions].map(item => [item.id, item])).values()]);
            setSessionOffset(previous => previous + response.sessions.length); setMoreSessions(response.sessions.length === 100);
          })}>加载更多会话</Button>}
        </nav>
        {sessions.length > 0 && <details className="rounded-lg border p-3">
          <summary className="cursor-pointer text-sm">彻底删除会话</summary>
          <p className="my-2 text-xs text-muted-foreground">删除所选会话的消息、关系、记忆和共享入口。你可以先查看关联分支和副本，再决定删除范围。</p>
          {sessions.map(item => <label key={item.id} className="my-2 flex items-start gap-2 text-sm">
            <input type="checkbox" disabled={busy} checked={deleteSelection.includes(item.id)} onChange={event => {
              clearDeletionPreview();
              const checked = event.target.checked;
              setDeleteSelection(previous => checked ? [...previous, item.id] : previous.filter(id => id !== item.id));
            }} />
            <span>{item.title} · {item.mode === 'branch' ? '分支' : item.mode === 'continuation' ? '延续' : '初识'}</span>
          </label>)}
          {moreSessions && <p className="text-xs">更多会话可通过上方“加载更多会话”展开。</p>}
          <Button variant="outline" disabled={busy || !deleteSelection.length || deleteSelection.length > 100} onClick={() => void run(reviewDeletion)}>
            预览删除范围（{deleteSelection.length}）
          </Button>
          {deleteSelection.length > 100 && <p className="text-xs">每次最多删除 100 个会话。</p>}
          {deletionPreview && <section aria-label="删除范围" className="mt-3 space-y-2 rounded border p-3 text-sm">
            <p>已选：{deletionPreview.selected.map(item => item.title).join('、')}</p>
            {deletionPreview.related.length > 0 ? <fieldset disabled={busy} className="space-y-1">
              <legend>关联会话，可选择一并删除</legend>
              {deletionPreview.related.map(item => <label key={item.id} className="flex items-start gap-2">
                <input type="checkbox" checked={relatedDeletionIds.includes(item.id)} onChange={event => {
                  const checked = event.target.checked;
                  setRelatedDeletionIds(previous => checked ? [...previous, item.id] : previous.filter(id => id !== item.id));
                }} />
                <span>{item.title} · {item.mode === 'branch' ? '分支' : item.mode === 'continuation' ? '延续' : '记忆副本'}{item.character_id !== characterId && ' · 其他角色'}</span>
              </label>)}
            </fieldset> : <p>未发现关联分支或会话副本。</p>}
            <p>只删除本次勾选的会话，包括其中已撤回和替换的内容。未选的副本仍保留。</p>
            {deletionPreview.selected.length + relatedDeletionIds.length > 100 && <p>每次最多删除 100 个会话。</p>}
            <div className="flex flex-wrap gap-2">
              <Button variant="destructive" disabled={busy || deletionPreview.selected.length + relatedDeletionIds.length > 100} onClick={() => void run(confirmDeletion)}>
                永久删除 {deletionPreview.selected.length + relatedDeletionIds.length} 个会话
              </Button>
              <Button variant="ghost" disabled={busy} onClick={clearDeletionPreview}>取消</Button>
            </div>
          </section>}
        </details>}
      </aside>
      <section className="min-w-0 space-y-4">
        {newSession ? <form className="space-y-4 rounded-xl border p-5" onSubmit={event => { event.preventDefault(); void run(async () => {
          if (!selectedCharacter) return;
          if (sourceSession && !preview) return;
          const created = await api.createCharacterChat({ character_id: characterId, title,
            ...(localOnly ? { network_policy: 'local_only' as const } : {}),
            expected_character_revision: preview?.character_revision ?? selectedCharacter.revision,
            ...(!sourceSession ? { shared_memory_ids: selectedShared } : {}),
            ...(sourceSession && preview ? { source_session_id: sourceSession.id, source_revision: preview.source_revision, accept_settings_change: acceptChange,
              excluded_memory_ids: excludedMemories, reset_relationship: resetRelationship, inherit_history: inheritHistory } : {}) });
          await refreshSessions(characterId); await open(created.id); setDraft(''); setEditing(undefined); pending.current = null;
        }); }}>
          <h2 className="text-lg font-medium">开始新的对话</h2>
          <label className="flex items-center gap-2"><input type="checkbox" checked={localOnly} disabled={busy} onChange={event => setLocalOnly(event.target.checked)} />仅使用本地模型</label>
          <p className="text-sm text-muted-foreground">开启后，聊天内容只交给管理员配置的本地模型；云端记忆、关系和摘要处理会跳过。继承仅本地会话或其共享记忆时自动保留此限制。</p>
          <Label htmlFor="chat-title">会话名称</Label><Input id="chat-title" required maxLength={100} value={title} onChange={event => setTitle(event.target.value)} disabled={busy} />
          <Label htmlFor="chat-source">关系起点</Label>
          <select id="chat-source" className="w-full rounded-md border bg-background p-2" disabled={busy} value={source} onChange={event => { setSource(event.target.value); setAcceptChange(false); }}>
            <option value="">从初识开始</option>
            {sessions.filter(item => item.mode !== 'branch').map(item => <option key={item.id} value={item.id}>延续「{item.title}」</option>)}
          </select>
          {!source && <details className="space-y-2 rounded-lg border p-3">
            <summary className="cursor-pointer">选择带入共享记忆（默认不带入）</summary>
            <p className="text-sm text-muted-foreground">仅供你自己的角色会话使用。取消共享只阻止之后带入，已经复制到其他会话的记忆需在对应会话中删除。</p>
            {sharedMemories.map(item => <div key={item.id} className="flex items-start justify-between gap-2 py-2">
              <label className="flex gap-2 text-sm"><input type="checkbox" disabled={busy} checked={selectedShared.includes(item.id)} onChange={event => setSelectedShared(previous => event.target.checked ? [...previous, item.id] : previous.filter(id => id !== item.id))} />{item.memory.content}</label>
              <Button type="button" size="sm" variant="ghost" disabled={busy} onClick={() => void run(async () => { await api.unshareChatMemory(item.id); await refreshShared(); })}>取消共享</Button>
            </div>)}
            {!sharedMemories.length && <p className="text-sm">尚无共享记忆，可在已有会话的记忆面板中选择共享。</p>}
            <Button type="button" variant="outline" disabled={busy} onClick={() => void run(refreshShared)}>刷新共享列表</Button>
            {moreShared && <Button type="button" variant="ghost" disabled={busy} onClick={() => void run(async () => {
              const result = await api.listSharedChatMemories(sharedMemories.length);
              setSharedMemories(previous => [...new Map([...previous, ...result.memories].map(item => [item.id, item])).values()]); setMoreShared(result.memories.length === 100);
            })}>加载更多共享记忆</Button>}
          </details>}
          {source && <Button type="button" variant="outline" disabled={busy} onClick={() => void run(async () => {
            setAcceptChange(false); setPreview(undefined); setExcludedMemories([]); setResetRelationship(false); setInheritHistory(true);
            setPreview(await api.getContinuationPreview(source));
          })}>刷新继承预览</Button>}
          {preview && <details className="rounded-lg border p-3" open={settingsChanged}>
            <summary className="cursor-pointer">查看设定差异与将继承的记忆</summary>
            {Object.entries(preview.changes).map(([field, change]) => <div key={field} className="my-3 space-y-1">
              <p className="font-medium">{({ name: '名字', persona: '人物设定', greeting: '开场白', scenario: '初始场景', relationships: '人物关系', examples: '示范对话', reference_text: '参考资料' } as Record<string, string>)[field] || field}</p>
              <pre className="whitespace-pre-wrap break-words text-sm">原设定：{typeof change.previous === 'string' ? change.previous : JSON.stringify(change.previous, null, 2)}</pre>
              <pre className="whitespace-pre-wrap break-words text-sm">新设定：{typeof change.next === 'string' ? change.next : JSON.stringify(change.next, null, 2)}</pre>
            </div>)}
            {!Object.keys(preview.changes).length && <p className="my-2 text-sm">设定内容没有变化。</p>}
            <p className="text-sm">关系记录：{preview.relationship.summary || '暂无关系摘要'}</p>
            <label className="my-2 flex items-start gap-2 text-sm"><input type="checkbox" disabled={busy} checked={resetRelationship} onChange={event => setResetRelationship(event.target.checked)} />重置关系状态</label>
            <p className="my-2 text-sm">选择要带入的记忆，可取消与新设定冲突的内容：</p>
            <ul className="space-y-2 text-sm">{preview.memories.map(memory => <li key={memory.id}>
              <label className="flex items-start gap-2"><input type="checkbox" disabled={busy} checked={!excludedMemories.includes(memory.id)} onChange={event => setExcludedMemories(previous => event.target.checked ? previous.filter(id => id !== memory.id) : [...previous, memory.id])} />{memory.content}</label>
            </li>)}</ul>
            <label className="my-2 flex items-start gap-2 text-sm"><input type="checkbox" disabled={busy} checked={inheritHistory} onChange={event => setInheritHistory(event.target.checked)} />带入原始对话文字</label>
            <p className="text-sm text-muted-foreground">原始文字可能仍提及被取消的记忆或旧关系。如不希望新会话读取这些内容，请取消带入原始对话。调整仅影响新会话。</p>
          </details>}
          {settingsChanged && <label className="flex items-start gap-2 rounded-lg border p-3 text-sm">
            <input type="checkbox" checked={acceptChange} disabled={busy} onChange={event => setAcceptChange(event.target.checked)} />
            角色设定从版本 {preview?.previous_character_revision} 更新为 {preview?.character_revision}。确认使用所展示的新设定，并按上方选择继承关系、记忆与对话。
          </label>}
          <Button disabled={busy || !characterId || (Boolean(source) && !preview) || (settingsChanged && !acceptChange)}>开始聊天</Button>
        </form> : session && <>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-medium">{session.title}</h2>
            <p className="text-sm text-muted-foreground">{session.network_policy === 'local_only' ? '仅使用本地模型 · 未配置本地辅助模型时，记忆、关系和摘要需手动管理' : '允许使用已配置的云端模型'}</p>
            <Button disabled={busy} variant="outline" onClick={() => void run(async () => { await open(session.id); await refreshSessions(characterId); })}>刷新会话</Button>
          </div>
          {session.hypothesis && <p className="rounded-lg bg-muted p-3">剧情假设：{session.hypothesis}</p>}
          {watchedRequest && <GenerationDraft key={`${session.id}:${watchedRequest}`} sessionId={session.id} requestId={watchedRequest} />}
          {enrichmentPending && <p role="status" className="rounded-lg border p-3 text-sm">回复已保存，正在整理记忆、关系和对话摘要。你可以先输入下一条，整理结束后即可发送。</p>}
          {enrichmentRetryable && <div role="status" className="space-y-2 rounded-lg border p-3 text-sm">
            <p>本轮后台整理失败，回复已保存。可以重试整理，也可以继续聊天。</p>
            <Button variant="outline" disabled={busy} onClick={() => run(async () => {
              await api.retryChatEnrichment(session.id, session.revision);
              await open(session.id);
            })}>重试后台整理</Button>
          </div>}
          {generation && <div role="status" className="space-y-2 rounded-lg border p-3">
            <p>{generation.status === 'running' ? '这条消息仍在生成，完成后会自动显示。关闭页面不会取消生成。' : '上次生成未完成，可恢复原消息后重试。'}</p>
            {generation.body && <p className="whitespace-pre-wrap text-sm">{generation.body.message}</p>}
            {generation.status !== 'running' && generation.body && <Button disabled={busy} variant="outline" onClick={() => {
              if (draft && draft !== generation.body!.message && !window.confirm('用上次未完成的消息替换当前草稿？')) return;
              const body = generation.body!;
              pending.current = { sessionId: session.id, body };
              setDraft(body.message); setEditing(body.replace_turn_id || undefined);
            }}>恢复到输入框</Button>}
          </div>}
          {!!session.state.summaries?.length && <details className="rounded-lg border p-4">
            <summary className="cursor-pointer">对话摘要 · {session.state.summaries.length} 段</summary>
            <p className="my-2 text-xs text-muted-foreground">摘要用于衔接较早对话，是模型概括，可展开原话核对。</p>
            {session.state.summaries.map((segment, index) => <details key={`${segment.source_request_id}-${index}`} className="my-2 rounded border p-3">
              <summary className="cursor-pointer whitespace-pre-wrap text-sm">{segment.summary}</summary>
              {segment.evidence.map((item, evidenceIndex) => <blockquote key={evidenceIndex} className="my-2 border-l-2 pl-3 text-sm">
                {item.role === 'user' ? '你' : '角色'}：{item.quote}
              </blockquote>)}
            </details>)}
          </details>}
          <details className="rounded-lg border p-4">
            <summary className="cursor-pointer">关系记录 · {{ stranger: '初识', acquaintance: '认识', familiar: '熟悉', close: '亲近' }[session.state.relationship.stage]}</summary>
            {session.state.relationship_processing === 'failed' && <p role="status" className="my-2 text-sm">本轮自动关系更新失败，已保留原关系和对话。</p>}
            {session.state.relationship_processing === 'disabled' && <p className="my-2 text-sm text-muted-foreground">服务端尚未启用自动关系更新，当前可手动调整。</p>}
            {session.state.relationship.evidence?.map((item, index) => <blockquote key={index} className="my-2 border-l-2 pl-3 text-sm">{item.role === 'user' ? '你' : '角色'}：{item.quote}</blockquote>)}
            {session.state.relationship_proposal && <div className="my-3 space-y-2 rounded-md border p-3">
              <p className="font-medium">关系变化建议 · 待你确认</p>
              <p className="text-sm">已保留你手动设定的关系。建议调整为：{{ stranger: '初识', acquaintance: '认识', familiar: '熟悉', close: '亲近' }[session.state.relationship_proposal.stage]}</p>
              <p className="whitespace-pre-wrap text-sm">{session.state.relationship_proposal.summary}</p>
              {session.state.relationship_proposal.evidence.map((item, index) => <blockquote key={index} className="border-l-2 pl-3 text-sm">{item.role === 'user' ? '你' : '角色'}：{item.quote}</blockquote>)}
              <div className="flex gap-2">{[true, false].map(accept => <Button key={String(accept)} disabled={busy} variant={accept ? 'default' : 'outline'} onClick={() => void run(async () => {
                await api.resolveChatRelationshipProposal(session.id, session.revision, session.state.relationship_proposal!.id, accept);
                await open(session.id); await refreshSessions(characterId);
              })}>{accept ? '接受建议' : '保留原关系'}</Button>)}</div>
            </div>}
            <p className="my-3 text-sm text-muted-foreground">记录目前的相处状态与共同经历。调整绑定当前对话位置，撤回到此前会恢复当时的关系；已有分支保持独立。人物仍保有自己的立场。</p>
            <form className="space-y-2" onSubmit={event => { event.preventDefault(); void run(async () => {
              await api.updateChatRelationship(session.id, session.revision, relationshipStage, relationshipSummary);
              await open(session.id); await refreshSessions(characterId);
            }); }}>
              <Label htmlFor="relationship-stage">熟悉程度</Label>
              <select id="relationship-stage" className="w-full rounded-md border bg-background p-2" disabled={busy} value={relationshipStage} onChange={event => setRelationshipStage(event.target.value as typeof relationshipStage)}>
                <option value="stranger">初识</option><option value="acquaintance">认识</option><option value="familiar">熟悉</option><option value="close">亲近</option>
              </select>
              <Label htmlFor="relationship-summary">关系摘要</Label>
              <textarea id="relationship-summary" disabled={busy} maxLength={2000} rows={3} className="w-full rounded-md border bg-background p-2" value={relationshipSummary} onChange={event => setRelationshipSummary(event.target.value)} />
              <Button disabled={busy}>保存关系记录</Button>
            </form>
          </details>
          <details className="rounded-lg border p-4">
            <summary className="cursor-pointer">会话记忆（{session.state.memories.length}）</summary>
            {session.state.memory_processing === 'failed' && <p role="status" className="my-2 text-sm">本轮自动记忆提取失败，对话已保存。可以手动补充记忆。</p>}
            {session.state.memory_processing === 'disabled' && <p className="my-2 text-sm text-muted-foreground">服务端尚未启用自动记忆，当前可手动管理。</p>}
            <p className="my-3 text-sm text-muted-foreground">修改仅作用于当前会话，已有分支保持独立。删除记忆不会删除原始聊天文字。手动添加的记忆绑定当前对话位置，撤回到此前会移除它。</p>
            <ul className="space-y-3">
              {session.state.memories.map(memory => <li key={memory.id} className="rounded-md border p-3">
                <p className="whitespace-pre-wrap">{memory.content}</p>
                <p className="text-xs text-muted-foreground">{memory.status === 'pending' ? '待确认' : '已确认'} · {memory.origin === 'user' ? '手动记录' : '对话提取'}</p>
                {memory.source_turn_id && <Button variant="link" size="sm" disabled={busy} onClick={() => void run(async () => {
                  const local = [...turns, ...session.baseline.history].find(turn => turn.id === memory.source_turn_id);
                  if (local) {
                    const element = document.getElementById(`turn-${local.id}`);
                    const details = element?.closest('details');
                    if (details) details.open = true;
                    element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    element?.focus({ preventScroll: true });
                    return;
                  }
                  const response = await api.getCharacterChat(memory.source_session_id);
                  const original = [...response.turns, ...response.session.baseline.history].find(turn => turn.id === memory.source_turn_id);
                  if (!original) throw new Error('来源消息已撤回或不再可用；可查看此记忆保留的来源原文。');
                  setSourceExcerpt({ message: original.message, reply: original.reply, title: response.session.title });
                })}>查看来源消息</Button>}
                {memory.evidence && <details className="text-sm"><summary className="cursor-pointer">查看来源原文</summary><p className="whitespace-pre-wrap">{memory.evidence}</p></details>}
                {memory.status === 'pending' && <Button variant="outline" size="sm" disabled={busy} onClick={() => {
                  if (!window.confirm('确认这条记忆？同一事项的旧记忆会由本条替换。')) return;
                  void run(async () => { await api.saveCharacterChatMemory(session.id, session.revision, memory.content.replace(/^待确认：/, ''), memory.id); await open(session.id); await refreshSessions(characterId); });
                }}>确认记忆</Button>}
                <Button variant="ghost" size="sm" disabled={busy} onClick={() => { setMemoryEditing(memory.id); setMemoryDraft(memory.content); }}>纠正</Button>
                {session.mode !== 'branch' && memory.status === 'confirmed' && <Button variant="ghost" size="sm" disabled={busy} onClick={() => {
                  if (!window.confirm('将这条记忆加入账号内共享列表，供其他角色的新会话选择带入？')) return;
                  void run(async () => { await api.shareChatMemory(session.id, memory.id, session.revision); await open(session.id); await refreshSessions(characterId); await refreshShared(); });
                }}>共享给其他角色</Button>}
                <Button variant="ghost" size="sm" disabled={busy} onClick={() => {
                  if (!window.confirm('从当前会话及其状态快照中删除这条记忆？已有分支不受影响。')) return;
                  void run(async () => { await api.deleteCharacterChatMemory(session.id, session.revision, memory.id); await open(session.id); await refreshSessions(characterId); });
                }}>删除</Button>
              </li>)}
            </ul>
            <form className="mt-3 space-y-2" onSubmit={event => { event.preventDefault(); void run(async () => {
              await api.saveCharacterChatMemory(session.id, session.revision, memoryDraft, memoryEditing);
              await open(session.id); await refreshSessions(characterId);
            }); }}>
              <Label htmlFor="memory-content">{memoryEditing ? '纠正记忆内容' : '添加会话记忆'}</Label>
              <textarea id="memory-content" required maxLength={2000} disabled={busy} rows={2} className="w-full rounded-md border bg-background p-2" value={memoryDraft} onChange={event => setMemoryDraft(event.target.value)} />
              <Button disabled={busy || !memoryDraft.trim()}>保存记忆</Button>
              {memoryEditing && <Button type="button" variant="ghost" disabled={busy} onClick={() => { setMemoryEditing(undefined); setMemoryDraft(''); }}>取消</Button>}
            </form>
          </details>
          {sourceExcerpt && <aside className="space-y-2 rounded-lg border p-4" aria-label="记忆来源对话">
            <p className="font-medium">来源会话：{sourceExcerpt.title}</p>
            <p className="whitespace-pre-wrap">你：{sourceExcerpt.message}</p>
            <p className="whitespace-pre-wrap">角色：{sourceExcerpt.reply}</p>
            <Button variant="ghost" onClick={() => setSourceExcerpt(undefined)}>关闭来源预览</Button>
          </aside>}
          <div className="space-y-4" aria-label="聊天记录">
            <p className="whitespace-pre-wrap rounded-lg bg-muted p-4">{session.character.settings.greeting}</p>
            {session.baseline.history.length > 0 && <details className="rounded-lg border p-3"><summary className="cursor-pointer">继承的对话经历</summary>
              {session.baseline.history.map((turn, index) => <div key={`${turn.id}-${index}`} id={`turn-${turn.id}`} tabIndex={-1} className="mt-3 space-y-2 whitespace-pre-wrap text-sm focus:outline focus:outline-2 focus:outline-ring">
                <p>你：{turn.message}</p><p>{session.character.settings.name}：{turn.reply}</p>
                <AnswerSources evidence={turn.evidence} />
                {turn.character_revision !== undefined && <Button variant="ghost" size="sm" disabled={busy} onClick={() => { setBranchTurn(turn.id); setHypothesis(''); }}>从这里创建分支</Button>}
              </div>)}
            </details>}
            {turns.map(turn => <article key={turn.id} id={`turn-${turn.id}`} tabIndex={-1} className="space-y-3 rounded-xl border p-4 focus:outline focus:outline-2 focus:outline-ring">
              <p className="whitespace-pre-wrap"><span className="font-medium">你</span><br />{turn.message}</p>
              <p className="whitespace-pre-wrap"><span className="font-medium">{session.character.settings.name}</span><br />{turn.reply}</p>
              <AnswerSources evidence={turn.evidence} />
              <div className="flex flex-wrap gap-2">
                <Button variant="ghost" size="sm" disabled={busy} onClick={() => { setEditing(turn.id); setDraft(turn.message); pending.current = null; }}>编辑</Button>
                <Button variant="ghost" size="sm" disabled={busy} onClick={() => {
                  if (!window.confirm('撤回这条消息和后续对话？相关关系和记忆将回到此前状态。')) return;
                  void run(async () => { await api.retractCharacterChat(session.id, turn.id, session.revision); await open(session.id); await refreshSessions(characterId); setEditing(undefined); setDraft(''); pending.current = null; });
                }}>从此处撤回</Button>
                <Button variant="ghost" size="sm" disabled={busy} onClick={() => { setBranchTurn(turn.id); setHypothesis(''); }}>创建分支</Button>
              </div>
            </article>)}
          </div>
          {branchTurn && <form className="space-y-3 rounded-lg border p-4" onSubmit={event => { event.preventDefault(); void run(async () => {
            const branch = await api.createCharacterChat({ character_id: characterId, title: `${session.title} · 分支`.slice(0, 100),
              source_session_id: session.id, source_revision: session.revision, source_turn_id: branchTurn, hypothesis });
            await refreshSessions(characterId); await open(branch.id); setDraft(''); setEditing(undefined); pending.current = null;
          }); }}>
            <Label htmlFor="chat-hypothesis">这个分支有什么不同？</Label>
            <textarea id="chat-hypothesis" required maxLength={8000} disabled={busy} rows={3} className="w-full rounded-md border bg-background p-3" value={hypothesis} onChange={event => setHypothesis(event.target.value)} />
            <div className="flex gap-2"><Button disabled={busy}>保存并进入分支</Button><Button type="button" variant="ghost" disabled={busy} onClick={() => setBranchTurn(undefined)}>取消</Button></div>
          </form>}
          <form className="space-y-3 rounded-xl border p-4" onSubmit={event => { event.preventDefault(); void run(send); }}>
            <Label htmlFor="chat-message">{editing ? '编辑消息并替换后续走向' : '发送消息'}</Label>
            {editing && <p className="text-sm text-muted-foreground">生成成功后，原消息及后续对话会退出当前走向。</p>}
            <textarea id="chat-message" required maxLength={8000} disabled={busy} rows={4} className="w-full rounded-md border bg-background p-3" value={draft} onChange={event => setDraft(event.target.value)} />
            <div className="flex gap-2"><Button disabled={busy || generation?.status === 'running' || (enrichmentPending && !editing) || !draft.trim()}>{busy ? '正在处理…' : editing ? '保存并继续' : '发送'}</Button>
              {editing && <Button type="button" variant="ghost" disabled={busy} onClick={() => { setEditing(undefined); setDraft(''); pending.current = null; }}>取消编辑</Button>}
            </div>
          </form>
        </>}
      </section>
    </div>
  </div>;
}
