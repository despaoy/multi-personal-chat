'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { api } from '@/lib/api';
import { createClientId } from '@/lib/client-id';
import { CharacterSharing, SharedCharacterPreview } from '@/components/chat/CharacterSharing';
import { CharacterSourceUpdatePanel } from '@/components/chat/CharacterSourceUpdatePanel';
import { CharacterSourceManager } from '@/components/chat/CharacterSourceManager';
import type { CharacterLibraryRecord, CharacterPackage, CharacterSettings, CharacterOriginalSource } from '@/lib/api-contracts';

// Covers the exported JSON of a fully populated v1 role, including escaped characters.
const CHARACTER_FILE_LIMIT_MIB = 3;

const emptySettings: CharacterSettings = {
  name: '', persona: '', greeting: '', scenario: '', relationships: '', examples: [], reference_text: '',
};

export default function CharacterLibraryPage() {
  return <AuthGuard><AppLayout><CharacterLibrary /></AppLayout></AuthGuard>;
}

function CharacterLibrary() {
  const [characters, setCharacters] = useState<CharacterLibraryRecord[]>([]);
  const [current, setCurrent] = useState<CharacterLibraryRecord>();
  const [draft, setDraft] = useState<CharacterSettings>({ ...emptySettings });
  const [examples, setExamples] = useState('[]');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [shareToken, setShareToken] = useState('');
  const [hasMore, setHasMore] = useState(false);
  const [sourceDraft, setSourceDraft] = useState<{ sources: CharacterOriginalSource[]; baseline: CharacterOriginalSource[] }>({ sources: [], baseline: [] });
  const [sourceEpoch, setSourceEpoch] = useState(0);
  const gate = useRef(false);
  const sourceDirty = JSON.stringify(sourceDraft.sources) !== JSON.stringify(sourceDraft.baseline);
  const settingsDirty = JSON.stringify(draft) !== JSON.stringify(current?.settings || emptySettings) ||
    examples !== JSON.stringify(current?.settings.examples || [], null, 2);
  const dirty = sourceDirty || settingsDirty;

  function acquire() {
    if (gate.current) return false;
    gate.current = true; setBusy(true); return true;
  }
  function release() { gate.current = false; setBusy(false); }
  function discardAllowed() {
    return !dirty || window.confirm('有未保存的角色设定或原作资料。确定明确丢弃这些编辑并继续？取消将保留编辑，可先保存后重新预览。');
  }
  function beginUpdate() {
    if (gate.current || !discardAllowed()) return false;
    if (dirty) {
      setDraft({ ...(current?.settings || emptySettings) });
      setExamples(JSON.stringify(current?.settings.examples || [], null, 2));
      setSourceDraft({ sources: [], baseline: [] });
    }
    return acquire();
  }
  useEffect(() => {
    const unload = (event: BeforeUnloadEvent) => {
      if (dirty || gate.current) { event.preventDefault(); event.returnValue = ''; }
    };
    const navigate = (event: MouseEvent) => {
      if (!(event.target instanceof Element) || !event.target.closest('a[href]')) return;
      if (gate.current || (dirty && !window.confirm('离开会丢弃未保存编辑，确定离开？'))) {
        event.preventDefault(); event.stopPropagation();
      }
    };
    window.addEventListener('beforeunload', unload);
    document.addEventListener('click', navigate, true);
    return () => { window.removeEventListener('beforeunload', unload); document.removeEventListener('click', navigate, true); };
  }, [dirty]);
  const [importPreview, setImportPreview] = useState<CharacterPackage & { request_id: string }>();
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const response = await api.listCharacterLibrary();
    setCharacters(response.characters);
    setHasMore(response.characters.length === 100);
  }, []);

  useEffect(() => {
    refresh().catch(error => setError(String(error)));
    setShareToken(new URLSearchParams(window.location.search).get('share') || '');
  }, [refresh]);

  function select(character?: CharacterLibraryRecord) {
    setCurrent(character);
    setSourceDraft({ sources: [], baseline: [] });
    setDraft(character ? { ...character.settings } : { ...emptySettings });
    setExamples(JSON.stringify(character?.settings.examples || [], null, 2));
    setError('');
    setNotice('');
  }

  function choose(character?: CharacterLibraryRecord) {
    if (!gate.current && discardAllowed()) select(character);
  }

  async function run(operation: () => Promise<void>) {
    if (!acquire()) return;
    setError('');
    setNotice('');
    try { await operation(); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { release(); }
  }

  async function save() {
    if (sourceDirty) throw new Error('请先保存或明确丢弃原作资料草稿，再保存角色设定，避免切换版本丢失资料编辑。');
    const parsed: unknown = JSON.parse(examples);
    if (!Array.isArray(parsed) || !parsed.every(item => item &&
      (item.role === 'user' || item.role === 'assistant') && typeof item.content === 'string')) {
      throw new Error('示范对话必须是包含 role 和 content 的数组');
    }
    const saved = await api.saveLibraryCharacter({ ...draft, examples: parsed }, current);
    select(saved);
    setNotice('角色设定已保存');
    await refresh();
  }

  async function publish(enabled: boolean) {
    if (!current) return;
    const result = await api.publishLibraryCharacter(current, enabled);
    setCurrent({ ...current, ...result });
    setNotice(enabled ? '已分享当前保存版本' : '分享已关闭');
    await refresh();
  }

  async function exportCharacter() {
    if (!current) return;
    const value = await api.exportLibraryCharacter(current.id);
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `character-${current.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const shareUrl = current?.share_token && typeof window !== 'undefined'
    ? `${window.location.origin}/character-library?share=${encodeURIComponent(current.share_token)}` : '';

  return <div className="mx-auto max-w-5xl space-y-6">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h1 className="text-2xl font-semibold">我的角色</h1>
      <div className="flex gap-2">
        <Button disabled={busy} onClick={() => choose()}>创建角色</Button>
        <Button disabled={busy} variant="outline" onClick={() => void run(async () => {
          if (!discardAllowed()) return;
          const preset = await api.installLibraryPreset('kamimaho-kisaki');
          select(preset);
          setNotice('已打开月社妃预设。可在资料版本面板查看出处和索引状态。');
          await refresh();
        })}>添加预设 · 月社妃</Button>
        <Button disabled={busy} variant="outline" onClick={() => fileInput.current?.click()}>导入角色</Button>
        <Button disabled={busy} variant="outline" onClick={() => void run(async () => {
          await refresh();
          setSourceEpoch(value => value + 1);
        })}>刷新列表</Button>
      </div>
    </div>
    <input ref={fileInput} type="file" accept="application/json,.json" className="hidden" aria-label="导入角色文件"
      onChange={event => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        void run(async () => {
          setImportPreview(undefined);
          if (file.size > CHARACTER_FILE_LIMIT_MIB * 1024 * 1024) {
            throw new Error(`角色文件不能超过 ${CHARACTER_FILE_LIMIT_MIB} MiB`);
          }
          const contents = await file.text();
          let decoded: unknown;
          try { decoded = JSON.parse(contents); }
          catch { throw new Error('角色文件不是有效的 JSON。'); }
          const preview = await api.previewLibraryCharacterImport(decoded);
          setImportPreview({ ...preview, request_id: createClientId() });
        });
      }} />
    <p className="text-sm text-muted-foreground">支持本应用导出的角色 JSON 文件（multipersonal-character-v1），最大 {CHARACTER_FILE_LIMIT_MIB} MiB。导入前可预览，确认后创建独立角色。</p>
    {importPreview && <section aria-label="角色导入预览" className="space-y-3 rounded-lg border p-4">
      <h2 className="font-medium">准备导入：{importPreview.settings.name}</h2>
      <p className="whitespace-pre-wrap text-sm">{importPreview.settings.persona}</p>
      <p className="whitespace-pre-wrap text-sm">开场白：{importPreview.settings.greeting}</p>
      <details><summary className="cursor-pointer">查看完整设定</summary>
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-sm">{JSON.stringify(importPreview.settings, null, 2)}</pre>
      </details>
      <div className="flex gap-2">
        <Button disabled={busy} onClick={() => void run(async () => {
          if (!discardAllowed()) return;
          const imported = await api.importLibraryCharacter(importPreview);
          select(imported); setImportPreview(undefined); setNotice('角色已导入，可继续编辑或开始试聊。');
          await refresh();
        })}>确认导入为新角色</Button>
        <Button disabled={busy} variant="outline" onClick={() => setImportPreview(undefined)}>取消导入</Button>
      </div>
    </section>}
    {error && <p role="alert" className="rounded-lg border border-destructive p-3 text-destructive">{error}</p>}
    {notice && <p role="status" className="text-sm text-muted-foreground">{notice}</p>}
    {shareToken && <SharedCharacterPreview key={shareToken} token={shareToken} disabled={busy} acquire={() => !gate.current && discardAllowed() && acquire()} release={release} onCopied={async character => {
      select(character); setShareToken('');
      window.history.replaceState(null, '', '/character-library');
      setNotice('已复制为独立角色，资料处理状态见下方。'); await refresh();
    }} />}
    {current?.source_character_id && <CharacterSourceUpdatePanel key={`update:${current.id}:${current.revision}:${sourceEpoch}`}
      character={current} disabled={busy} begin={beginUpdate} end={release} onImported={async saved => {
        select(saved); setNotice('已导入最终预览中的更新，已有会话保持原版本。'); await refresh();
      }} />}
    <div className="grid gap-6 md:grid-cols-[220px_1fr]">
      <nav aria-label="角色列表" className="space-y-2">
        {characters.map(character => <button key={character.id} disabled={busy}
          aria-current={current?.id === character.id ? 'true' : undefined}
          className={`block w-full rounded-lg border p-3 text-left ${current?.id === character.id ? 'bg-muted' : ''}`}
          onClick={() => choose(character)}>
          <span className="block font-medium">{character.settings.name}</span>
          <span className="text-sm text-muted-foreground">版本 {character.revision}</span>
        </button>)}
        {characters.length === 0 && <p className="text-sm text-muted-foreground">创建或导入你的第一个角色。</p>}
        {hasMore && <Button disabled={busy} variant="outline" onClick={() => void run(async () => {
          const response = await api.listCharacterLibrary(characters.length);
          setCharacters(previous => [...new Map([...previous, ...response.characters].map(item => [item.id, item])).values()]);
          setHasMore(response.characters.length === 100);
        })}>加载更多</Button>}
      </nav>
      <form className="space-y-4 rounded-xl border p-5" onSubmit={event => { event.preventDefault(); void run(save); }}>
        <fieldset disabled={busy} className="space-y-4">
          <div className="space-y-2"><Label htmlFor="character-name">名字</Label>
            <Input id="character-name" required maxLength={100} value={draft.name} onChange={event => setDraft({ ...draft, name: event.target.value })} /></div>
          {([
            ['persona', '人物设定', 16000, true], ['greeting', '开场白', 8000, true],
            ['scenario', '初始场景', 8000, false], ['relationships', '人物关系', 8000, false],
            ['reference_text', '参考资料', 40000, false],
          ] as const).map(([key, label, limit, required]) => <div className="space-y-2" key={key}>
            <Label htmlFor={`character-${key}`}>{label}</Label>
            <textarea id={`character-${key}`} required={required} maxLength={limit} rows={key === 'persona' ? 5 : 3}
              className="w-full rounded-md border bg-background p-3 text-sm" value={draft[key]}
              onChange={event => setDraft({ ...draft, [key]: event.target.value })} />
          </div>)}
          <details><summary className="cursor-pointer text-sm">示范对话</summary>
            <Label htmlFor="character-examples" className="my-2 block">JSON 数组，每项包含 role（user 或 assistant）和 content</Label>
            <textarea id="character-examples" rows={6} className="w-full rounded-md border bg-background p-3 font-mono text-sm"
              value={examples} onChange={event => setExamples(event.target.value)} />
          </details>
          <div className="flex flex-wrap gap-2">
            <Button type="submit">{busy ? '处理中…' : '保存角色'}</Button>
            {current && <>
              <Button type="button" variant="outline" asChild><Link href={`/chat?character=${encodeURIComponent(current.id)}`}>开始聊天</Link></Button>
              <Button type="button" variant="outline" onClick={() => void run(exportCharacter)}>导出角色</Button>
              {current.share_token && <Button type="button" variant="outline" onClick={() => void run(() => publish(false))}>关闭分享</Button>}
            </>}
          </div>
          {shareUrl && <div className="space-y-2"><Label htmlFor="character-share">分享链接 · 版本 {current?.shared_revision}</Label>
            <Input id="character-share" readOnly value={shareUrl} onFocus={event => event.target.select()} /></div>}
        </fieldset>
      </form>
    </div>
    {current && <CharacterSharing disabled={busy} acquire={acquire} release={release} key={`sharing:${current.id}:${current.revision}`} character={current} onPublished={async () => {
      setCurrent(await api.getLibraryCharacter(current.id)); await refresh();
    }} />}
    {current && <CharacterSourceManager key={`${current.id}:${current.revision}`} character={current} disabled={busy}
      draft={sourceDraft} onDraftChange={setSourceDraft} acquire={acquire} release={release}
      onSaved={() => setSourceEpoch(value => value + 1)} />}
  </div>;
}
