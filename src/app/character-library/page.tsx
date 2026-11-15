'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { api } from '@/lib/api';
import { CharacterSourceManager } from '@/components/chat/CharacterSourceManager';
import type { CharacterLibraryRecord, CharacterPackage, CharacterSettings, CharacterSourceUpdate } from '@/lib/api-contracts';

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
  const [sourceUpdate, setSourceUpdate] = useState<CharacterSourceUpdate>();
  const [importPreview, setImportPreview] = useState<CharacterPackage>();
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

  useEffect(() => {
    let active = true;
    setSourceUpdate(undefined);
    if (current?.source_character_id) {
      api.getCharacterSourceUpdate(current.id).then(update => { if (active) setSourceUpdate(update); })
        .catch(error => { if (active) setError(String(error)); });
    }
    return () => { active = false; };
  }, [current?.id, current?.revision, current?.source_character_id]);

  function select(character?: CharacterLibraryRecord) {
    setCurrent(character);
    setDraft(character ? { ...character.settings } : { ...emptySettings });
    setExamples(JSON.stringify(character?.settings.examples || [], null, 2));
    setError('');
    setNotice('');
  }

  async function run(operation: () => Promise<void>) {
    setBusy(true);
    setError('');
    setNotice('');
    try { await operation(); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  }

  async function save() {
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
        <Button disabled={busy} onClick={() => select()}>创建角色</Button>
        <Button disabled={busy} variant="outline" onClick={() => fileInput.current?.click()}>导入角色</Button>
        <Button disabled={busy} variant="outline" onClick={() => void run(refresh)}>刷新列表</Button>
      </div>
    </div>
    <input ref={fileInput} type="file" accept="application/json,.json" className="hidden" aria-label="导入角色文件"
      onChange={event => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        void run(async () => {
          if (file.size > 1024 * 1024) throw new Error('角色文件不能超过 1 MB');
          setImportPreview(undefined);
          const preview = await api.previewLibraryCharacterImport(JSON.parse(await file.text()));
          setImportPreview(preview);
        });
      }} />
    <p className="text-sm text-muted-foreground">支持本应用导出的角色 JSON 文件（multipersonal-character-v1），最大 1 MB；暂不支持 PNG 角色卡及其他应用格式。导入前可预览，确认后创建独立角色。</p>
    {importPreview && <section aria-label="角色导入预览" className="space-y-3 rounded-lg border p-4">
      <h2 className="font-medium">准备导入：{importPreview.settings.name}</h2>
      <p className="whitespace-pre-wrap text-sm">{importPreview.settings.persona}</p>
      <p className="whitespace-pre-wrap text-sm">开场白：{importPreview.settings.greeting}</p>
      <details><summary className="cursor-pointer">查看完整设定</summary>
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-sm">{JSON.stringify(importPreview.settings, null, 2)}</pre>
      </details>
      <div className="flex gap-2">
        <Button disabled={busy} onClick={() => void run(async () => {
          const imported = await api.importLibraryCharacter(importPreview);
          select(imported); setImportPreview(undefined); setNotice('角色已导入，可继续编辑或开始试聊。');
          await refresh();
        })}>确认导入为新角色</Button>
        <Button disabled={busy} variant="outline" onClick={() => setImportPreview(undefined)}>取消导入</Button>
      </div>
    </section>}
    {error && <p role="alert" className="rounded-lg border border-destructive p-3 text-destructive">{error}</p>}
    {notice && <p role="status" className="text-sm text-muted-foreground">{notice}</p>}
    {shareToken && <div className="flex flex-wrap items-center gap-3 rounded-lg border p-4">
      <p>收到角色分享，可以复制到自己的角色库。</p>
      <Button disabled={busy} onClick={() => void run(async () => {
        select(await api.copySharedCharacter(shareToken));
        setShareToken('');
        window.history.replaceState(null, '', '/character-library');
        setNotice('已复制为独立角色');
        await refresh();
      })}>复制角色</Button>
    </div>}
    <div className="grid gap-6 md:grid-cols-[220px_1fr]">
      <nav aria-label="角色列表" className="space-y-2">
        {characters.map(character => <button key={character.id} disabled={busy}
          aria-current={current?.id === character.id ? 'true' : undefined}
          className={`block w-full rounded-lg border p-3 text-left ${current?.id === character.id ? 'bg-muted' : ''}`}
          onClick={() => select(character)}>
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
          {current?.source_character_id && <div className="space-y-3 rounded-lg border p-3">
            <p>{sourceUpdate?.available ? `原作者已发布版本 ${sourceUpdate.source_revision}` : '暂无可导入的作者更新'}</p>
            <Button type="button" variant="outline" onClick={() => void run(async () => { setSourceUpdate(await api.getCharacterSourceUpdate(current.id)); })}>检查作者更新</Button>
            {sourceUpdate?.available && <>
              <p className="text-sm text-muted-foreground">导入将使用作者发布的完整设定，替换你的本地修改和未保存内容。已有会话仍使用原设定版本。</p>
              <details><summary className="cursor-pointer">查看与已保存设定的差异</summary>
                {Object.entries(sourceUpdate.changes).map(([field, change]) => <div key={field} className="my-2 space-y-1">
                  <p className="font-medium">{({ name: '名字', persona: '人物设定', greeting: '开场白', scenario: '初始场景', relationships: '人物关系', examples: '示范对话', reference_text: '参考资料' } as Record<string, string>)[field] || field}</p>
                  <pre className="whitespace-pre-wrap break-words text-sm">当前：{typeof change.current === 'string' ? change.current : JSON.stringify(change.current, null, 2)}</pre>
                  <pre className="whitespace-pre-wrap break-words text-sm">更新：{typeof change.incoming === 'string' ? change.incoming : JSON.stringify(change.incoming, null, 2)}</pre>
                </div>)}
              </details>
              <Button type="button" onClick={() => {
                if (!window.confirm('确认导入作者的完整设定并替换当前编辑内容？')) return;
                void run(async () => {
                  if (sourceUpdate.source_revision === null) return;
                  select(await api.importCharacterSourceUpdate(current.id, sourceUpdate.revision, sourceUpdate.source_revision));
                  setNotice('已导入作者更新，已有会话保持原版本'); await refresh();
                });
              }}>导入此版本</Button>
            </>}
          </div>}
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
              <Button type="button" variant="outline" onClick={() => void run(() => publish(true))}>分享已保存版本</Button>
              {current.share_token && <Button type="button" variant="outline" onClick={() => void run(() => publish(false))}>关闭分享</Button>}
            </>}
          </div>
          {shareUrl && <div className="space-y-2"><Label htmlFor="character-share">分享链接 · 版本 {current?.shared_revision}</Label>
            <Input id="character-share" readOnly value={shareUrl} onFocus={event => event.target.select()} /></div>}
        </fieldset>
      </form>
    </div>
    {current && <CharacterSourceManager key={`${current.id}:${current.revision}`} character={current} />}
  </div>;
}
