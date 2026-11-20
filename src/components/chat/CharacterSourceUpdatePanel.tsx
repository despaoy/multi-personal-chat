'use client';

import { useEffect, useRef, useState } from 'react';
import { api, ApiRequestError } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { SourcePreview } from './CharacterSharing';
import type { CharacterLibraryRecord, CharacterSourceUpdate, FinalSourceUpdatePreview, SourceChoices } from '@/lib/api-contracts';

const actions: Record<string, string> = {
  keep_local_addition: '保留本地新增', keep_local_modification: '保留本地修改', keep_local_deletion: '保留本地删除',
  take_author_addition: '采用作者新增', take_author_modification: '采用作者修改', take_author_deletion: '采用作者删除',
  unchanged: '保持原样', both_deleted: '双方均已删除', same_modification: '双方修改一致',
};
const reasons: Record<string, string> = {
  baseline_unavailable: '共同基线不可用，不能安全判断原作变化。请保留现有资料，等待明确处理。',
  incoming_publication_unavailable: '作者当前没有可读取的发布，可能已关闭分享或撤销资料。',
  invalid_source_identity_or_content: '资料身份或内容无效，不能安全导入。',
  merged_source_capacity_exceeded: '合并后的资料超过容量限制，不能导入。',
  source_identity_collision_requires_explicit_handling: '独立新增资料的 ID 碰撞，不能通过选边解决。',
};
const kinds: Record<string, string> = {
  both_modified: '双方都修改了此篇', local_delete_author_modify: '本地已删除，作者修改了此篇',
  local_modify_author_delete: '本地修改了此篇，作者已删除', addition_id_collision: '独立新增资料 ID 碰撞，不能选边导入',
};

export function CharacterSourceUpdatePanel({ character, disabled, begin, end, onImported }: {
  character: CharacterLibraryRecord; disabled: boolean; begin: () => boolean; end: () => void;
  onImported: (character: CharacterLibraryRecord) => Promise<void>;
}) {
  const [plan, setPlan] = useState<CharacterSourceUpdate>();
  const [choices, setChoices] = useState<SourceChoices>({});
  const [preview, setPreview] = useState<FinalSourceUpdatePreview>();
  const [reviewing, setReviewing] = useState(false);
  const [pending, setPending] = useState(true);
  const [uncertain, setUncertain] = useState(false);
  const [error, setError] = useState('');
  const flight = useRef(false);

  useEffect(() => {
    let active = true;
    api.getCharacterSourceUpdate(character.id).then(value => { if (active) setPlan(value); })
      .catch(error => { if (active) setError(String(error)); })
      .finally(() => { if (active) setPending(false); });
    return () => { active = false; };
  }, [character.id]);

  async function request(operation: () => Promise<void>) {
    if (flight.current) return;
    flight.current = true; setPending(true); setError('');
    try { await operation(); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { flight.current = false; setPending(false); }
  }

  function restart() {
    if (flight.current || (!reviewing && !begin())) return;
    setReviewing(true); setChoices({}); setPreview(undefined); setPlan(undefined); setUncertain(false);
    void request(async () => { setPlan(await api.getCharacterSourceUpdate(character.id)); });
  }

  const merge = plan?.source_merge;
  const resolvable = merge && (merge.status === 'ready' || (merge.status === 'conflict' && merge.conflicts.length > 0 &&
    merge.conflicts.every(c => ['both_modified', 'local_delete_author_modify', 'local_modify_author_delete'].includes(c.kind))));
  const complete = resolvable && merge.conflicts.every(c => choices[c.id] === 'local' || choices[c.id] === 'author');
  const title = (id: string) => plan?.current_sources.find(s => s.id === id)?.title || plan?.incoming_sources.find(s => s.id === id)?.title || id;

  return <section className="space-y-3 rounded-xl border p-5" aria-label="作者资料更新">
    <h2 className="text-lg font-medium">作者更新</h2>
    <p className="text-sm">设定采用作者发布的完整版本；独立资料按服务端三方比较处理。已有会话保持原版。</p>
    {error && <p role="alert" className="text-destructive">{error}</p>}
    {pending && <p role="status">正在读取或确认，请稍候…</p>}
    {!reviewing && <Button type="button" disabled={disabled || pending} onClick={restart}>检查并预览作者更新</Button>}
    {plan && <p>{plan.available ? `有可检查的作者发布（设定版本 ${plan.source_revision}）` : '暂无可导入的作者更新'}</p>}
    {merge?.status === 'blocked' && <p role="status">{reasons[merge.reason || ''] || `服务端阻止导入：${merge.reason || '未知原因'}`}</p>}
    {reviewing && <>
      <p className="text-sm text-muted-foreground">审核期间暂时锁定角色及资料编辑。取消后可继续编辑，保存后请重新预览。</p>
      {plan?.available && merge && <>
        <details><summary className="cursor-pointer">查看已保存设定与作者设定的差异</summary>
          {Object.entries(plan.changes).map(([field, change]) => <div key={field} className="my-2">
            <p>{field}</p><pre className="whitespace-pre-wrap break-words text-sm">当前：{JSON.stringify(change.current, null, 2)}</pre>
            <pre className="whitespace-pre-wrap break-words text-sm">作者：{JSON.stringify(change.incoming, null, 2)}</pre>
          </div>)}
        </details>
        <ul className="space-y-1">{merge.actions.map(a => <li key={a.id}>{actions[a.action] || `服务端处理：${a.action}`}：{title(a.id)}</li>)}</ul>
        {merge.conflicts.map(conflict => {
          const local = plan.current_sources.filter(s => s.id === conflict.id);
          const author = plan.incoming_sources.filter(s => s.id === conflict.id);
          return <fieldset key={conflict.id} disabled={pending || !!preview || !resolvable} className="space-y-3 rounded border p-3">
            <legend>{title(conflict.id)} · {kinds[conflict.kind] || `无法处理的冲突：${conflict.kind}`}</legend>
            <div className="grid gap-3 md:grid-cols-2">
              <div><p>本地版本</p>{local.length ? <SourcePreview sources={local} /> : <p>本地已删除；选择本地将使新版不包含此篇。</p>}</div>
              <div><p>作者版本</p>{author.length ? <SourcePreview sources={author} /> : <p>作者已删除；选择作者将使新版不包含此篇。</p>}</div>
            </div>
            {resolvable && <div className="flex flex-wrap gap-4">{(['local', 'author'] as const).map(side => <label key={side}>
              <input type="radio" name={`resolve-${conflict.id}`} checked={choices[conflict.id] === side} onChange={() => {
                setChoices(previous => ({ ...previous, [conflict.id]: side })); setPreview(undefined);
              }} /> {side === 'local' ? local.length ? '保留本地版本' : '采用本地删除' : author.length ? '采用作者版本' : '采用作者删除'}
            </label>)}</div>}
          </fieldset>;
        })}
        {!resolvable && <p role="status">此计划只能查看，不能原地导入。请保留当前资料；刷新不会自动修复未知基线或 ID 碰撞。</p>}
        {!preview && <Button type="button" disabled={pending || !complete} onClick={() => void request(async () => {
          if (plan.source_revision === null || plan.publication_id === null) return;
          try {
            setPreview(await api.previewCharacterSourceUpdate(character.id, { revision: plan.revision, source_revision: plan.source_revision,
              publication_id: plan.publication_id, local_corpus_id: plan.local_corpus_id, choices }));
          } catch (error) {
            setChoices({}); setPlan(undefined);
            throw new Error(`无法生成最终预览，请重新检查并重新选择。${error instanceof Error ? error.message : String(error)}`);
          }
        })}>生成最终预览</Button>}
      </>}
      {preview && <div className="space-y-3 rounded border p-3" aria-label="作者更新最终预览">
        <h3 className="font-medium">确认后将保存以下完整内容</h3>
        <details><summary className="cursor-pointer">查看最终角色设定</summary><pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-sm">{JSON.stringify(preview.settings, null, 2)}</pre></details>
        <SourcePreview sources={preview.sources} />
        {preview.source_merge.resolved_conflicts?.map(c => <p key={c.id}>{title(c.id)}：已选择{c.choice === 'local' ? '本地' : '作者'}{preview.sources.some(s => s.id === c.id) ? '版本' : '删除（新版不包含此篇）'}</p>)}
        {uncertain && <p role="status">上次确认结果未能核实。请重试同一确认以核实结果，暂不允许编辑或切换选择。</p>}
        <Button type="button" disabled={pending} onClick={() => void request(async () => {
          let saved: CharacterLibraryRecord;
          try { saved = await api.confirmCharacterSourceUpdate(character.id, preview); }
          catch (error) {
            if (error instanceof ApiRequestError && [404, 409, 422].includes(error.status)) {
              setPreview(undefined); setChoices({}); setPlan(undefined); setUncertain(false);
              throw new Error(`预览已失效或被拒绝，请重新检查并重新选择。${error.message}`);
            }
            setUncertain(true); throw error;
          }
          setPreview(undefined); setUncertain(false); setReviewing(false);
          try { await onImported(saved); } finally { end(); }
        })}>{uncertain ? '重试同一确认' : '确认保存此最终版本'}</Button>
      </div>}
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" disabled={pending || uncertain} onClick={restart}>重新检查（清空选择）</Button>
        <Button type="button" variant="outline" disabled={pending || uncertain} onClick={() => {
          if (flight.current) return;
          setReviewing(false); setPreview(undefined); setChoices({}); setError(''); end();
        }}>取消更新，继续编辑</Button>
      </div>
    </>}
  </section>;
}
