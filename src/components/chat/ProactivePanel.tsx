'use client';

import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { api, ApiRequestError } from '@/lib/api';
import { createClientId } from '@/lib/client-id';
import { BrowserPushControl } from './BrowserPushControl';
import type {
  CharacterFictionHistory, CharacterProactivePolicy, CharacterProactiveState, ProactiveCandidate, ProactiveTopicKind,
} from '@/lib/api-contracts';

function timeLabel(minutes: number | null) {
  return minutes === null ? '' : [Math.floor(minutes / 60), minutes % 60].map(part => String(part).padStart(2, '0')).join(':');
}

function minutesAt(value: string) {
  if (!value) return null;
  const [hours, minutes] = value.split(':').map(Number);
  return hours * 60 + minutes;
}

function localDateTime(timestamp: number) {
  const date = new Date(timestamp * 1000);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
}

export function ProactivePanel({ sessionId }: { sessionId: string }) {
  const [view, setView] = useState<CharacterProactiveState>();
  const [edits, setDraft] = useState<CharacterProactivePolicy>();
  const policy = view?.policy;
  const draft = edits ?? (policy && (policy.revision ? policy : {
    ...policy, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  }));
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [topic, setTopic] = useState('');
  const [kind, setKind] = useState<ProactiveTopicKind>('follow_up');
  const [time, setTime] = useState('');
  const [candidates, setCandidates] = useState<ProactiveCandidate[]>();
  const [source, setSource] = useState<ProactiveCandidate>();
  const [fiction, setFiction] = useState<CharacterFictionHistory>();
  const [scheduleDraft, setScheduleDraft] = useState<{ id: string; expectedDueAt: number; time: string }>();
  const flight = useRef(false);
  const readGeneration = useRef(0);
  const retry = useRef<{ key: string; id: string } | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const generation = ++readGeneration.current;
      try {
        const result = await api.getCharacterProactive(sessionId);
        if (!active || generation !== readGeneration.current) return;
        setView(result);
      } catch (error) {
        if (active && generation === readGeneration.current) setError(error instanceof Error ? error.message : String(error));
      }
    };
    void refresh();
    const interval = setInterval(() => { if (!flight.current) void refresh(); }, 30000);
    return () => { active = false; clearInterval(interval); };
  }, [sessionId]);

  async function run(operation: () => Promise<void>) {
    if (flight.current) return;
    flight.current = true;
    readGeneration.current++;
    setError('');
    setBusy(true);
    try {
      await operation();
      setView(await api.getCharacterProactive(sessionId));
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      flight.current = false;
      setBusy(false);
    }
  }

  function changeSchedule(jobId: string, expectedDueAt: number, dueAt: number | null) {
    void run(async () => {
      try {
        await api.changeCharacterProactiveSchedule(sessionId, jobId, expectedDueAt, dueAt);
        setScheduleDraft(undefined);
      } catch (error) {
        if (error instanceof ApiRequestError && [404, 409].includes(error.status)) {
          setScheduleDraft(undefined);
          try { setView(await api.getCharacterProactive(sessionId)); } catch { /* Report the original error. */ }
        }
        throw error;
      }
    });
  }

  return <section aria-label="主动联系" className="space-y-3">
    {!!view?.jobs.some(job => job.status === 'sent') && <div className="space-y-2 rounded-lg border p-4 text-sm">
      <h2 className="font-medium">角色主动消息</h2>
      <ul className="space-y-2">{view.jobs.filter(job => job.status === 'sent').map(job => <li key={job.id} className="rounded border p-2">
        <p>{job.content}</p>
        <time className="text-muted-foreground">{new Date((job.sent_at ?? job.due_at) * 1000).toLocaleString()}</time>
      </li>)}</ul>
    </div>}
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    <details className="space-y-2 rounded-lg border p-4">
      <summary className="cursor-pointer">虚构日常剧情记录</summary>
      <Button type="button" variant="outline" disabled={busy} onClick={() => void run(async () => {
        setFiction(await api.getCharacterFictionHistory(sessionId));
      })}>{fiction ? '刷新剧情记录' : '载入剧情记录'}</Button>
      {fiction?.entries.length === 0 && <p className="text-sm text-muted-foreground">暂无已送达的虚构日常。</p>}
      {!!fiction?.entries.length && <ol className="space-y-2 text-sm">{fiction.entries.map(entry => <li key={entry.id}
        className="rounded border p-2">
        <p>{entry.content}</p>
        <time className="text-muted-foreground">{new Date(entry.sent_at * 1000).toLocaleString()}</time>
      </li>)}</ol>}
      {fiction?.next_before && <Button type="button" variant="ghost" disabled={busy} onClick={() => {
        const before = fiction.next_before;
        if (!before) return;
        void run(async () => {
          const result = await api.getCharacterFictionHistory(sessionId, before);
          setFiction({ entries: [...fiction.entries, ...result.entries], next_before: result.next_before });
        });
      }}>加载更早记录</Button>}
    </details>
    <details className="space-y-3 rounded-lg border p-4">
    <summary className="cursor-pointer">主动联系与消息</summary>
    <BrowserPushControl sessionId={sessionId} authorized={Boolean(view?.policy.enabled)} />
    {draft && <form className="space-y-2" onSubmit={event => {
      event.preventDefault();
      void run(async () => {
        const saved = await api.saveCharacterProactive(sessionId, draft);
        setView(current => current && { ...current, policy: saved });
        setDraft(undefined);
      });
    }}>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={draft.enabled} disabled={busy} onChange={event => setDraft({ ...draft, enabled: event.target.checked })} />
        允许这个会话主动联系我
      </label>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={draft.inactivity_hours !== null} disabled={busy || !draft.enabled}
          onChange={event => setDraft({ ...draft, inactivity_hours: event.target.checked ? 168 : null })} />
        允许久未联系时问候一次
      </label>
      {draft.inactivity_hours !== null && <label className="block text-sm">未联系多久后问候（小时）
        <input className="ml-2 w-20 rounded border p-1" required type="number" min={72} max={720}
          value={draft.inactivity_hours} disabled={busy || !draft.enabled}
          onChange={event => setDraft({ ...draft, inactivity_hours: Number(event.target.value) })} />
      </label>}
      <label className="block text-sm">发送间隔（小时）
        <input className="ml-2 w-20 rounded border p-1" required type="number" min={24} max={168} value={draft.cadence_hours}
          disabled={busy} onChange={event => setDraft({ ...draft, cadence_hours: Number(event.target.value) })} />
      </label>
      <label className="block text-sm">免打扰时区
        <input className="ml-2 rounded border p-1" required maxLength={80} value={draft.timezone}
          disabled={busy} onChange={event => setDraft({ ...draft, timezone: event.target.value })} />
      </label>
      <div className="flex flex-wrap gap-3 text-sm">
        <label>免打扰开始 <input disabled={busy} type="time" value={timeLabel(draft.quiet_start)} onChange={event => setDraft({ ...draft, quiet_start: minutesAt(event.target.value) })} /></label>
        <label>免打扰结束 <input disabled={busy} type="time" value={timeLabel(draft.quiet_end)} onChange={event => setDraft({ ...draft, quiet_end: minutesAt(event.target.value) })} /></label>
      </div>
      <p className="text-sm text-muted-foreground">两处时间都留空可取消免打扰。保存设置会取消此前尚未发送的安排。</p>
      <Button type="submit" variant="outline" disabled={busy || policy?.revision !== draft.revision}>保存主动联系设置</Button>
      {view?.policy.revision !== draft.revision && <Button type="button" variant="ghost" disabled={busy}
        onClick={() => setDraft(undefined)}>重新载入最新设置</Button>}
    </form>}
    {view?.policy.enabled && <section className="space-y-2" aria-label="从对话选取话题">
      <Button type="button" variant="outline" disabled={busy} onClick={() => void run(async () => {
        setCandidates((await api.getCharacterProactiveCandidates(sessionId)).candidates);
      })}>从最近对话查找候选</Button>
      {candidates?.length === 0 && <p className="text-sm text-muted-foreground">最近对话没有可选话题。</p>}
      {!!candidates?.length && <ul className="space-y-2 text-sm">{candidates.map(candidate => <li
        className="flex flex-wrap items-center gap-2" key={JSON.stringify([candidate.source_turn_id, candidate.kind, candidate.topic])}>
        <q>{candidate.topic}</q>
        <Button type="button" variant="ghost" disabled={busy} onClick={() => {
          setSource(candidate);
          setKind(candidate.kind);
          setTopic(candidate.topic);
          setTime(candidate.suggested_at === null ? '' : localDateTime(candidate.suggested_at));
        }}>使用这句话</Button>
      </li>)}</ul>}
    </section>}
    {view?.policy.enabled && <form className="space-y-2" onSubmit={event => {
      event.preventDefault();
      const dueAt = new Date(time).getTime() / 1000;
      if (!Number.isFinite(dueAt)) return;
      const job = { revision: view.policy.revision, kind, topic: topic.trim(), due_at: dueAt,
        ...(source ? { source_turn_id: source.source_turn_id } : {}) };
      const key = JSON.stringify(job);
      if (retry.current?.key !== key) retry.current = { key, id: createClientId() };
      const requestId = retry.current.id;
      void run(async () => {
        await api.scheduleCharacterProactive(sessionId, { ...job, request_id: requestId });
        setTopic('');
        setTime('');
        setSource(undefined);
        if (source) setCandidates(previous => previous?.filter(candidate =>
          candidate.source_turn_id !== source.source_turn_id || candidate.kind !== source.kind || candidate.topic !== source.topic,
        ));
        retry.current = null;
      });
    }}>
      <label className="block text-sm">消息类型
        <select className="ml-2 rounded border p-1" value={kind} disabled={busy || !!source} onChange={event => setKind(event.target.value as ProactiveTopicKind)}>
          <option value="follow_up">接续话题</option>
          <option value="event">关心近况</option>
          <option value="fiction">虚构日常</option>
        </select>
      </label>
      <label className="block text-sm">话题 <input className="ml-2 rounded border p-1" required maxLength={300} value={topic}
        disabled={busy} readOnly={!!source} onChange={event => setTopic(event.target.value)} /></label>
      {source && <Button type="button" variant="ghost" disabled={busy} onClick={() => {
        setSource(undefined);
        setTopic('');
        setTime('');
      }}>移除引用并手动填写</Button>}
      <label className="block text-sm">最早发送时间（本机时区） <input className="ml-2 rounded border p-1" required type="datetime-local" value={time}
        disabled={busy} onChange={event => setTime(event.target.value)} /></label>
      <Button type="submit" variant="outline" disabled={busy || !topic.trim()}>安排应用内消息</Button>
    </form>}
      {!!view?.jobs.some(job => job.status === 'pending') && <ul className="space-y-2 text-sm">{view.jobs.filter(job => job.status === 'pending').map(job => <li key={job.id} className="rounded border p-2">
        <p>待发送：{job.topic}</p>
        <time className="text-muted-foreground">{new Date(job.due_at * 1000).toLocaleString()}</time>
        {scheduleDraft?.id === job.id ? <form className="mt-2 flex flex-wrap items-center gap-2" onSubmit={event => {
          event.preventDefault();
          const dueAt = new Date(scheduleDraft.time).getTime() / 1000;
          if (Number.isFinite(dueAt)) changeSchedule(job.id, scheduleDraft.expectedDueAt, dueAt);
        }}>
          <label>新的最早发送时间（本机时区） <input className="rounded border p-1" type="datetime-local" required
            value={scheduleDraft.time} disabled={busy}
            onChange={event => setScheduleDraft({ ...scheduleDraft, time: event.target.value })} /></label>
          <Button type="submit" variant="outline" size="sm" disabled={busy}>保存新时间</Button>
          <Button type="button" variant="ghost" size="sm" disabled={busy} onClick={() => setScheduleDraft(undefined)}>放弃改期</Button>
        </form> : <div className="mt-2 flex flex-wrap gap-2">
          <Button type="button" variant="outline" size="sm" disabled={busy} onClick={() => {
            setScheduleDraft({ id: job.id, expectedDueAt: job.due_at, time: localDateTime(job.due_at) });
          }}>改期</Button>
          <Button type="button" variant="ghost" size="sm" disabled={busy}
            onClick={() => changeSchedule(job.id, job.due_at, null)}>取消安排</Button>
        </div>}
      </li>)}</ul>}
    </details>
  </section>;
}
