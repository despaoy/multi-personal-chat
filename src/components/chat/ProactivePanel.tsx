'use client';

import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { api } from '@/lib/api';
import { createClientId } from '@/lib/client-id';
import type { CharacterProactivePolicy, CharacterProactiveState, ProactiveTopicKind } from '@/lib/api-contracts';

function timeLabel(minutes: number | null) {
  return minutes === null ? '' : [Math.floor(minutes / 60), minutes % 60].map(part => String(part).padStart(2, '0')).join(':');
}

function minutesAt(value: string) {
  if (!value) return null;
  const [hours, minutes] = value.split(':').map(Number);
  return hours * 60 + minutes;
}

export function ProactivePanel({ sessionId }: { sessionId: string }) {
  const [view, setView] = useState<CharacterProactiveState>();
  const [draft, setDraft] = useState<CharacterProactivePolicy>();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [topic, setTopic] = useState('');
  const [kind, setKind] = useState<ProactiveTopicKind>('follow_up');
  const [time, setTime] = useState('');
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
        setDraft(previous => previous ?? {
          revision: result.policy.revision, enabled: result.policy.enabled, cadence_hours: result.policy.cadence_hours,
          quiet_start: result.policy.quiet_start, quiet_end: result.policy.quiet_end,
          inactivity_hours: result.policy.inactivity_hours ?? null,
          timezone: result.policy.revision ? result.policy.timezone : Intl.DateTimeFormat().resolvedOptions().timeZone,
        });
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

  return <section aria-label="主动联系" className="space-y-3">
    {!!view?.jobs.some(job => job.status === 'sent') && <div className="space-y-2 rounded-lg border p-4 text-sm">
      <h2 className="font-medium">角色主动消息</h2>
      <ul className="space-y-2">{view.jobs.filter(job => job.status === 'sent').map(job => <li key={job.id} className="rounded border p-2">
        <p>{job.content}</p>
        <time className="text-muted-foreground">{new Date((job.sent_at ?? job.due_at) * 1000).toLocaleString()}</time>
      </li>)}</ul>
    </div>}
    <details className="space-y-3 rounded-lg border p-4">
    <summary className="cursor-pointer">主动联系与消息</summary>
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {draft && <form className="space-y-2" onSubmit={event => {
      event.preventDefault();
      void run(async () => {
        setDraft(await api.saveCharacterProactive(sessionId, draft));
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
      <Button type="submit" variant="outline" disabled={busy}>保存主动联系设置</Button>
      {view?.policy.revision !== draft.revision && <Button type="button" variant="ghost" disabled={busy}
        onClick={() => { if (view) setDraft(view.policy); }}>重新载入最新设置</Button>}
    </form>}
    {view?.policy.enabled && <form className="space-y-2" onSubmit={event => {
      event.preventDefault();
      const dueAt = new Date(time).getTime() / 1000;
      if (!Number.isFinite(dueAt)) return;
      const job = { revision: view.policy.revision, kind, topic: topic.trim(), due_at: dueAt };
      const key = JSON.stringify(job);
      if (retry.current?.key !== key) retry.current = { key, id: createClientId() };
      const requestId = retry.current.id;
      void run(async () => {
        await api.scheduleCharacterProactive(sessionId, { ...job, request_id: requestId });
        setTopic('');
        setTime('');
        retry.current = null;
      });
    }}>
      <label className="block text-sm">消息类型
        <select className="ml-2 rounded border p-1" value={kind} disabled={busy} onChange={event => setKind(event.target.value as ProactiveTopicKind)}>
          <option value="follow_up">接续话题</option>
          <option value="event">关心近况</option>
          <option value="fiction">虚构日常</option>
        </select>
      </label>
      <label className="block text-sm">话题 <input className="ml-2 rounded border p-1" required maxLength={300} value={topic}
        disabled={busy} onChange={event => setTopic(event.target.value)} /></label>
      <label className="block text-sm">最早发送时间（本机时区） <input className="ml-2 rounded border p-1" required type="datetime-local" value={time}
        disabled={busy} onChange={event => setTime(event.target.value)} /></label>
      <Button type="submit" variant="outline" disabled={busy || !topic.trim()}>安排应用内消息</Button>
    </form>}
      {!!view?.jobs.some(job => job.status === 'pending') && <ul className="space-y-2 text-sm">{view.jobs.filter(job => job.status === 'pending').map(job => <li key={job.id} className="rounded border p-2">
        <p>待发送：{job.topic}</p>
        <time className="text-muted-foreground">{new Date(job.due_at * 1000).toLocaleString()}</time>
      </li>)}</ul>}
    </details>
  </section>;
}
