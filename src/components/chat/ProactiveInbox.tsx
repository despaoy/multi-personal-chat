'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { ProactiveInboxMessage } from '@/lib/api-contracts';

export function ProactiveInbox({ hiddenSessions }: { hiddenSessions: ReadonlySet<string> }) {
  const [messages, setMessages] = useState<ProactiveInboxMessage[]>([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    let generation = 0;
    const refresh = async () => {
      const current = ++generation;
      try {
        const result = await api.getCharacterProactiveInbox();
        if (active && current === generation) { setMessages(result.messages); setError(false); }
      } catch {
        if (active && current === generation) setError(true);
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 30000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  const visible = messages.filter(message => !hiddenSessions.has(message.session_id));
  if (!visible.length && !error) return null;
  return <section aria-label="主动消息收件箱" className="space-y-2 rounded-lg border p-4">
    <h2 className="text-sm font-medium">最近主动消息</h2>
    {error && <p role="status" className="text-sm">暂时无法刷新主动消息。</p>}
    <ul className="space-y-2 text-sm">{visible.map(message => <li key={message.session_id}>
      <a className="block rounded-md border p-2 hover:bg-muted"
        href={`/chat?${new URLSearchParams({ character: message.character_id, session: message.session_id })}`}>
        <span className="block font-medium">{message.title}</span>
        <span className="block break-words">{message.content}</span>
        <time className="text-xs text-muted-foreground">{new Date(message.sent_at * 1000).toLocaleString()}</time>
      </a>
    </li>)}</ul>
  </section>;
}
