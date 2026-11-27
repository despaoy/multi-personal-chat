'use client';

import { useEffect, useState } from 'react';

/** A provisional snapshot, never a substitute for a persisted conversation turn. */
export function GenerationDraft({ sessionId, requestId }: { sessionId: string; requestId: string }) {
  const [text, setText] = useState('');
  const [status, setStatus] = useState('正在连接生成进度…');

  useEffect(() => {
    const source = new EventSource(`/api/character-chats/${encodeURIComponent(sessionId)}/generations/${encodeURIComponent(requestId)}/events`, { withCredentials: true });
    source.addEventListener('generation', event => {
      try {
        const value = JSON.parse((event as MessageEvent).data);
        if (value.request_id !== requestId || typeof value.status !== 'string') return;
        if (value.status === 'running') {
          setText(typeof value.draft_text === 'string' ? value.draft_text : '');
          setStatus(value.draft_text ? '以下为尚未保存的草稿' : '正在等待回复草稿…');
        } else {
          source.close();
          setText('');
          setStatus(value.status === 'completed' ? '回复已保存，正在刷新会话…' : '生成已停止，请查看会话状态后重试。');
        }
      } catch {
        source.close();
        setText('');
        setStatus('生成进度无法读取，请刷新会话核对结果。');
      }
    });
    source.addEventListener('revoked', () => {
      source.close(); setText(''); setStatus('会话已不可访问。');
    });
    source.addEventListener('unavailable', () => {
      source.close(); setText(''); setStatus('暂未取得生成进度，请刷新会话核对结果。');
    });
    source.onerror = () => setStatus('生成进度连接中断，正在重连…');
    return () => source.close();
  }, [sessionId, requestId]);

  return <section aria-label="正在生成的回复" className="space-y-2 rounded-lg border border-dashed p-4">
    <p role="status" className="text-sm text-muted-foreground">{status}</p>
    {text && <p className="whitespace-pre-wrap break-words">{text}</p>}
  </section>;
}
