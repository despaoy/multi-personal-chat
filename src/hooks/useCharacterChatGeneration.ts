'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { SendCharacterChatMessage } from '@/lib/api-contracts';
import { createClientId } from '@/lib/client-id';

export type CharacterChatSnapshot = Awaited<ReturnType<typeof api.getCharacterChat>>;

type PendingMessage = { sessionId: string; body: SendCharacterChatMessage };

function includesTurn(snapshot: CharacterChatSnapshot, requestId: string) {
  return snapshot.turns.some(turn => turn.request_id === requestId);
}

export function useCharacterChatGeneration(
  snapshot: CharacterChatSnapshot | undefined,
  { paused, onPoll }: { paused: boolean; onPoll: (snapshot: CharacterChatSnapshot) => void },
) {
  const [draft, setDraft] = useState('');
  const [editing, setEditing] = useState<string>();
  const [streamRequest, setStreamRequest] = useState<{ sessionId: string; requestId: string } | null>(null);
  const pending = useRef<PendingMessage | null>(null);
  const sessionId = snapshot?.session.id;
  const generation = snapshot?.generation;
  const generationStatus = generation?.status;
  const enrichmentPending = snapshot?.enrichment_pending;

  const clearDraft = useCallback(() => {
    pending.current = null;
    setDraft('');
    setEditing(undefined);
  }, []);

  function editDraft(turnId: string, message: string) {
    pending.current = null;
    setDraft(message);
    setEditing(turnId);
  }

  function restoreDraft(sessionId: string, body: SendCharacterChatMessage) {
    pending.current = { sessionId, body };
    setDraft(body.message);
    setEditing(body.replace_turn_id || undefined);
  }

  async function send({
    reload,
    refreshSessions,
    refreshShared,
  }: {
    reload: (sessionId: string) => Promise<CharacterChatSnapshot>;
    refreshSessions: () => Promise<void>;
    refreshShared: () => Promise<void>;
  }) {
    const session = snapshot?.session;
    if (!session || !draft.trim() || generationStatus === 'running' || (enrichmentPending && !editing)) return;
    const existing = pending.current;
    const body = existing?.sessionId === session.id
      && existing.body.message === draft
      && (existing.body.replace_turn_id || undefined) === editing
      ? existing.body
      : {
        revision: session.revision,
        request_id: createClientId(),
        message: draft,
        ...(editing ? { replace_turn_id: editing } : {}),
      };
    pending.current = { sessionId: session.id, body };
    setStreamRequest({ sessionId: session.id, requestId: body.request_id });
    try {
      await api.sendCharacterChat(session.id, body);
      clearDraft();
      await reload(session.id);
      await refreshSessions();
      if (body.replace_turn_id) await refreshShared();
    } catch (error) {
      // The server may have saved the reply before the connection failed.
      try {
        const response = await reload(session.id);
        if (includesTurn(response, body.request_id)) {
          clearDraft();
          if (body.replace_turn_id) await refreshShared();
          return;
        }
        if (response.session.revision !== body.revision) pending.current = null;
      } catch { /* Keep the original failure and exact retry identity. */ }
      throw error;
    } finally {
      setStreamRequest(null);
    }
  }

  const watchedRequest = streamRequest && streamRequest.sessionId === sessionId
    ? streamRequest.requestId
    : generation?.status === 'running' ? generation.request_id : undefined;

  useEffect(() => {
    if (!sessionId || (generationStatus !== 'running' && !enrichmentPending) || paused) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const response = await api.getCharacterChat(sessionId);
        if (!active) return;
        onPoll(response);
        const waiting = pending.current;
        if (waiting?.sessionId === sessionId && includesTurn(response, waiting.body.request_id)) {
          pending.current = null;
          setDraft(current => current === waiting.body.message ? '' : current);
          setEditing(undefined);
        }
        if (response.generation?.status !== 'running' && !response.enrichment_pending) return;
      } catch { /* Keep polling after a transient disconnect; manual refresh remains available. */ }
      if (active) timer = setTimeout(poll, 3000);
    };
    timer = setTimeout(poll, 1500);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [sessionId, generationStatus, enrichmentPending, paused, onPoll]);

  return { draft, setDraft, editing, editDraft, clearDraft, restoreDraft, watchedRequest, send };
}
