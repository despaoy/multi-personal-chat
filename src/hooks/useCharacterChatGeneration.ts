'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiRequestError } from '@/lib/api';
import type { SendCharacterChatMessage } from '@/lib/api-contracts';
import { createClientId } from '@/lib/client-id';

export type CharacterChatSnapshot = Awaited<ReturnType<typeof api.getCharacterChat>>;

type PendingMessage = { sessionId: string; body: SendCharacterChatMessage };
type Recovery = PendingMessage & { deadline: number; error: string; previousAttempt?: number };
const STATUS_READ_TIMEOUT_MS = 10_000;

function includesTurn(snapshot: CharacterChatSnapshot, requestId: string) {
  return snapshot.turns.some(turn => turn.request_id === requestId);
}

export function useCharacterChatGeneration(
  snapshot: CharacterChatSnapshot | undefined,
  { paused, onPoll }: { paused: boolean; onPoll: (snapshot: CharacterChatSnapshot, recoveredError?: string) => void },
) {
  const [draft, setDraft] = useState('');
  const [editing, setEditing] = useState<string>();
  const [streamRequest, setStreamRequest] = useState<{ sessionId: string; requestId: string } | null>(null);
  const pending = useRef<PendingMessage | null>(null);
  const [recovery, setRecovery] = useState<Recovery | null>(null);
  const sessionId = snapshot?.session.id;
  const generation = snapshot?.generation;
  const generationStatus = generation?.status;
  const enrichmentPending = snapshot?.enrichment_pending;

  const clearDraft = useCallback(() => {
    setRecovery(null);
    pending.current = null;
    setDraft('');
    setEditing(undefined);
  }, []);

  function editDraft(turnId: string, message: string) {
    setRecovery(null);
    pending.current = null;
    setDraft(message);
    setEditing(turnId);
  }

  function restoreDraft(sessionId: string, body: SendCharacterChatMessage) {
    setRecovery(null);
    pending.current = { sessionId, body };
    setDraft(body.message);
    setEditing(body.replace_turn_id || undefined);
  }

  async function send({
    reload,
    refreshSessions,
    refreshShared,
  }: {
    reload: (sessionId: string, options: { signal: AbortSignal }) => Promise<CharacterChatSnapshot>;
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
    setRecovery(null);
    setStreamRequest({ sessionId: session.id, requestId: body.request_id });
    try {
      await api.sendCharacterChat(session.id, body);
      await reload(session.id, { signal: AbortSignal.timeout(STATUS_READ_TIMEOUT_MS) });
      clearDraft();
    } catch (error) {
      // The server may have saved the reply before the connection failed.
      try {
        const response = await reload(session.id, { signal: AbortSignal.timeout(STATUS_READ_TIMEOUT_MS) });
        if (includesTurn(response, body.request_id)) {
          clearDraft();
          if (body.replace_turn_id) await refreshShared();
          return;
        }
        if (response.session.revision !== body.revision) {
          pending.current = null;
        }
      } catch (readError) {
        if (readError instanceof ApiRequestError && [401, 403, 404].includes(readError.status)) throw error;
      }
      if (pending.current?.body === body && (!(error instanceof ApiRequestError) || error.status >= 500)) {
        setRecovery({
          sessionId: session.id, body, deadline: Date.now() + 60_000,
          error: error instanceof Error ? error.message : String(error),
          previousAttempt: generation?.request_id === body.request_id ? generation.attempt : undefined,
        });
      }
      throw error;
    } finally {
      setStreamRequest(null);
    }
    await refreshSessions();
    if (body.replace_turn_id) await refreshShared();
  }

  const watchedRequest = streamRequest && streamRequest.sessionId === sessionId
    ? streamRequest.requestId
    : generation?.status === 'running' ? generation.request_id : undefined;

  const recovering = recovery?.sessionId === sessionId ? recovery : null;
  useEffect(() => {
    if (!sessionId || (generationStatus !== 'running' && !enrichmentPending && !recovering) || paused) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(STATUS_READ_TIMEOUT_MS)]);
        const response = await api.getCharacterChat(sessionId, signal);
        if (controller.signal.aborted) return;
        const saved = recovering && includesTurn(response, recovering.body.request_id);
        onPoll(response, saved ? recovering.error : undefined);
        const waiting = pending.current;
        if (waiting?.sessionId === sessionId && includesTurn(response, waiting.body.request_id)) {
          pending.current = null;
          setDraft(current => current === waiting.body.message ? '' : current);
          setEditing(undefined);
        }
        const attempt = response.generation;
        const stopped = recovering && attempt && attempt.request_id === recovering.body.request_id
          && attempt.status !== 'running' && attempt.attempt !== recovering.previousAttempt;
        if (recovering && (saved || stopped || response.session.revision !== recovering.body.revision)) {
          setRecovery(null);
          return;
        }
        if (!recovering && response.generation?.status !== 'running' && !response.enrichment_pending) return;
      } catch (error) {
        if (controller.signal.aborted) return;
        if (error instanceof ApiRequestError && [401, 403, 404].includes(error.status)) {
          setRecovery(null);
          return;
        }
      }
      if (recovering && Date.now() >= recovering.deadline) {
        setRecovery(null);
        return;
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 3000);
    };
    timer = setTimeout(poll, 1500);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [sessionId, generationStatus, enrichmentPending, paused, onPoll, recovering]);

  return { draft, setDraft, editing, editDraft, clearDraft, restoreDraft, watchedRequest, send };
}
