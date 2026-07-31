'use client';

/**
 * 评估管理 Hook
 *
 * 管理 Gold 评估集、评估运行历史和用户反馈的获取与操作。
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import type {
  EvaluationRunRecord,
  EvaluationRunRequest,
  FeedbackRecord,
  GoldPromptRecord,
  GoldSetResponse,
} from '@/lib/api-contracts';

export type GoldPrompt = GoldPromptRecord;

export type EvalRun = EvaluationRunRecord;

export type Feedback = FeedbackRecord;
export type GoldSet = Pick<GoldSetResponse, 'total' | 'category_breakdown' | 'prompts'>;

export function useEvaluation(enabled = true) {
  const [goldSet, setGoldSet] = useState<GoldSet>({
    total: 0,
    category_breakdown: {},
    prompts: [],
  });
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [runsTotal, setRunsTotal] = useState(0);
  const [feedbacks, setFeedbacks] = useState<Feedback[]>([]);
  const [feedbackTotal, setFeedbackTotal] = useState(0);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const fetchAll = useCallback(async () => {
    if (!enabled) return;
    try {
      setLoading(true);
      setError(null);
      const [gold, runsResp, fbResp] = await Promise.all([
        api.getGoldSet(),
        api.getEvaluationRuns(),
        api.getFeedback().catch(() => ({ feedbacks: [], total: 0 })),
      ]);
      setGoldSet({
        total: gold.total,
        category_breakdown: gold.category_breakdown || {},
        prompts: gold.prompts || [],
      });
      setRuns(runsResp.runs || []);
      setRunsTotal(runsResp.total ?? runsResp.runs?.length ?? 0);
      setFeedbacks(fbResp.feedbacks || []);
      setFeedbackTotal(fbResp.total ?? fbResp.feedbacks?.length ?? 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载评估数据失败');
      console.error('Failed to fetch evaluation data:', err);
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  const runEvaluation = useCallback(async (req: EvaluationRunRequest) => {
    try {
      setRunning(true);
      const result = await api.runEvaluation(req);
      await fetchAll();
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : '运行评估失败');
      console.error('Failed to run evaluation:', err);
      throw err;
    } finally {
      setRunning(false);
    }
  }, [fetchAll]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    fetchAll();
  }, [enabled, fetchAll]);

  return {
    goldSet,
    runs,
    runsTotal,
    feedbacks,
    feedbackTotal,
    loading,
    error,
    running,
    refetch: fetchAll,
    runEvaluation,
  };
}
