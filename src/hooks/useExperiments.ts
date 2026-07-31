'use client';

/**
 * 实验管理 Hook
 *
 * 管理 LoRA 消融、RAG 消融、量化基准实验的获取与启动。
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import type { ExperimentRecord, ExperimentStartRequest } from '@/lib/api-contracts';

export type Experiment = ExperimentRecord;

export type ExperimentType = 'lora-ablation' | 'rag-ablation' | 'quantization-benchmark';

export function useExperiments(enabled = true) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const fetchExperiments = useCallback(async () => {
    if (!enabled) return;
    try {
      setLoading(true);
      setError(null);
      const data = await api.getExperiments();
      setExperiments(data.experiments || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取实验列表失败');
      console.error('Failed to fetch experiments:', err);
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  const startExperiment = useCallback(async (type: ExperimentType, req: ExperimentStartRequest) => {
    try {
      setStarting(true);
      const result = await api.startExperiment(type, req);
      await fetchExperiments();
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动实验失败');
      console.error('Failed to start experiment:', err);
      throw err;
    } finally {
      setStarting(false);
    }
  }, [fetchExperiments]);

  const getReport = useCallback(async (expId: string) => {
    try {
      return await api.getExperimentReport(expId);
    } catch (err) {
      console.error('Failed to get experiment report:', err);
      throw err;
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    fetchExperiments();
  }, [enabled, fetchExperiments]);

  return {
    experiments,
    loading,
    error,
    starting,
    refetch: fetchExperiments,
    startExperiment,
    getReport,
  };
}
