'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { RouterAdapterRecord } from '@/lib/api-contracts';

export function useAdapterChecks() {
  const [adapters, setAdapters] = useState<RouterAdapterRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getRouterAdapters();
      if (!result.success || !Array.isArray(result.adapters)) throw new Error('适配器列表格式错误');
      setAdapters(result.adapters);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载适配器列表失败');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const checkAdapter = useCallback(async (name: string) => {
    setChecking(name);
    try {
      const result = await api.checkAdapter(name);
      if (!result.success) throw new Error('适配器检查未完成');
      if (!(await refetch())) throw new Error('检查已完成，但列表刷新失败，请刷新查看结果');
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : '适配器检查失败';
      setError(message);
      throw err;
    } finally {
      setChecking(null);
    }
  }, [refetch]);

  useEffect(() => { void refetch(); }, [refetch]);
  return { adapters, loading, error, checking, refetch, checkAdapter };
}
