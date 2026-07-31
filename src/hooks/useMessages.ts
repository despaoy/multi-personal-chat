'use client';

/**
 * 消息记录 Hook
 *
 * 获取分页消息记录列表，包含用户消息和机器人回复。
 * 支持设置分页参数，提供手动刷新功能。
 *
 * @param {number} [limit=20] - 每页返回的消息数
 * @param {number} [offset=0] - 分页偏移量
 * @returns {{ messages: Message[], total: number, loading: boolean, error: string | null, refetch: () => Promise<void> }}
 *   - messages: 消息记录数组
 *   - total: 消息总数
 *   - loading: 是否正在加载
 *   - error: 错误信息（如有）
 *   - refetch: 手动刷新数据的函数
 */

import { useState, useEffect, useCallback } from 'react';
import { api, type Message, type MessageFilters } from '@/lib/api';

export function useMessages(
  limit = 20,
  offset = 0,
  enabled = true,
  filters?: MessageFilters,
) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [total, setTotal] = useState(0);
  const [totalAll, setTotalAll] = useState(0);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);

  const fetchMessages = useCallback(async () => {
    if (!enabled) return;
    try {
      setLoading(true);
      setError(null);
      const data = await api.getMessages(limit, offset, filters);
      setMessages(data.messages);
      setTotal(data.total);
      setTotalAll(data.total_all ?? data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取消息记录失败');
      console.error('Failed to fetch messages:', err);
    } finally {
      setLoading(false);
    }
  }, [limit, offset, enabled, filters]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }

    fetchMessages();
  }, [fetchMessages, enabled]);

  return {
    messages,
    total,
    totalAll,
    loading,
    error,
    refetch: fetchMessages,
  };
}
