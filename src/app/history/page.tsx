'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { format } from 'date-fns';
import { AlertCircle, RefreshCw } from 'lucide-react';

import {
  BatchDeleteDialog,
  DeleteMessageDialog,
  MessageDetailDialog,
} from '@/components/history/HistoryDialogs';
import { HistoryFiltersCard } from '@/components/history/HistoryFiltersCard';
import { HistoryMessagesCard } from '@/components/history/HistoryMessagesCard';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { useMessages } from '@/hooks/useMessages';
import {
  api,
  type LoraModel,
  type Message,
  type MessageFilters,
} from '@/lib/api';

const PAGE_SIZE = 50;

export default function HistoryPage() {
  return (
    <AuthGuard requireAdmin>
      <HistoryContent />
    </AuthGuard>
  );
}

function HistoryContent() {
  const { user } = useAuth();
  const [searchTerm, setSearchTerm] = useState('');
  const [sessionType, setSessionType] = useState('all');
  const [platformFilter, setPlatformFilter] = useState('all');
  const [sessionNameFilter, setSessionNameFilter] = useState('');
  const [selectedLora, setSelectedLora] = useState('all');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [debouncedSessionName, setDebouncedSessionName] = useState('');
  const [page, setPage] = useState(0);
  const [detailMessage, setDetailMessage] = useState<Message | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [confirmBatchDelete, setConfirmBatchDelete] = useState(false);
  const [batchDeleting, setBatchDeleting] = useState(false);
  const [loraModels, setLoraModels] = useState<LoraModel[]>([]);

  const messageFilters = useMemo<MessageFilters>(() => ({
    platform: platformFilter,
    sessionType,
    search: debouncedSearch,
    lora: selectedLora,
    sessionName: debouncedSessionName,
  }), [platformFilter, sessionType, debouncedSearch, selectedLora, debouncedSessionName]);

  const batchDeleteFilters = useMemo<MessageFilters>(() => {
    const filters: MessageFilters = {};
    if (debouncedSearch) filters.search = debouncedSearch;
    if (sessionType !== 'all') filters.sessionType = sessionType;
    if (platformFilter !== 'all') filters.platform = platformFilter;
    if (selectedLora !== 'all') filters.lora = selectedLora;
    if (debouncedSessionName) filters.sessionName = debouncedSessionName;
    return filters;
  }, [debouncedSearch, sessionType, platformFilter, selectedLora, debouncedSessionName]);

  const { messages, total, totalAll, loading, error, refetch } = useMessages(
    PAGE_SIZE,
    page * PAGE_SIZE,
    Boolean(user),
    messageFilters,
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearch(searchTerm.trim());
      setDebouncedSessionName(sessionNameFilter.trim());
    }, 300);
    return () => window.clearTimeout(timer);
  }, [searchTerm, sessionNameFilter]);

  const loadLoras = useCallback(async () => {
    if (!user) return;
    try {
      const response = await api.getLoras();
      setLoraModels(response.loras);
    } catch {
      // 筛选项是增强信息，加载失败时保留消息中实际出现的 LoRA 名称。
    }
  }, [user]);

  useEffect(() => {
    loadLoras();
  }, [loadLoras]);

  const availableLoraNames = useMemo(() => {
    const names = new Set(loraModels.map((model) => model.name));
    for (const message of messages) {
      if (message.loraName) names.add(message.loraName);
    }
    return Array.from(names).sort();
  }, [loraModels, messages]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const handleExport = useCallback(() => {
    if (messages.length === 0) return;

    const headers = ['时间', '用户', '会话类型', '会话名称', '消息', '回复', '模型', 'LoRA', '耗时(s)'];
    const rows = messages.map((message) => [
      format(new Date(message.createdAt), 'yyyy-MM-dd HH:mm:ss'),
      message.userName,
      message.sessionType === 'group' ? '群聊' : '私聊',
      message.sessionName,
      message.message,
      message.reply,
      message.modelName,
      message.loraName,
      message.costTime,
    ]);
    const escapeCsvCell = (value: unknown) => `"${String(value ?? '').replace(/"/g, '""')}"`;
    const csv = [headers, ...rows]
      .map((row) => row.map(escapeCsvCell).join(','))
      .join('\r\n');
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `历史记录_${format(new Date(), 'yyyyMMdd_HHmmss')}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [messages]);

  const handleDelete = useCallback(async (id: string) => {
    setDeletingId(id);
    try {
      await api.deleteMessage(id);
      if (page === 0) {
        await refetch();
      } else {
        setPage(0);
      }
    } catch (deleteError) {
      console.error('Failed to delete message:', deleteError);
    } finally {
      setDeletingId(null);
    }
  }, [page, refetch]);

  const executeDelete = useCallback(() => {
    if (!confirmDeleteId) return;
    void handleDelete(confirmDeleteId);
    setConfirmDeleteId(null);
  }, [confirmDeleteId, handleDelete]);

  const handleBatchDelete = useCallback(async () => {
    setBatchDeleting(true);
    setConfirmBatchDelete(false);
    try {
      await api.deleteMessagesBatch(batchDeleteFilters);
      if (page === 0) {
        await refetch();
      } else {
        setPage(0);
      }
    } catch (deleteError) {
      console.error('Failed to batch delete messages:', deleteError);
    } finally {
      setBatchDeleting(false);
    }
  }, [batchDeleteFilters, page, refetch]);

  if (error) {
    return (
      <AppLayout>
        <div className="flex min-h-[400px] flex-col items-center justify-center space-y-4">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <div className="text-center">
            <h3 className="text-lg font-semibold">加载失败</h3>
            <p className="text-muted-foreground">{error}</p>
          </div>
          <Button onClick={refetch}>
            <RefreshCw className="mr-2 h-4 w-4" />
            重试
          </Button>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">历史记录</h2>
            <p className="text-muted-foreground">查看和管理所有对话历史记录</p>
          </div>
          <Button variant="ghost" size="icon" onClick={refetch} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        <HistoryFiltersCard
          searchTerm={searchTerm}
          platform={platformFilter}
          sessionType={sessionType}
          lora={selectedLora}
          sessionName={sessionNameFilter}
          availableLoraNames={availableLoraNames}
          onSearchTermChange={(value) => {
            setSearchTerm(value);
            setPage(0);
          }}
          onPlatformChange={(value) => {
            setPlatformFilter(value);
            setPage(0);
          }}
          onSessionTypeChange={(value) => {
            setSessionType(value);
            setPage(0);
          }}
          onLoraChange={(value) => {
            setSelectedLora(value);
            setPage(0);
          }}
          onSessionNameChange={(value) => {
            setSessionNameFilter(value);
            setPage(0);
          }}
          onExport={handleExport}
        />

        <HistoryMessagesCard
          messages={messages}
          total={total}
          totalAll={totalAll}
          loading={loading}
          page={page}
          totalPages={totalPages}
          deletingId={deletingId}
          batchDeleting={batchDeleting}
          onMessageOpen={setDetailMessage}
          onDeleteRequest={setConfirmDeleteId}
          onBatchDeleteRequest={() => setConfirmBatchDelete(true)}
          onPreviousPage={() => setPage((current) => Math.max(0, current - 1))}
          onNextPage={() => setPage((current) => current + 1)}
        />

        <BatchDeleteDialog
          open={confirmBatchDelete}
          total={total}
          filters={batchDeleteFilters}
          onOpenChange={setConfirmBatchDelete}
          onConfirm={handleBatchDelete}
        />
        <DeleteMessageDialog
          open={Boolean(confirmDeleteId)}
          onOpenChange={(open) => {
            if (!open) setConfirmDeleteId(null);
          }}
          onConfirm={executeDelete}
        />
        <MessageDetailDialog
          message={detailMessage}
          onOpenChange={(open) => {
            if (!open) setDetailMessage(null);
          }}
        />
      </div>
    </AppLayout>
  );
}
