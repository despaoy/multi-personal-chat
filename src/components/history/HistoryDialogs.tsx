'use client';

import { format } from 'date-fns';
import { Bot, Eye, User } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Separator } from '@/components/ui/separator';
import type { Message, MessageFilters } from '@/lib/api';

interface BatchDeleteDialogProps {
  open: boolean;
  total: number;
  filters: MessageFilters;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}

export function BatchDeleteDialog({
  open,
  total,
  filters,
  onOpenChange,
  onConfirm,
}: BatchDeleteDialogProps) {
  const hasFilters = Boolean(
    filters.search
      || (filters.platform && filters.platform !== 'all')
      || (filters.sessionType && filters.sessionType !== 'all')
      || (filters.lora && filters.lora !== 'all')
      || filters.sessionName,
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[450px]">
        <DialogHeader>
          <DialogTitle>确认批量删除</DialogTitle>
        </DialogHeader>
        <div className="space-y-2">
          <p className="text-muted-foreground">
            将删除当前筛选条件下的 <strong>{total}</strong> 条对话记录，此操作不可撤销。
          </p>
          {hasFilters && (
            <div className="space-y-0.5 rounded-md bg-muted p-2 text-xs text-muted-foreground">
              <p>当前筛选条件：</p>
              {filters.search && <p>· 搜索：{filters.search}</p>}
              {filters.platform && filters.platform !== 'all' && <p>· 平台：{filters.platform}</p>}
              {filters.sessionType && filters.sessionType !== 'all' && (
                <p>· 类型：{filters.sessionType === 'group' ? '群聊' : '私聊'}</p>
              )}
              {filters.lora && filters.lora !== 'all' && <p>· LoRA：{filters.lora}</p>}
              {filters.sessionName && <p>· 会话名：{filters.sessionName}</p>}
            </div>
          )}
        </div>
        <div className="mt-4 flex justify-end gap-3">
          <Button variant="outline" onClick={() => onOpenChange(false)}>取消</Button>
          <Button variant="destructive" onClick={onConfirm}>确认删除</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

interface DeleteMessageDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}

export function DeleteMessageDialog({
  open,
  onOpenChange,
  onConfirm,
}: DeleteMessageDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle>确认删除</DialogTitle>
        </DialogHeader>
        <p className="text-muted-foreground">确定要删除这条记录吗？此操作不可撤销。</p>
        <div className="mt-4 flex justify-end gap-3">
          <Button variant="outline" onClick={() => onOpenChange(false)}>取消</Button>
          <Button variant="destructive" onClick={onConfirm}>删除</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

interface MessageDetailDialogProps {
  message: Message | null;
  onOpenChange: (open: boolean) => void;
}

export function MessageDetailDialog({
  message,
  onOpenChange,
}: MessageDetailDialogProps) {
  return (
    <Dialog open={Boolean(message)} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] flex-col sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            对话详情
          </DialogTitle>
        </DialogHeader>
        {message && (
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto">
            <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
              <Badge variant="outline">{(message.platform || 'qq').toUpperCase()}</Badge>
              <Badge variant={message.sessionType === 'group' ? 'default' : 'secondary'}>
                {message.sessionType === 'group' ? '群聊' : '私聊'}
              </Badge>
              <span>{message.sessionName}</span>
              <span>·</span>
              <span>{format(new Date(message.createdAt), 'yyyy-MM-dd HH:mm:ss')}</span>
              <span>·</span>
              <span>{message.modelName}</span>
              {message.loraName && message.loraName !== 'default' && (
                <>
                  <span>·</span>
                  <span>LoRA: {message.loraName}</span>
                </>
              )}
              <span>·</span>
              <span>耗时 {message.costTime}s</span>
            </div>

            <Separator />

            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-blue-700">
                <User className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="mb-1 text-xs font-medium text-muted-foreground">
                  {message.userName}
                </div>
                <div className="break-words whitespace-pre-wrap rounded-lg bg-blue-50 p-3 text-sm dark:bg-blue-950/20">
                  {message.message}
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-purple-100 text-purple-700">
                <Bot className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="mb-1 text-xs font-medium text-muted-foreground">
                  {message.modelName}
                </div>
                <div className="break-words whitespace-pre-wrap rounded-lg bg-purple-50 p-3 text-sm dark:bg-purple-950/20">
                  {message.reply}
                </div>
              </div>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
