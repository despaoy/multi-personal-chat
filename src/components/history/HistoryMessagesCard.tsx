'use client';

import { format } from 'date-fns';
import { Calendar, Eye, Trash2 } from 'lucide-react';

import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { Message } from '@/lib/api';

interface HistoryMessagesCardProps {
  messages: Message[];
  total: number;
  totalAll: number;
  loading: boolean;
  page: number;
  totalPages: number;
  deletingId: string | null;
  batchDeleting: boolean;
  onMessageOpen: (message: Message) => void;
  onDeleteRequest: (id: string) => void;
  onBatchDeleteRequest: () => void;
  onPreviousPage: () => void;
  onNextPage: () => void;
}

export function HistoryMessagesCard({
  messages,
  total,
  totalAll,
  loading,
  page,
  totalPages,
  deletingId,
  batchDeleting,
  onMessageOpen,
  onDeleteRequest,
  onBatchDeleteRequest,
  onPreviousPage,
  onNextPage,
}: HistoryMessagesCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>对话记录</CardTitle>
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">
            共 {totalAll} 条记录（当前筛选 {total} 条）
          </span>
          <Button
            variant="destructive"
            size="sm"
            disabled={batchDeleting || total === 0}
            onClick={onBatchDeleteRequest}
          >
            {batchDeleting ? '删除中...' : '删除全部'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map((item) => (
              <div key={item} className="flex items-center gap-4">
                <Skeleton className="h-8 w-8 rounded-full" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>时间</TableHead>
                <TableHead>用户</TableHead>
                <TableHead>会话</TableHead>
                <TableHead>消息预览</TableHead>
                <TableHead>LoRA模型</TableHead>
                <TableHead>耗时</TableHead>
                <TableHead>操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {messages.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="h-24 text-center">
                    <div className="text-muted-foreground">暂无记录</div>
                  </TableCell>
                </TableRow>
              ) : (
                messages.map((message) => (
                  <TableRow key={message.id}>
                    <TableCell className="text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {format(new Date(message.createdAt), 'yyyy-MM-dd HH:mm:ss')}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Avatar className="h-8 w-8">
                          <AvatarImage src="" />
                          <AvatarFallback>{message.userName?.trim().charAt(0) || message.senderId?.trim().charAt(0) || '?'}</AvatarFallback>
                        </Avatar>
                        <span className="font-medium">{message.userName?.trim() || message.senderId?.trim() || '未知用户'}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={message.sessionType === 'group' ? 'default' : 'secondary'}>
                        {message.sessionType === 'group' ? '群聊' : '私聊'}
                      </Badge>
                      <div className="mt-1 text-sm text-muted-foreground">{message.sessionName}</div>
                    </TableCell>
                    <TableCell className="max-w-[200px]">
                      <p className="truncate text-sm">{message.message}</p>
                    </TableCell>
                    <TableCell>
                      <div className="text-sm">
                        {message.loraName === 'default'
                          ? '基础模型（无LoRA）'
                          : (message.loraName || message.modelName || '-')}
                      </div>
                      {message.loraName && message.loraName !== 'default' && message.modelName && (
                        <div className="text-xs text-muted-foreground">{message.modelName}</div>
                      )}
                    </TableCell>
                    <TableCell className="text-sm">{message.costTime}s</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        <Button variant="ghost" size="icon" onClick={() => onMessageOpen(message)}>
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => onDeleteRequest(String(message.id))}
                          disabled={deletingId === String(message.id)}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        )}
        <div className="mt-4 flex items-center justify-end gap-3">
          <span className="text-sm text-muted-foreground">
            第 {page + 1} / {totalPages} 页
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={loading || page === 0}
            onClick={onPreviousPage}
          >
            上一页
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={loading || page + 1 >= totalPages}
            onClick={onNextPage}
          >
            下一页
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
