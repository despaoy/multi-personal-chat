'use client';

/**
 * 测试对话对话框
 *
 * 从 DashboardClient 拆分而来，隔离聊天输入状态，
 * 避免输入消息时触发整个仪表盘重渲染。
 */

import { useState, useEffect, useRef } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { MessageSquare, Send, Trash2, Bot, User } from 'lucide-react';
import { api, type LoraModel } from '@/lib/api';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  loraName?: string;
  costTime?: number;
}

export function TestChatDialog({ loras }: { loras: LoraModel[] }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedLora, setSelectedLora] = useState('');
  const [sessionType, setSessionType] = useState<'private' | 'group'>('private');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const requestedLora = selectedLora && selectedLora !== 'default'
        ? selectedLora
        : undefined;
      const response = await api.generateReply({
        message: input,
        sessionType,
        sessionId: sessionType === 'group' ? 'test-group' : 'test-private',
        userId: 'dashboard-user',
        userName: sessionType === 'group' ? '群成员' : '测试用户',
        loraName: requestedLora,
      });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.reply,
        loraName: selectedLora || 'default',
        costTime: response.costTime,
      }]);
    } catch (err) {
      console.error('Failed to generate reply:', err);
      setMessages(prev => [...prev, { role: 'assistant', content: '生成失败，请稍后重试' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" className="flex flex-col items-center justify-center rounded-lg border p-4 h-auto hover:bg-muted transition-colors">
          <MessageSquare className="h-6 w-6 mb-2 text-primary" />
          <span className="text-sm font-medium">测试回复</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[550px] max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>测试机器人回复</DialogTitle>
          <DialogDescription>
            模拟私聊或群聊场景，测试不同LoRA的回复效果
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
          {/* 配置栏 */}
          <div className="flex gap-2">
            <Select value={sessionType} onValueChange={(v) => setSessionType(v as 'private' | 'group')}>
              <SelectTrigger className="w-[100px] h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="private">私聊</SelectItem>
                <SelectItem value="group">群聊</SelectItem>
              </SelectContent>
            </Select>
            <Select value={selectedLora} onValueChange={setSelectedLora}>
              <SelectTrigger className="flex-1 h-8 text-xs">
                <SelectValue placeholder="选择LoRA模型" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">基础模型（无LoRA）</SelectItem>
                {loras.map((lora) => (
                  <SelectItem key={lora.id} value={lora.name}>
                    {lora.name} {lora.status === 'active' ? '(激活)' : ''}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              onClick={() => setMessages([])}
              title="清空对话"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>

          <Separator />

          {/* 聊天区域 */}
          <ScrollArea className="h-[300px] rounded-lg border p-3">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                发送一条消息开始测试
              </div>
            ) : (
              <div className="space-y-3">
                {messages.map((msg, i) => (
                  <div key={i} className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role === 'assistant' && (
                      <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-purple-100 text-purple-700">
                        <Bot className="h-3.5 w-3.5" />
                      </div>
                    )}
                    <div className={`max-w-[75%] rounded-lg p-2.5 text-sm ${
                      msg.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted'
                    }`}>
                      <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                      {msg.role === 'assistant' && msg.costTime && (
                        <p className="text-[10px] text-muted-foreground mt-1">
                          {msg.loraName || 'default'} · {msg.costTime}s
                        </p>
                      )}
                    </div>
                    {msg.role === 'user' && (
                      <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-blue-100 text-blue-700">
                        <User className="h-3.5 w-3.5" />
                      </div>
                    )}
                  </div>
                ))}
                {loading && (
                  <div className="flex gap-2 justify-start">
                    <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-purple-100 text-purple-700">
                      <Bot className="h-3.5 w-3.5" />
                    </div>
                    <div className="bg-muted rounded-lg p-2.5 text-sm text-muted-foreground">
                      正在思考...
                    </div>
                  </div>
                )}
                <div ref={endRef} />
              </div>
            )}
          </ScrollArea>

          {/* 输入区域 */}
          <div className="flex gap-2">
            <Input
              placeholder={sessionType === 'group' ? '在群聊中发送消息...' : '发送消息...'}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={loading}
            />
            <Button size="icon" onClick={handleSend} disabled={loading || !input.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
