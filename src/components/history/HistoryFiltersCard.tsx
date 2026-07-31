'use client';

import { Download, Search } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface HistoryFiltersCardProps {
  searchTerm: string;
  platform: string;
  sessionType: string;
  lora: string;
  sessionName: string;
  availableLoraNames: string[];
  onSearchTermChange: (value: string) => void;
  onPlatformChange: (value: string) => void;
  onSessionTypeChange: (value: string) => void;
  onLoraChange: (value: string) => void;
  onSessionNameChange: (value: string) => void;
  onExport: () => void;
}

export function HistoryFiltersCard({
  searchTerm,
  platform,
  sessionType,
  lora,
  sessionName,
  availableLoraNames,
  onSearchTermChange,
  onPlatformChange,
  onSessionTypeChange,
  onLoraChange,
  onSessionNameChange,
  onExport,
}: HistoryFiltersCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>筛选条件</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4 md:flex-row">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="搜索消息内容..."
                value={searchTerm}
                onChange={(event) => onSearchTermChange(event.target.value)}
                className="pl-8"
              />
            </div>
          </div>
          <Select value={platform} onValueChange={onPlatformChange}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="平台" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部平台</SelectItem>
              <SelectItem value="qq">QQ</SelectItem>
              <SelectItem value="telegram">Telegram</SelectItem>
              <SelectItem value="wecom">企业微信</SelectItem>
              <SelectItem value="wechat_official">微信公众号</SelectItem>
              <SelectItem value="wechat_personal">个人微信</SelectItem>
            </SelectContent>
          </Select>
          <Select value={sessionType} onValueChange={onSessionTypeChange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="会话类型" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部</SelectItem>
              <SelectItem value="group">群聊</SelectItem>
              <SelectItem value="private">私聊</SelectItem>
            </SelectContent>
          </Select>
          <Select value={lora} onValueChange={onLoraChange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="LoRA模型" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部</SelectItem>
              {availableLoraNames.map((name) => (
                <SelectItem key={name} value={name}>
                  {name === 'default' ? '基础模型（无LoRA）' : name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            placeholder="群聊名/私聊名称..."
            value={sessionName}
            onChange={(event) => onSessionNameChange(event.target.value)}
            className="w-[200px]"
          />
          <Button onClick={onExport}>
            <Download className="mr-2 h-4 w-4" />
            导出当前页
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
