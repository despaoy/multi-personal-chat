'use client';

import { useState, useEffect } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { Shuffle, RefreshCw, AlertCircle, Save, CheckCircle2, XCircle, ShieldCheck } from 'lucide-react';
import { useRouter, type RouterConfig } from '@/hooks/useRouter';
import { toast } from 'sonner';

export default function RouterPage() {
  return (
    <AuthGuard requireAdmin>
      <RouterContent />
    </AuthGuard>
  );
}

function RouterContent() {
  const { config, adapters, logs, loading, error, saving, checking, refetch, updateConfig, checkAdapter } = useRouter();
  const [editConfig, setEditConfig] = useState<RouterConfig | null>(null);

  useEffect(() => {
    if (config) setEditConfig({ ...config });
  }, [config]);

  if (error) {
    return (
      <AppLayout>
        <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
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

  const handleSave = async () => {
    if (!editConfig) return;
    try {
      await updateConfig({
        enabled: editConfig.enabled,
        default_adapter: editConfig.default_adapter,
        mode: editConfig.mode,
        persona_adapters: editConfig.persona_adapters,
        rag_confidence_threshold: editConfig.rag_confidence_threshold,
        persona_keywords: editConfig.persona_keywords,
      });
      toast.success('路由配置已保存');
    } catch {
      toast.error('保存配置失败');
    }
  };

  const handleCheck = async (name: string) => {
    try {
      await checkAdapter(name);
      toast.success(`适配器 ${name} 兼容性检查完成`);
    } catch {
      toast.error('兼容性检查失败');
    }
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Shuffle className="h-6 w-6" />
              多 LoRA 路由
            </h2>
            <p className="text-muted-foreground">路由配置 · 适配器兼容性 · 路由日志</p>
          </div>
          <Button variant="ghost" size="icon" onClick={refetch} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        {/* 配置卡片 */}
        <Card>
          <CardHeader>
            <CardTitle>路由配置</CardTitle>
            <CardDescription>控制多 LoRA 路由行为</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {loading || !editConfig ? (
              <Skeleton className="h-32" />
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>启用路由器</Label>
                    <p className="text-xs text-muted-foreground">关闭时所有查询使用默认适配器</p>
                  </div>
                  <Switch
                    checked={editConfig.enabled}
                    onCheckedChange={(v) => setEditConfig({ ...editConfig, enabled: v })}
                  />
                </div>
                <Separator />
                <div className="space-y-2">
                  <Label>路由模式</Label>
                  <Select
                    value={editConfig.mode || 'manual'}
                    onValueChange={(value: 'manual' | 'rule' | 'intent') => setEditConfig({ ...editConfig, mode: value })}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="manual">手动选择</SelectItem>
                      <SelectItem value="rule">关键词规则</SelectItem>
                      <SelectItem value="intent">意图路由</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Separator />
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>默认适配器</Label>
                    <p className="text-sm font-mono mt-1">{editConfig.default_adapter}</p>
                  </div>
                  <div>
                    <Label>RAG 置信度阈值</Label>
                    <p className="text-sm font-mono mt-1">{editConfig.rag_confidence_threshold}</p>
                  </div>
                </div>
                <Separator />
                <div>
                  <Label>角色关键词</Label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {Object.entries(editConfig.persona_keywords || {}).map(([persona, keywords]) => (
                      <div key={persona} className="flex items-center gap-1">
                        <Badge variant="outline">{persona}</Badge>
                        <span className="text-xs text-muted-foreground">{keywords.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <Separator />
                <div>
                  <Label>角色适配器映射</Label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {Object.entries(editConfig.persona_adapters || {}).map(([persona, adapter]) => (
                      <Badge key={persona} variant="secondary">{persona} → {adapter}</Badge>
                    ))}
                  </div>
                </div>
                <Button onClick={handleSave} disabled={saving}>
                  <Save className="mr-2 h-4 w-4" />
                  {saving ? '保存中...' : '保存配置'}
                </Button>
              </>
            )}
          </CardContent>
        </Card>

        <div className="grid gap-6 md:grid-cols-2">
          {/* 适配器列表 */}
          <Card>
            <CardHeader>
              <CardTitle>适配器列表</CardTitle>
              <CardDescription>{adapters.length} 个已注册适配器</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <Skeleton className="h-48" />
              ) : adapters.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">暂无适配器</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>名称</TableHead>
                      <TableHead>兼容性</TableHead>
                      <TableHead>操作</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {adapters.map((adp) => (
                      <TableRow key={adp.name}>
                        <TableCell className="font-mono text-xs">{adp.name}</TableCell>
                        <TableCell>
                          {adp.compatibility === null ? (
                            <Badge variant="outline">未检查</Badge>
                          ) : adp.compatibility.compatible ? (
                            <Badge className="bg-green-100 text-green-800">
                              <CheckCircle2 className="mr-1 h-3 w-3" />兼容
                            </Badge>
                          ) : (
                            <Badge className="bg-red-100 text-red-800">
                              <XCircle className="mr-1 h-3 w-3" />不兼容
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleCheck(adp.name)}
                            disabled={checking === adp.name}
                          >
                            <ShieldCheck className="mr-1 h-3 w-3" />
                            {checking === adp.name ? '检查中...' : '检查'}
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          {/* 路由日志 */}
          <Card>
            <CardHeader>
              <CardTitle>路由日志</CardTitle>
              <CardDescription>最近 {logs.length} 条路由决策</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <Skeleton className="h-48" />
              ) : logs.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">暂无路由日志</p>
              ) : (
                <div className="max-h-96 overflow-y-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>时间</TableHead>
                        <TableHead>目标</TableHead>
                        <TableHead>适配器</TableHead>
                        <TableHead>置信度</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {logs.map((log, i) => (
                        <TableRow key={i}>
                          <TableCell className="text-xs">{new Date(log.timestamp).toLocaleTimeString()}</TableCell>
                          <TableCell><Badge variant="outline">{log.target}</Badge></TableCell>
                          <TableCell className="font-mono text-xs">{log.adapter_name}</TableCell>
                          <TableCell className="font-mono text-xs">{log.confidence.toFixed(2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </AppLayout>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="text-sm font-medium">{children}</span>;
}
