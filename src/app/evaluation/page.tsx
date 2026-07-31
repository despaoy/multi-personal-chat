'use client';

import { useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import { ClipboardCheck, RefreshCw, AlertCircle, Play, ThumbsUp, ThumbsDown, MessageSquare } from 'lucide-react';
import { useEvaluation } from '@/hooks/useEvaluation';
import { toast } from 'sonner';

export default function EvaluationPage() {
  return (
    <AuthGuard requireAdmin>
      <EvaluationContent />
    </AuthGuard>
  );
}

function EvaluationContent() {
  const {
    goldSet,
    runs,
    runsTotal,
    feedbacks,
    feedbackTotal,
    loading,
    error,
    running,
    refetch,
    runEvaluation,
  } = useEvaluation();
  const [runDialogOpen, setRunDialogOpen] = useState(false);
  const [mockMode, setMockMode] = useState(false);
  const [adapterName, setAdapterName] = useState('');

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

  const handleRun = async () => {
    try {
      await runEvaluation({ adapter_name: adapterName || undefined, mock: mockMode });
      toast.success(mockMode ? '演示评估任务已加入队列' : '真实评估任务已加入队列');
      setRunDialogOpen(false);
    } catch {
      toast.error('评估运行失败');
    }
  };

  const categories = Object.entries(goldSet.category_breakdown);

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <ClipboardCheck className="h-6 w-6" />
              评估仪表盘
            </h2>
            <p className="text-muted-foreground">Gold 评估集 · 评估运行 · 用户反馈</p>
          </div>
          <Button variant="ghost" size="icon" onClick={refetch} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        <Tabs defaultValue="gold">
          <TabsList>
            <TabsTrigger value="gold">Gold 集</TabsTrigger>
            <TabsTrigger value="runs">评估运行</TabsTrigger>
            <TabsTrigger value="feedback">反馈</TabsTrigger>
          </TabsList>

          {/* Gold 集标签 */}
          <TabsContent value="gold" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>总数</CardDescription>
                  <CardTitle className="text-3xl">{goldSet.total}</CardTitle>
                </CardHeader>
              </Card>
              {categories.map(([cat, count]) => (
                <Card key={cat}>
                  <CardHeader className="pb-2">
                    <CardDescription>{cat}</CardDescription>
                    <CardTitle className="text-3xl">{count}</CardTitle>
                  </CardHeader>
                </Card>
              ))}
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Gold 评估提示词</CardTitle>
                <CardDescription>共 {goldSet.total} 条提示词</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <Skeleton className="h-64" />
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>ID</TableHead>
                        <TableHead>类别</TableHead>
                        <TableHead>分割</TableHead>
                        <TableHead>提示词</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {goldSet.prompts.slice(0, 50).map((p) => (
                        <TableRow key={p.id}>
                          <TableCell className="font-mono text-xs">{p.id}</TableCell>
                          <TableCell><Badge variant="outline">{p.category}</Badge></TableCell>
                          <TableCell>{p.split || '-'}</TableCell>
                          <TableCell className="max-w-md truncate">{p.prompt}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* 评估运行标签 */}
          <TabsContent value="runs" className="space-y-4">
            <div className="flex items-center gap-4">
              <Button onClick={() => setRunDialogOpen(true)} disabled={running}>
                <Play className="mr-2 h-4 w-4" />
                {running ? '运行中...' : '运行评估'}
              </Button>
              <div className="flex items-center gap-2">
                <Switch checked={mockMode} onCheckedChange={setMockMode} />
                <span className="text-sm text-muted-foreground">Mock 模式</span>
              </div>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>评估运行历史</CardTitle>
                <CardDescription>共 {runsTotal} 次评估运行，当前显示 {runs.length} 条</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <Skeleton className="h-48" />
                ) : runs.length === 0 ? (
                  <p className="text-center text-muted-foreground py-8">暂无评估运行记录</p>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>运行 ID</TableHead>
                        <TableHead>类型</TableHead>
                        <TableHead>时间</TableHead>
                        <TableHead>适配器</TableHead>
                        <TableHead>提示词数</TableHead>
                        <TableHead>指标摘要</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {runs.map((r) => (
                        <TableRow key={r.id}>
                          <TableCell className="font-mono text-xs">{r.id}</TableCell>
                          <TableCell>
                            {r.metrics?.mock === true ? (
                              <Badge className="bg-amber-100 text-amber-900">演示数据</Badge>
                            ) : (
                              <Badge variant="outline">真实运行</Badge>
                            )}
                          </TableCell>
                          <TableCell className="text-xs">{new Date(r.run_at).toLocaleString()}</TableCell>
                          <TableCell>{r.adapter_name || '-'}</TableCell>
                          <TableCell>{r.total_prompts}</TableCell>
                          <TableCell className="text-xs">
                            {Object.entries(r.metrics || {}).slice(0, 3).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : String(v)}`).join(', ')}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* 反馈标签 */}
          <TabsContent value="feedback" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>用户反馈</CardTitle>
                <CardDescription>共 {feedbackTotal} 条反馈，当前显示 {feedbacks.length} 条</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <Skeleton className="h-48" />
                ) : feedbacks.length === 0 ? (
                  <p className="text-center text-muted-foreground py-8">暂无用户反馈</p>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Trace ID</TableHead>
                        <TableHead>评分</TableHead>
                        <TableHead>原因</TableHead>
                        <TableHead>适配器</TableHead>
                        <TableHead>时间</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {feedbacks.map((f, i) => (
                        <TableRow key={i}>
                          <TableCell className="font-mono text-xs">{f.trace_id || '-'}</TableCell>
                          <TableCell>
                            {f.rating === 'positive' || f.rating === 'thumbs_up' ? (
                              <Badge className="bg-green-100 text-green-800"><ThumbsUp className="mr-1 h-3 w-3" />正面</Badge>
                            ) : f.rating === 'negative' || f.rating === 'thumbs_down' ? (
                              <Badge className="bg-red-100 text-red-800"><ThumbsDown className="mr-1 h-3 w-3" />负面</Badge>
                            ) : (
                              <Badge variant="outline"><MessageSquare className="mr-1 h-3 w-3" />{f.rating}</Badge>
                            )}
                          </TableCell>
                          <TableCell className="max-w-xs truncate">{f.reason || '-'}</TableCell>
                          <TableCell>{f.adapter_name || '-'}</TableCell>
                          <TableCell className="text-xs">{new Date(f.created_at).toLocaleString()}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* 运行评估对话框 */}
        <Dialog open={runDialogOpen} onOpenChange={setRunDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>运行评估</DialogTitle>
              <DialogDescription>配置评估参数并启动</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="adapter">适配器名称（可选）</Label>
                <Input
                  id="adapter"
                  value={adapterName}
                  onChange={(e) => setAdapterName(e.target.value)}
                  placeholder="例如: hutao_lora_7b"
                />
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={mockMode} onCheckedChange={setMockMode} />
                <span className="text-sm">Mock 模式（CPU 验证用）</span>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setRunDialogOpen(false)}>取消</Button>
              <Button onClick={handleRun} disabled={running}>
                {running ? '运行中...' : '启动'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </AppLayout>
  );
}
