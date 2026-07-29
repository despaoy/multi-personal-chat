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
import { FlaskConical, RefreshCw, AlertCircle, Play, Eye } from 'lucide-react';
import { useExperiments, type Experiment, type ExperimentType } from '@/hooks/useExperiments';
import type { JsonRecord } from '@/lib/api';
import { toast } from 'sonner';

const statusColors: Record<string, string> = {
  running: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
};

const EXPERIMENT_TABS: { key: ExperimentType; label: string; typeLabel: string }[] = [
  { key: 'lora-ablation', label: 'LoRA 消融', typeLabel: 'lora_ablation' },
  { key: 'rag-ablation', label: 'RAG 消融', typeLabel: 'rag_ablation' },
  { key: 'quantization-benchmark', label: '量化基准', typeLabel: 'quantization_benchmark' },
];

export default function ExperimentsPage() {
  return (
    <AuthGuard requireAdmin>
      <ExperimentsContent />
    </AuthGuard>
  );
}

function ExperimentsContent() {
  const { experiments, loading, error, starting, refetch, startExperiment, getReport } = useExperiments();
  const [activeTab, setActiveTab] = useState<string>('lora-ablation');
  const [mockMode, setMockMode] = useState(false);
  const [hypothesis, setHypothesis] = useState('');
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [detailExperiment, setDetailExperiment] = useState<Experiment | null>(null);
  const [reportContent, setReportContent] = useState<string | null>(null);

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

  const currentTab = EXPERIMENT_TABS.find(t => t.key === activeTab)!;
  const filteredExperiments = experiments.filter(e => e.experiment_type === currentTab.typeLabel);

  const handleStart = async () => {
    try {
      await startExperiment(currentTab.key, { hypothesis: hypothesis || undefined, mock: mockMode });
      toast.success(mockMode ? '实验已完成（mock 模式）' : '实验已启动');
      setStartDialogOpen(false);
      setHypothesis('');
    } catch {
      toast.error('启动实验失败');
    }
  };

  const handleViewDetail = async (exp: Experiment) => {
    setDetailExperiment(exp);
    setReportContent(null);
    try {
      const report = await getReport(exp.id);
      setReportContent(report.report || '');
    } catch {
      // 报告获取失败，仅展示结果
    }
  };

  const getComparisonRows = (results: JsonRecord | JsonRecord[]): JsonRecord[] => {
    const root = Array.isArray(results) ? undefined : results;
    const nested = root?.results;
    const candidate = root?.comparison_table ?? (typeof nested === 'object' && nested !== null && !Array.isArray(nested)
      ? (nested as JsonRecord).comparison_table
      : undefined);
    if (!Array.isArray(candidate)) return [];
    return candidate.filter((row): row is JsonRecord => typeof row === 'object' && row !== null && !Array.isArray(row));
  };

  const renderComparisonTable = (results: JsonRecord | JsonRecord[]) => {
    const table = getComparisonRows(results);
    if (table.length === 0) {
      return <p className="text-muted-foreground text-center py-4">无对比数据</p>;
    }
    const columns = Object.keys(table[0]);
    return (
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map(col => <TableHead key={col}>{col}</TableHead>)}
          </TableRow>
        </TableHeader>
        <TableBody>
          {table.map((row, i) => (
            <TableRow key={i}>
              {columns.map(col => (
                <TableCell key={col} className="font-mono text-xs">
                  {typeof row[col] === 'number' ? (Number.isInteger(row[col]) ? row[col] : row[col].toFixed(4)) : String(row[col])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <FlaskConical className="h-6 w-6" />
              实验对比
            </h2>
            <p className="text-muted-foreground">LoRA 消融 · RAG 消融 · 量化基准</p>
          </div>
          <Button variant="ghost" size="icon" onClick={refetch} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            {EXPERIMENT_TABS.map(tab => (
              <TabsTrigger key={tab.key} value={tab.key}>{tab.label}</TabsTrigger>
            ))}
          </TabsList>

          {EXPERIMENT_TABS.map(tab => (
            <TabsContent key={tab.key} value={tab.key} className="space-y-4">
              <div className="flex items-center gap-4">
                <Button onClick={() => { setActiveTab(tab.key); setStartDialogOpen(true); }} disabled={starting}>
                  <Play className="mr-2 h-4 w-4" />
                  {starting ? '启动中...' : `运行 ${tab.label}`}
                </Button>
                <div className="flex items-center gap-2">
                  <Switch checked={mockMode} onCheckedChange={setMockMode} />
                  <span className="text-sm text-muted-foreground">Mock 模式</span>
                </div>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>{tab.label} 实验列表</CardTitle>
                  <CardDescription>
                    {filteredExperiments.length} 条记录
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <Skeleton className="h-48" />
                  ) : filteredExperiments.length === 0 ? (
                    <p className="text-center text-muted-foreground py-8">暂无实验记录</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>实验 ID</TableHead>
                          <TableHead>状态</TableHead>
                          <TableHead>开始时间</TableHead>
                          <TableHead>完成时间</TableHead>
                          <TableHead>假设</TableHead>
                          <TableHead>操作</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredExperiments.map((exp) => (
                          <TableRow key={exp.id}>
                            <TableCell className="font-mono text-xs">
                              <div className="flex items-center gap-2">
                                <span>{exp.id}</span>
                                {!Array.isArray(exp.results) && exp.results?.mock === true && (
                                  <Badge variant="destructive">演示数据</Badge>
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              <Badge className={statusColors[exp.status] || 'bg-gray-100 text-gray-800'}>
                                {exp.status}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-xs">{new Date(exp.started_at).toLocaleString()}</TableCell>
                            <TableCell className="text-xs">{exp.completed_at ? new Date(exp.completed_at).toLocaleString() : '-'}</TableCell>
                            <TableCell className="max-w-xs truncate">{exp.hypothesis || '-'}</TableCell>
                            <TableCell>
                              <Button variant="ghost" size="sm" onClick={() => handleViewDetail(exp)}>
                                <Eye className="mr-1 h-3 w-3" />
                                详情
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          ))}
        </Tabs>

        {/* 启动实验对话框 */}
        <Dialog open={startDialogOpen} onOpenChange={setStartDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>运行 {currentTab?.label}</DialogTitle>
              <DialogDescription>配置实验参数</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="hypothesis">实验假设（可选）</Label>
                <Input
                  id="hypothesis"
                  value={hypothesis}
                  onChange={(e) => setHypothesis(e.target.value)}
                  placeholder="例如: DoRA 比 LoRA 在 perplexity 上更优"
                />
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={mockMode} onCheckedChange={setMockMode} />
                <span className="text-sm">Mock 模式（CPU 验证用）</span>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setStartDialogOpen(false)}>取消</Button>
              <Button onClick={handleStart} disabled={starting}>
                {starting ? '启动中...' : '启动'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* 实验详情对话框 */}
        <Dialog open={!!detailExperiment} onOpenChange={(open) => !open && setDetailExperiment(null)}>
          <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>实验详情: {detailExperiment?.id}</DialogTitle>
              <DialogDescription>
                {detailExperiment?.experiment_type} · {detailExperiment?.status}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div>
                <h4 className="text-sm font-semibold mb-2">对比表</h4>
                {detailExperiment?.results && renderComparisonTable(detailExperiment.results)}
              </div>
              {reportContent && (
                <div>
                  <h4 className="text-sm font-semibold mb-2">报告</h4>
                  <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto max-h-60">{reportContent}</pre>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </AppLayout>
  );
}
