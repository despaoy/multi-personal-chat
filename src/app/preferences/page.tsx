'use client';

import { useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Scale, RefreshCw, AlertCircle, Plus, Download, History, Eye } from 'lucide-react';
import { usePreferences, type PreferencePair } from '@/hooks/usePreferences';
import type { PreferenceCandidate, PreferenceReviewStatus } from '@/lib/api';
import { toast } from 'sonner';

const statusColors: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  approved: 'bg-green-100 text-green-800',
  rejected: 'bg-red-100 text-red-800',
};

export default function PreferencesPage() {
  return (
    <AuthGuard requireAdmin>
      <PreferencesContent />
    </AuthGuard>
  );
}

function PreferencesContent() {
  const {
    preferences, total, loading, error, filterStatus, exporting, sampling,
    refetch, createPreference, updatePreference, exportPreferences, sampleFromHistory, setFilterStatus,
  } = usePreferences();
  const [createOpen, setCreateOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [sampleOpen, setSampleOpen] = useState(false);
  const [detailPair, setDetailPair] = useState<PreferencePair | null>(null);
  const [exportStatus, setExportStatus] = useState<PreferenceReviewStatus>('approved');
  const [sampleLimit, setSampleLimit] = useState(20);
  const [sampleSessionId, setSampleSessionId] = useState('');
  const [createForm, setCreateForm] = useState({ prompt: '', chosen: '', rejected: '', annotator: '' });
  const [sampleCandidates, setSampleCandidates] = useState<PreferenceCandidate[] | null>(null);

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

  const handleCreate = async () => {
    if (!createForm.prompt || !createForm.chosen || !createForm.rejected) {
      toast.error('请填写 prompt、chosen 和 rejected');
      return;
    }
    try {
      await createPreference({
        prompt: createForm.prompt,
        chosen: createForm.chosen,
        rejected: createForm.rejected,
        annotator: createForm.annotator || undefined,
      });
      toast.success('偏好对已创建');
      setCreateOpen(false);
      setCreateForm({ prompt: '', chosen: '', rejected: '', annotator: '' });
    } catch {
      toast.error('创建失败');
    }
  };

  const handleExport = async () => {
    try {
      const result = await exportPreferences({ review_status: exportStatus, format: 'jsonl' });
      const blob = new Blob([JSON.stringify(result.data || [], null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `preferences_${exportStatus}.jsonl`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`已导出 ${result.count} 条偏好对`);
      setExportOpen(false);
    } catch {
      toast.error('导出失败');
    }
  };

  const handleSample = async () => {
    try {
      const result = await sampleFromHistory({ limit: sampleLimit, session_id: sampleSessionId || undefined });
      setSampleCandidates(result.candidates || []);
      toast.success(`已采样 ${result.total} 条候选`);
    } catch {
      toast.error('采样失败');
    }
  };

  const handleStatusChange = async (id: string, newStatus: 'pending' | 'approved' | 'rejected') => {
    try {
      await updatePreference(id, { review_status: newStatus });
      toast.success('状态已更新');
    } catch {
      toast.error('更新失败');
    }
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Scale className="h-6 w-6" />
              偏好数据管理
            </h2>
            <p className="text-muted-foreground">DPO/ORPO 训练数据 · 共 {total} 条</p>
          </div>
          <Button variant="ghost" size="icon" onClick={refetch} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        {/* 操作栏 */}
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            新建偏好对
          </Button>
          <Button variant="outline" onClick={() => setExportOpen(true)} disabled={exporting}>
            <Download className="mr-2 h-4 w-4" />
            {exporting ? '导出中...' : '导出 JSONL'}
          </Button>
          <Button variant="outline" onClick={() => { setSampleOpen(true); setSampleCandidates(null); }} disabled={sampling}>
            <History className="mr-2 h-4 w-4" />
            {sampling ? '采样中...' : '从历史采样'}
          </Button>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-sm text-muted-foreground">过滤:</span>
            <Select value={filterStatus || 'all'} onValueChange={(v) => setFilterStatus(v === 'all' ? '' : v as PreferenceReviewStatus)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部</SelectItem>
                <SelectItem value="pending">待审核</SelectItem>
                <SelectItem value="approved">已批准</SelectItem>
                <SelectItem value="rejected">已拒绝</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* 偏好对列表 */}
        <Card>
          <CardHeader>
            <CardTitle>偏好对列表</CardTitle>
            <CardDescription>{preferences.length} 条（当前过滤）</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-64" />
            ) : preferences.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">暂无偏好对</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Prompt</TableHead>
                    <TableHead>Chosen</TableHead>
                    <TableHead>Rejected</TableHead>
                    <TableHead>状态</TableHead>
                    <TableHead>时间</TableHead>
                    <TableHead>操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {preferences.map((p) => (
                    <TableRow key={p.id}>
                      <TableCell className="max-w-[160px] truncate text-xs">{p.prompt}</TableCell>
                      <TableCell className="max-w-[160px] truncate text-xs text-green-700">{p.chosen}</TableCell>
                      <TableCell className="max-w-[160px] truncate text-xs text-red-700">{p.rejected}</TableCell>
                      <TableCell>
                        <Select
                          value={p.review_status}
                          onValueChange={(v) => handleStatusChange(p.id, v as 'pending' | 'approved' | 'rejected')}
                        >
                          <SelectTrigger className="w-28 h-7 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="pending">待审核</SelectItem>
                            <SelectItem value="approved">已批准</SelectItem>
                            <SelectItem value="rejected">已拒绝</SelectItem>
                          </SelectContent>
                        </Select>
                      </TableCell>
                      <TableCell className="text-xs">{new Date(p.created_at).toLocaleDateString()}</TableCell>
                      <TableCell>
                        <Button variant="ghost" size="sm" onClick={() => setDetailPair(p)}>
                          <Eye className="h-3 w-3" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        {/* 新建偏好对对话框 */}
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>新建偏好对</DialogTitle>
              <DialogDescription>用于 DPO/ORPO 训练</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-2">
                <Label htmlFor="prompt">Prompt</Label>
                <Textarea id="prompt" value={createForm.prompt} onChange={(e) => setCreateForm({ ...createForm, prompt: e.target.value })} rows={3} placeholder="用户提问内容..." />
              </div>
              <div className="space-y-2">
                <Label htmlFor="chosen">Chosen（优选回复）</Label>
                <Textarea id="chosen" value={createForm.chosen} onChange={(e) => setCreateForm({ ...createForm, chosen: e.target.value })} rows={3} placeholder="期望的回复..." />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rejected">Rejected（拒绝回复）</Label>
                <Textarea id="rejected" value={createForm.rejected} onChange={(e) => setCreateForm({ ...createForm, rejected: e.target.value })} rows={3} placeholder="不期望的回复..." />
              </div>
              <div className="space-y-2">
                <Label htmlFor="annotator">标注者（可选）</Label>
                <Input id="annotator" value={createForm.annotator} onChange={(e) => setCreateForm({ ...createForm, annotator: e.target.value })} placeholder="标注者名称" />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateOpen(false)}>取消</Button>
              <Button onClick={handleCreate}>创建</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* 导出对话框 */}
        <Dialog open={exportOpen} onOpenChange={setExportOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>导出偏好数据</DialogTitle>
              <DialogDescription>选择要导出的审核状态</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-2">
                <Label>审核状态</Label>
                <Select value={exportStatus} onValueChange={(value) => setExportStatus(value as PreferenceReviewStatus)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="approved">已批准</SelectItem>
                    <SelectItem value="pending">待审核</SelectItem>
                    <SelectItem value="rejected">已拒绝</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setExportOpen(false)}>取消</Button>
              <Button onClick={handleExport} disabled={exporting}>{exporting ? '导出中...' : '导出'}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* 采样对话框 */}
        <Dialog open={sampleOpen} onOpenChange={setSampleOpen}>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>从历史采样</DialogTitle>
              <DialogDescription>从消息历史中随机采样候选偏好对</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="sampleLimit">采样数量</Label>
                  <Input id="sampleLimit" type="number" value={sampleLimit} onChange={(e) => setSampleLimit(Number(e.target.value))} min={1} max={100} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="sampleSession">会话 ID（可选）</Label>
                  <Input id="sampleSession" value={sampleSessionId} onChange={(e) => setSampleSessionId(e.target.value)} placeholder="留空采样全部" />
                </div>
              </div>
              <Button onClick={handleSample} disabled={sampling}>
                <History className="mr-2 h-4 w-4" />
                {sampling ? '采样中...' : '执行采样'}
              </Button>
              {sampleCandidates && sampleCandidates.length > 0 && (
                <div className="space-y-2">
                  <Label>候选结果（{sampleCandidates.length} 条）</Label>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {sampleCandidates.map((c, i) => (
                      <div key={i} className="border rounded p-2 text-xs">
                        <p className="font-medium">Q: {c.prompt}</p>
                        <p className="text-muted-foreground mt-1">A: {c.response}</p>
                        <Badge variant="outline" className="mt-1">{c.lora_name}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* 详情对话框 */}
        <Dialog open={!!detailPair} onOpenChange={(open) => !open && setDetailPair(null)}>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>偏好对详情</DialogTitle>
              <DialogDescription>{detailPair?.id}</DialogDescription>
            </DialogHeader>
            {detailPair && (
              <div className="space-y-4 py-2 text-sm">
                <div>
                  <Label>Prompt</Label>
                  <p className="mt-1 p-2 bg-muted rounded">{detailPair.prompt}</p>
                </div>
                <div>
                  <Label>Chosen</Label>
                  <p className="mt-1 p-2 bg-green-50 rounded text-green-900">{detailPair.chosen}</p>
                </div>
                <div>
                  <Label>Rejected</Label>
                  <p className="mt-1 p-2 bg-red-50 rounded text-red-900">{detailPair.rejected}</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>标注者</Label>
                    <p className="mt-1">{detailPair.annotator || '-'}</p>
                  </div>
                  <div>
                    <Label>状态</Label>
                    <Badge className={statusColors[detailPair.review_status] || ''}>{detailPair.review_status}</Badge>
                  </div>
                </div>
                {Object.keys(detailPair.rubric || {}).length > 0 && (
                  <div>
                    <Label>评分标准</Label>
                    <pre className="mt-1 p-2 bg-muted rounded text-xs overflow-x-auto">{JSON.stringify(detailPair.rubric, null, 2)}</pre>
                  </div>
                )}
                {Object.keys(detailPair.metadata || {}).length > 0 && (
                  <div>
                    <Label>元数据</Label>
                    <pre className="mt-1 p-2 bg-muted rounded text-xs overflow-x-auto">{JSON.stringify(detailPair.metadata, null, 2)}</pre>
                  </div>
                )}
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </AppLayout>
  );
}
