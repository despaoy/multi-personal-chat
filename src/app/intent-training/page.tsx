'use client';

import { useState, useEffect, useCallback } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import {
  Brain, Play, XCircle, RefreshCw, Loader2, Database,
  CheckCircle2, Info, FileText, Plus, Pencil, Trash2, X, Check,
  SquareCheck, Square, FlipHorizontal2, Trash, ShieldCheck,
} from 'lucide-react';
import { api, type KnowledgeBase } from '@/lib/api';
import type {
  ActiveKnowledgeBase,
  IntentModelInfo,
} from '@/lib/api-contracts';
import { toast } from 'sonner';

const MAX_KB_COUNT = 8;

export default function IntentTrainingPage() {
  return (
    <AuthGuard requireAdmin>
      <IntentTrainingContent />
    </AuthGuard>
  );
}

function IntentTrainingContent() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKbIds, setSelectedKbIds] = useState<number[]>([]);
  const [activeKbIds, setActiveKbIds] = useState<number[]>([]);
  const [samplesPerKb, setSamplesPerKb] = useState(100);
  const [negativeCount, setNegativeCount] = useState(200);

  // 样本生成状态
  const [generating, setGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genMessage, setGenMessage] = useState('');

  // 训练状态
  const [training, setTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);
  const [trainStage, setTrainStage] = useState('');
  const [trainMessage, setTrainMessage] = useState('');
  const [logs, setLogs] = useState<string[]>([]);

  // 样本数据
  const [samples, setSamples] = useState<Record<string, string[]>>({});
  const [sampleStats, setSampleStats] = useState<Record<string, number>>({});
  const [activeTab, setActiveTab] = useState('config');

  // 编辑状态
  const [editingLabel, setEditingLabel] = useState<string | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editingText, setEditingText] = useState('');

  // 添加样本
  const [addingLabel, setAddingLabel] = useState<string | null>(null);
  const [addingText, setAddingText] = useState('');

  // 批量选择状态: Record<label, Set<index>>
  const [selectedMap, setSelectedMap] = useState<Record<string, Set<number>>>({});

  // 模型信息
  const [modelInfo, setModelInfo] = useState<IntentModelInfo | null>(null);

  // 加载数据（仅在挂载时执行一次，避免依赖循环导致无限重渲染/compiling）
  const loadKnowledgeBases = useCallback(async () => {
    try {
      const res = await api.getKnowledgeBases();
      setKnowledgeBases(res.bases || []);
    } catch { /* ignore */ }
  }, []);

  const loadActiveKbs = useCallback(async () => {
    try {
      const res = await api.getActiveKnowledgeBases();
      const ids = (res.active_kbs || [])
        .map((kb: ActiveKnowledgeBase) => {
          if ('id' in kb) return kb.id;
          if (!kb.isActive) return -1;
          return knowledgeBases.find((base) => base.name === kb.kbName)?.id ?? -1;
        })
        .filter((id) => id > 0);
      setActiveKbIds(ids);
    } catch { /* ignore */ }
  }, [knowledgeBases]);

  const loadModelInfo = useCallback(async () => {
    try {
      const res = await api.getIntentModelInfo();
      setModelInfo(res.model);
    } catch { /* ignore */ }
  }, []);

  const loadSamples = useCallback(async () => {
    try {
      const res = await api.getIntentSamples();
      setSamples(res.samples || {});
      setSampleStats(res.stats || {});
    } catch { /* ignore */ }
  }, []);

  // 仅挂载时加载一次数据，selectedKbIds 通过用户点击 toggleKb 管理，
  // 不再由 loadActiveKbs 自动覆盖（避免选中后立即被 API 返回值清空）
  useEffect(() => {
    loadKnowledgeBases();
    loadModelInfo();
    loadSamples();
  }, [loadKnowledgeBases, loadModelInfo, loadSamples]);

  // knowledgeBases 加载完成后，异步获取活跃 KB 列表（不影响 selectedKbIds）
  useEffect(() => {
    if (knowledgeBases.length > 0) {
      loadActiveKbs();
    }
  }, [knowledgeBases, loadActiveKbs]);

  // 轮询生成进度
  useEffect(() => {
    if (!generating) return;
    const timer = setInterval(async () => {
      try {
        const res = await api.getGenerationStatus();
        if (res.success && res.status) {
          setGenProgress(res.status.progress);
          setGenMessage(res.status.message);
          if (!res.status.running) {
            setGenerating(false);
            if (res.status.stage === 'done') {
              toast.success('样本生成完成');
              loadSamples();
            } else if (res.status.stage === 'cancelled') {
              toast.info('已取消');
            } else if (res.status.stage === 'error') {
              toast.error(res.status.message);
            }
          }
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => clearInterval(timer);
  }, [generating, loadSamples]);

  // 轮询训练进度
  useEffect(() => {
    if (!training) return;
    const timer = setInterval(async () => {
      try {
        const res = await api.getIntentTrainingStatus();
        if (res.success && res.status) {
          setTrainProgress(res.status.progress);
          setTrainStage(res.status.stage);
          setTrainMessage(res.status.message);
          setLogs(res.status.logs || []);
          if (!res.status.running) {
            setTraining(false);
            const result = res.status.result;
            if (result && 'error' in result) {
              toast.error(`训练失败: ${result.error}`);
            } else if (result && 'cancelled' in result) {
              toast.info('训练已取消');
            } else if (result) {
              toast.success(`训练完成！准确率=${result.accuracy ?? 'N/A'}`);
              loadModelInfo();
              loadActiveKbs();
            }
          }
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => clearInterval(timer);
  }, [training, loadModelInfo, loadActiveKbs]);

  // 生成样本
  const handleGenerate = async () => {
    if (selectedKbIds.length === 0) { toast.error('请至少选择一个知识库'); return; }
    try {
      setGenerating(true);
      setGenProgress(0);
      setGenMessage('启动样本生成...');
      const res = await api.generateIntentSamples({
        kb_ids: selectedKbIds, samples_per_kb: samplesPerKb, negative_count: negativeCount,
      });
      if (!res.success) { setGenerating(false); toast.error(res.error || '启动失败'); }
      else toast.success('样本生成已启动');
    } catch { setGenerating(false); toast.error('启动失败'); }
  };

  // 训练
  const handleTrain = async () => {
    if (Object.keys(samples).length === 0) { toast.error('请先生成样本'); return; }
    try {
      setTraining(true);
      setTrainProgress(0);
      setTrainMessage('启动训练...');
      setLogs([]);
      const res = await api.trainIntentClassifier({ kb_ids: selectedKbIds });
      if (!res.success) { setTraining(false); toast.error(res.error || '启动失败'); }
      else toast.success('训练已启动');
    } catch { setTraining(false); toast.error('启动失败'); }
  };

  // 取消
  const handleCancel = async () => {
    try { await api.cancelIntentTraining(); toast.info('已发送取消请求'); } catch { toast.error('取消失败'); }
  };

  // 切换知识库
  const toggleKb = (id: number) => {
    setSelectedKbIds(prev => {
      if (prev.includes(id)) return prev.filter(i => i !== id);
      if (prev.length >= MAX_KB_COUNT) { toast.warning(`最多选择 ${MAX_KB_COUNT} 个`); return prev; }
      return [...prev, id];
    });
  };

  // 样本编辑
  const startEdit = (label: string, index: number, text: string) => {
    setEditingLabel(label);
    setEditingIndex(index);
    setEditingText(text);
  };

  const saveEdit = async () => {
    if (editingLabel === null || editingIndex === null) return;
    try {
      await api.updateIntentSample(editingLabel, editingIndex, editingText);
      setEditingLabel(null);
      setEditingIndex(null);
      loadSamples();
    } catch { toast.error('保存失败'); }
  };

  const handleDelete = async (label: string, index: number) => {
    try {
      await api.deleteIntentSample(label, index);
      loadSamples();
    } catch { toast.error('删除失败'); }
  };

  const handleAdd = async () => {
    if (!addingLabel || !addingText.trim()) return;
    try {
      await api.addIntentSample(addingLabel, addingText.trim());
      setAddingLabel(null);
      setAddingText('');
      loadSamples();
    } catch { toast.error('添加失败'); }
  };

  // ── 批量选择操作 ──
  const toggleSelect = (label: string, idx: number) => {
    setSelectedMap(prev => {
      const set = new Set(prev[label] || []);
      if (set.has(idx)) set.delete(idx); else set.add(idx);
      return { ...prev, [label]: set };
    });
  };

  const selectAll = (label: string) => {
    const items = samples[label] || [];
    setSelectedMap(prev => ({ ...prev, [label]: new Set(items.map((_, i) => i)) }));
  };

  const deselectAll = (label: string) => {
    setSelectedMap(prev => ({ ...prev, [label]: new Set() }));
  };

  const invertSelection = (label: string) => {
    const items = samples[label] || [];
    const current = selectedMap[label] || new Set();
    const allIndices = new Set(items.map((_, i) => i));
    const inverted = new Set([...allIndices].filter(i => !current.has(i)));
    setSelectedMap(prev => ({ ...prev, [label]: inverted }));
  };

  const getSelectedCount = (label: string) => (selectedMap[label]?.size || 0);

  const totalSelected = Object.values(selectedMap).reduce((acc, s) => acc + s.size, 0);

  const handleDeleteSelected = async () => {
    if (totalSelected === 0) return;
    const newSamples: Record<string, string[]> = {};
    for (const [label, items] of Object.entries(samples)) {
      const selected = selectedMap[label] || new Set();
      newSamples[label] = items.filter((_, i) => !selected.has(i));
    }
    try {
      await api.saveIntentSamples(newSamples);
      setSelectedMap({});
      loadSamples();
      toast.success('已删除选中样本');
    } catch { toast.error('批量删除失败'); }
  };

  const handleKeepSelected = async () => {
    if (totalSelected === 0) return;
    const newSamples: Record<string, string[]> = {};
    for (const [label, items] of Object.entries(samples)) {
      const selected = selectedMap[label] || new Set();
      newSamples[label] = items.filter((_, i) => selected.has(i));
    }
    try {
      await api.saveIntentSamples(newSamples);
      setSelectedMap({});
      loadSamples();
      toast.success('已保留选中样本，其余已删除');
    } catch { toast.error('保留选中失败'); }
  };

  const totalSamples = Object.values(sampleStats).reduce((a, b) => a + b, 0);
  const kbSampleCount = Object.entries(sampleStats).filter(([k]) => k !== 'none').reduce((a, [, v]) => a + v, 0);
  const negSampleCount = sampleStats['none'] || 0;

  return (
    <AppLayout>
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">意图训练</h2>
          <p className="text-muted-foreground text-sm">
            训练多分类意图识别模型，自动将用户查询路由到正确的知识库
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="config">1. 配置 & 生成</TabsTrigger>
            <TabsTrigger value="samples" disabled={totalSamples === 0 && !generating}>
              2. 审查样本 {totalSamples > 0 && `(${totalSamples})`}
            </TabsTrigger>
            <TabsTrigger value="train" disabled={totalSamples === 0}>
              3. 训练模型
            </TabsTrigger>
          </TabsList>

          {/* ── Tab 1: 配置 & 生成 ── */}
          <TabsContent value="config" className="space-y-6 mt-4">
            <div className="grid gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2 space-y-6">
                {/* 知识库选择 */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2"><Database className="h-5 w-5" />选择知识库</CardTitle>
                        <CardDescription>选择参与训练的知识库（最多 {MAX_KB_COUNT} 个）</CardDescription>
                      </div>
                      <Button variant="outline" size="sm" onClick={() => setSelectedKbIds(selectedKbIds.length === knowledgeBases.length ? [] : knowledgeBases.slice(0, MAX_KB_COUNT).map(b => b.id))}>
                        {selectedKbIds.length === knowledgeBases.length ? '取消全选' : '全选'}
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {knowledgeBases.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <Database className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>暂无知识库，请先创建</p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {knowledgeBases.map(kb => {
                          const selected = selectedKbIds.includes(kb.id);
                          return (
                            <div key={kb.id} onClick={() => !generating && toggleKb(kb.id)}
                              className={`flex items-center gap-3 rounded-lg border p-3 cursor-pointer transition-colors ${selected ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'} ${generating ? 'opacity-60 pointer-events-none' : ''}`}>
                              <Checkbox checked={selected} />
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate">{kb.name}</p>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                    <div className="mt-3 text-xs text-muted-foreground">已选择 {selectedKbIds.length}/{MAX_KB_COUNT} 个</div>
                  </CardContent>
                </Card>

                {/* 参数 */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">生成参数</CardTitle>
                    <CardDescription>LLM将读取知识库文档内容后生成样本</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>每个知识库样本数</Label>
                        <Input type="number" min={20} max={500} value={samplesPerKb} onChange={e => setSamplesPerKb(parseInt(e.target.value) || 100)} disabled={generating} />
                      </div>
                      <div className="space-y-2">
                        <Label>负例样本数（闲聊）</Label>
                        <Input type="number" min={50} max={500} value={negativeCount} onChange={e => setNegativeCount(parseInt(e.target.value) || 200)} disabled={generating} />
                      </div>
                    </div>
                    <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                      <Info className="h-4 w-4" />
                      <span>预计总样本数：{selectedKbIds.length * samplesPerKb + negativeCount} 条</span>
                    </div>
                  </CardContent>
                </Card>

                {/* 生成进度 */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {generating ? <Loader2 className="h-5 w-5 animate-spin" /> : <FileText className="h-5 w-5" />}
                      样本生成
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {(generating || genMessage) && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>{genMessage || '等待启动...'}</span>
                          <span className="text-muted-foreground">{genProgress}%</span>
                        </div>
                        <Progress value={genProgress} className="h-2" />
                      </div>
                    )}
                    <div className="flex gap-3">
                      {!generating ? (
                        <Button onClick={handleGenerate} disabled={selectedKbIds.length === 0}>
                          <Play className="mr-2 h-4 w-4" />生成样本
                        </Button>
                      ) : (
                        <Button variant="destructive" onClick={handleCancel}>
                          <XCircle className="mr-2 h-4 w-4" />取消
                        </Button>
                      )}
                      {totalSamples > 0 && (
                        <Button variant="outline" onClick={() => setActiveTab('samples')}>
                          查看样本 ({totalSamples})
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* 右侧：模型信息 */}
              <div className="space-y-6">
                <Card>
                  <CardHeader><CardTitle className="text-sm">当前模型</CardTitle></CardHeader>
                  <CardContent>
                    {modelInfo && modelInfo.exists ? (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                          <span className="text-sm font-medium">模型已就绪</span>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between"><span className="text-muted-foreground">类型</span><span>{modelInfo.model_type === 'multiclass_logistic_regression' ? '多分类LR' : modelInfo.model_type}</span></div>
                          <div className="flex justify-between"><span className="text-muted-foreground">样本数</span><span>{modelInfo.training_samples}</span></div>
                          <div className="flex justify-between"><span className="text-muted-foreground">准确率</span><span className="font-medium text-green-600">{modelInfo.accuracy ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : 'N/A'}</span></div>
                          <div className="flex justify-between"><span className="text-muted-foreground">训练时间</span><span>{modelInfo.trained_at}</span></div>
                        </div>
                        {modelInfo.label_names && modelInfo.label_names.length > 0 && (
                          <div className="space-y-1">
                            <span className="text-xs text-muted-foreground">分类类别</span>
                            <div className="flex flex-wrap gap-1">
                              {modelInfo.label_names.map(name => (
                                <Badge key={name} variant={name === 'none' ? 'outline' : 'secondary'} className="text-xs">
                                  {name === 'none' ? '闲聊' : name}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center py-4 text-muted-foreground">
                        <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">尚未训练模型</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle className="text-sm">检索知识库</CardTitle></CardHeader>
                  <CardContent>
                    {activeKbIds.length > 0 ? (
                      <div className="space-y-1">
                        {knowledgeBases.filter(kb => activeKbIds.includes(kb.id)).map(kb => (
                          <div key={kb.id} className="flex items-center gap-2 text-sm py-1">
                            <div className="h-2 w-2 rounded-full bg-green-500" /><span>{kb.name}</span>
                          </div>
                        ))}
                      </div>
                    ) : <p className="text-xs text-muted-foreground">训练后自动设置</p>}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* ── Tab 2: 审查样本 ── */}
          <TabsContent value="samples" className="mt-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>训练样本审查</CardTitle>
                    <CardDescription>
                      检查并修正LLM生成的样本，确保质量后再训练
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">知识库样本: {kbSampleCount}</Badge>
                    <Badge variant="outline">闲聊样本: {negSampleCount}</Badge>
                    <Badge>总计: {totalSamples}</Badge>
                    {totalSelected > 0 && (
                      <Badge variant="destructive">已选: {totalSelected}</Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {Object.keys(samples).length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">暂无样本，请先生成</div>
                ) : (
                  <div className="space-y-6">
                    {/* 全局批量操作栏 */}
                    {totalSelected > 0 && (
                      <div className="flex items-center gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
                        <span className="text-sm font-medium">已选中 {totalSelected} 条样本</span>
                        <div className="flex-1" />
                        <Button variant="destructive" size="sm" onClick={handleDeleteSelected}>
                          <Trash className="mr-1 h-3 w-3" />删除选中
                        </Button>
                        <Button variant="outline" size="sm" onClick={handleKeepSelected}>
                          <ShieldCheck className="mr-1 h-3 w-3" />保留选中
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedMap({})}>
                          取消选择
                        </Button>
                      </div>
                    )}

                    {Object.entries(samples).map(([label, items]) => {
                      const selCount = getSelectedCount(label);
                      const allSelected = selCount === items.length && items.length > 0;
                      return (
                        <div key={label}>
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Badge variant={label === 'none' ? 'outline' : 'default'}>
                                {label === 'none' ? '闲聊（不需要RAG）' : label}
                              </Badge>
                              <span className="text-xs text-muted-foreground">{items.length} 条</span>
                              {selCount > 0 && (
                                <span className="text-xs text-primary">已选 {selCount}</span>
                              )}
                            </div>
                            <div className="flex items-center gap-1">
                              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => allSelected ? deselectAll(label) : selectAll(label)}>
                                {allSelected ? <SquareCheck className="mr-1 h-3 w-3" /> : <Square className="mr-1 h-3 w-3" />}
                                {allSelected ? '取消全选' : '全选'}
                              </Button>
                              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => invertSelection(label)}>
                                <FlipHorizontal2 className="mr-1 h-3 w-3" />反选
                              </Button>
                              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => { setAddingLabel(label); setAddingText(''); }}>
                                <Plus className="h-3 w-3 mr-1" />添加
                              </Button>
                            </div>
                          </div>
                          <div className="space-y-1.5">
                            {items.map((text, idx) => {
                              const isSelected = selectedMap[label]?.has(idx) ?? false;
                              return (
                                <div key={idx} className={`group flex items-start gap-2 rounded border px-3 py-2 text-sm hover:bg-muted/50 ${isSelected ? 'border-primary/50 bg-primary/5' : ''}`}>
                                  <Checkbox
                                    checked={isSelected}
                                    onCheckedChange={() => toggleSelect(label, idx)}
                                    className="mt-0.5 shrink-0"
                                  />
                                  {editingLabel === label && editingIndex === idx ? (
                                    <div className="flex-1 flex items-start gap-2">
                                      <Textarea value={editingText} onChange={e => setEditingText(e.target.value)} className="min-h-[60px] text-sm resize-y" onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveEdit(); } }} />
                                      <div className="flex flex-col gap-1">
                                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={saveEdit}><Check className="h-3 w-3" /></Button>
                                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => { setEditingLabel(null); setEditingIndex(null); }}><X className="h-3 w-3" /></Button>
                                      </div>
                                    </div>
                                  ) : (
                                    <>
                                      <span className="flex-1 whitespace-pre-wrap break-all leading-relaxed">{text}</span>
                                      <div className="hidden group-hover:flex items-center gap-0.5 shrink-0">
                                        <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => startEdit(label, idx, text)}><Pencil className="h-3 w-3" /></Button>
                                        <Button variant="ghost" size="sm" className="h-6 w-6 p-0 text-destructive" onClick={() => handleDelete(label, idx)}><Trash2 className="h-3 w-3" /></Button>
                                      </div>
                                    </>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                          {/* 添加样本输入 */}
                          {addingLabel === label && (
                            <div className="mt-2 flex items-start gap-2">
                              <Textarea placeholder="输入新样本..." value={addingText} onChange={e => setAddingText(e.target.value)} className="min-h-[60px] text-sm resize-y" onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAdd(); } }} />
                              <div className="flex flex-col gap-1">
                                <Button size="sm" onClick={handleAdd} disabled={!addingText.trim()}>添加</Button>
                                <Button variant="ghost" size="sm" onClick={() => setAddingLabel(null)}>取消</Button>
                              </div>
                            </div>
                          )}
                          <Separator className="mt-4" />
                        </div>
                      );
                    })}
                  </div>
                )}
                <div className="mt-4 flex gap-3">
                  <Button onClick={() => setActiveTab('train')} disabled={totalSamples === 0}>
                    确认样本，进入训练
                  </Button>
                  <Button variant="outline" onClick={loadSamples}>
                    <RefreshCw className="mr-2 h-4 w-4" />刷新
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ── Tab 3: 训练 ── */}
          <TabsContent value="train" className="mt-4">
            <div className="grid gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {training ? <Loader2 className="h-5 w-5 animate-spin" /> : trainStage === 'done' ? <CheckCircle2 className="h-5 w-5 text-green-500" /> : <Brain className="h-5 w-5" />}
                      训练多分类模型
                    </CardTitle>
                    <CardDescription>
                      使用 {totalSamples} 条样本训练（{Object.keys(samples).filter(k => k !== 'none').length} 个知识库 + 闲聊）
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>{trainMessage || '等待启动...'}</span>
                        <span className="text-muted-foreground">{trainProgress}%</span>
                      </div>
                      <Progress value={trainProgress} className="h-2" />
                      {trainStage && <Badge variant="outline" className="text-xs">{trainStage}</Badge>}
                    </div>

                    <div className="flex gap-3">
                      {!training ? (
                        <Button onClick={handleTrain} disabled={totalSamples === 0}>
                          <Play className="mr-2 h-4 w-4" />开始训练
                        </Button>
                      ) : (
                        <Button variant="destructive" onClick={handleCancel}>
                          <XCircle className="mr-2 h-4 w-4" />取消训练
                        </Button>
                      )}
                      <Button variant="outline" onClick={() => { loadModelInfo(); loadActiveKbs(); }}>
                        <RefreshCw className="mr-2 h-4 w-4" />刷新
                      </Button>
                    </div>

                    {logs.length > 0 && (
                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">训练日志</Label>
                        <ScrollArea className="h-40 rounded-md border bg-muted/30 p-3">
                          {logs.map((log, i) => (
                            <p key={i} className="text-xs font-mono text-muted-foreground">{log}</p>
                          ))}
                        </ScrollArea>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* 右侧：训练样本分布 */}
              <div className="space-y-6">
                <Card>
                  <CardHeader><CardTitle className="text-sm">样本分布</CardTitle></CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {Object.entries(sampleStats).map(([label, count]) => (
                        <div key={label} className="flex items-center justify-between text-sm">
                          <span className="flex items-center gap-2">
                            <Badge variant={label === 'none' ? 'outline' : 'secondary'} className="text-xs">
                              {label === 'none' ? '闲聊' : label}
                            </Badge>
                          </span>
                          <span>{count} 条</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle className="text-sm">当前模型</CardTitle></CardHeader>
                  <CardContent>
                    {modelInfo && modelInfo.exists ? (
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                          <span className="font-medium">模型已就绪</span>
                        </div>
                        <div className="flex justify-between"><span className="text-muted-foreground">准确率</span><span className="font-medium text-green-600">{modelInfo.accuracy ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : 'N/A'}</span></div>
                        <div className="flex justify-between"><span className="text-muted-foreground">训练时间</span><span>{modelInfo.trained_at}</span></div>
                        {modelInfo.samples_per_class && (
                          <div className="mt-2 space-y-1">
                            <span className="text-xs text-muted-foreground">各类别样本</span>
                            {Object.entries(modelInfo.samples_per_class).map(([name, count]) => (
                              <div key={name} className="flex justify-between text-xs">
                                <span>{name === 'none' ? '闲聊' : name}</span>
                                <span>{count}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">尚未训练</p>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </AppLayout>
  );
}
