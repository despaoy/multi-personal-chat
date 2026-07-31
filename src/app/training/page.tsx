'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import {
  Database,
  RefreshCw,
  Upload,
  Plus,
  Zap,
  FileText,
  XCircle,
  Sparkles,
  Download,
  Trash2,
  Edit3,
  Eye,
  Search,
  MessageSquare,
  User,
  Bot,
  Loader2,
  Save,
} from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { api, type Dataset, type TrainingTask, type CreateDatasetRequest, type StartTrainingRequest, type DialogueConversation, type DialogueGenerateRequest, type SavedDialogueItem } from '@/lib/api';
import { toast } from 'sonner';
import { usePageData } from '@/hooks/usePageData';
import { SavedDialoguesPanel } from '@/components/training/SavedDialoguesPanel';
import { TrainingTasksPanel } from '@/components/training/TrainingTasksPanel';

export default function TrainingPage() {
  return (
    <AuthGuard requireAdmin>
      <TrainingContent />
    </AuthGuard>
  );
}

function TrainingContent() {
  const [activeTab, setActiveTab] = useState('datasets');

  // 表单数据持久化
  const [pageFormData, setPageFormData] = usePageData('training', {
    charDescription: '',
    charStyle: '',
    charCustomPrompt: '',
    numDialogues: 500,
    newDatasetName: '',
    newDatasetStyle: '',
    newDatasetCustomPrompt: '',
    newDatasetData: '',
    selectedModelConfig: 'qwen3-8b-3090',
  });

  // 便捷访问器
  const updateFormField = <K extends keyof typeof pageFormData>(key: K, value: (typeof pageFormData)[K]) => {
    setPageFormData(prev => ({ ...prev, [key]: value }));
  };
  const { charDescription, charStyle, charCustomPrompt, numDialogues,
    newDatasetName, newDatasetStyle, newDatasetCustomPrompt, newDatasetData,
    selectedModelConfig } = pageFormData;

  // 数据集状态
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);

  // 扫描文件夹状态
  const [scanFolderPath, setScanFolderPath] = useState('');
  const [scanResults, setScanResults] = useState<Array<{ name: string; path: string; file_count: number; files: string[]; valid: boolean }>>([]);
  const [scanLoading, setScanLoading] = useState(false);
  const [importLoading, setImportLoading] = useState<string | null>(null);
  const [showScanSection, setShowScanSection] = useState(false);

  // 模型配置状态

  // 训练任务状态
  const [trainingTasks, setTrainingTasks] = useState<TrainingTask[]>([]);
  const [tasksLoading, setTasksLoading] = useState(true);

  // 新建数据集对话框
  const [createDatasetOpen, setCreateDatasetOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [dataInputMode, setDataInputMode] = useState<'text' | 'file'>('text');

  // 启动训练对话框
  const [startTrainingOpen, setStartTrainingOpen] = useState(false);
  const [trainingSubmitting, setTrainingSubmitting] = useState(false);

  // ========== 快速生成对话状态 ==========
  const [generating, setGenerating] = useState(false);
  const [reconnecting, setReconnecting] = useState(true); // 挂载时检查中
  const [genProgress, setGenProgress] = useState(0);
  const [genBatchInfo, setGenBatchInfo] = useState('');
  const [genGeneratedCount, setGenGeneratedCount] = useState(0);
  const [genTotal, setGenTotal] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [generatedDialogues, setGeneratedDialogues] = useState<DialogueConversation[]>([]);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editContent, setEditContent] = useState('');
  const [editRole, setEditRole] = useState<'human' | 'gpt'>('human');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDialogues, setSelectedDialogues] = useState<Set<number>>(new Set());

  // ========== 已保存对话管理状态 ==========
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(true);
  const [savedItems, setSavedItems] = useState<SavedDialogueItem[]>([]);
  const [savedLoading, setSavedLoading] = useState(false);
  const [savedSearchQuery, setSavedSearchQuery] = useState('');
  const [previewSaveId, setPreviewSaveId] = useState<number | null>(null);
  const [previewSaveDialogues, setPreviewSaveDialogues] = useState<DialogueConversation[]>([]);
  const [previewSaveLoading, setPreviewSaveLoading] = useState(false);

  // ========== 挂载时检测后台是否有正在进行的生成任务（断线重连） ==========
  const reconnectTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 可复用的重连逻辑：从进度数据恢复生成状态并开始轮询
  const reconnectToGeneration = useCallback((progress: { total: number; progress: number; batch_num: number; total_batches: number; generated_count: number; all_generated_dialogues: DialogueConversation[]; new_dialogues: DialogueConversation[] }) => {
    setGenerating(true);
    setGenTotal(progress.total);
    setGenProgress(progress.progress);
    setGenGeneratedCount(progress.generated_count);
    setGenBatchInfo(`批次 ${progress.batch_num}/${progress.total_batches}`);

    if (progress.all_generated_dialogues && progress.all_generated_dialogues.length > 0) {
      setGeneratedDialogues(progress.all_generated_dialogues);
    }

    // 清除旧的定时器
    if (reconnectTimerRef.current) clearInterval(reconnectTimerRef.current);
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);

    // 启动进度轮询
    reconnectTimerRef.current = setInterval(async () => {
      try {
        const p = await api.getDialogueGenerationProgress();
        if (!p.is_generating) {
          setGenerating(false);
          setGenProgress(p.progress || 0);
          setGenGeneratedCount(p.generated_count);
          if (reconnectTimerRef.current) {
            clearInterval(reconnectTimerRef.current);
            reconnectTimerRef.current = null;
          }
          return;
        }
        setGenProgress(p.progress);
        setGenGeneratedCount(p.generated_count);
        setGenBatchInfo(`批次 ${p.batch_num}/${p.total_batches}`);
        if (p.new_dialogues && p.new_dialogues.length > 0) {
          setGeneratedDialogues(prev => [...prev, ...p.new_dialogues]);
        }
      } catch {
        // ignore
      }
    }, 2000);

    toast.info('检测到后台正在生成对话，已恢复连接');
  }, []);

  useEffect(() => {
    const checkAndReconnect = async () => {
      try {
        const progress = await api.getDialogueGenerationProgress();
        if (progress.is_generating) {
          reconnectToGeneration(progress);
        }
      } catch {
        // 后台不可用，忽略
      } finally {
        setReconnecting(false);
      }
    };
    checkAndReconnect();

    return () => {
      if (reconnectTimerRef.current) {
        clearInterval(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };
  }, [reconnectToGeneration]);

  // 加载数据
  const loadDatasets = useCallback(async () => {
    try {
      setDatasetsLoading(true);
      const response = await api.listDatasets();
      if (response.success) {
        setDatasets(response.datasets);
      }
    } catch (error) {
      console.error('Failed to load datasets:', error);
      toast.error('加载数据集失败');
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  // 扫描数据集文件夹
  const handleScanFolder = useCallback(async () => {
    setScanLoading(true);
    try {
      const result = await api.scanDatasets(scanFolderPath || undefined);
      setScanResults(result.datasets);
      toast.success(`扫描完成，发现 ${result.count} 个数据集`);
    } catch (err) {
      toast.error('扫描文件夹失败');
      console.error('Failed to scan datasets folder:', err);
    } finally {
      setScanLoading(false);
    }
  }, [scanFolderPath]);

  // 导入单个数据集
  const handleImportDataset = useCallback(async (sourcePath: string, datasetName: string) => {
    setImportLoading(datasetName);
    try {
      const result = await api.importDataset(sourcePath, datasetName);
      toast.success(`数据集「${result.name}」导入成功`);
      loadDatasets();
      setScanResults(prev => prev.filter(r => r.name !== datasetName));
    } catch (err) {
      toast.error('导入数据集失败');
      console.error('Failed to import dataset:', err);
    } finally {
      setImportLoading(null);
    }
  }, [loadDatasets]);

  // 一键导入全部
  const handleImportAll = useCallback(async () => {
    for (const ds of scanResults) {
      if (!ds.valid) continue;
      await handleImportDataset(ds.path, ds.name);
    }
  }, [scanResults, handleImportDataset]);

  const loadTrainingTasks = useCallback(async () => {
    try {
      setTasksLoading(true);
      const response = await api.listTrainingTasks();
      if (response.success) {
        setTrainingTasks(response.tasks);
      }
    } catch (error) {
      console.error('Failed to load training tasks:', error);
      toast.error('加载训练任务失败');
    } finally {
      setTasksLoading(false);
    }
  }, []);

  useEffect(() => {

    loadDatasets();
    loadTrainingTasks();
  }, [loadDatasets, loadTrainingTasks]);

  // 定期刷新训练任务
  // 使用 ref 保存最新的 trainingTasks，避免把它放进定时器 effect 依赖导致
  // 每次轮询后 trainingTasks 变化 → effect 重跑 → interval 反复重建（丢 tick + 抖动）。
  const trainingTasksRef = useRef(trainingTasks);
  useEffect(() => {
    trainingTasksRef.current = trainingTasks;
  }, [trainingTasks]);

  useEffect(() => {
    const POLL_INTERVAL = 3000;
    const interval = setInterval(() => {
      // 页面不可见时跳过轮询，节省后端请求
      if (typeof document !== 'undefined' && document.visibilityState !== 'visible') {
        return;
      }
      const hasActiveTasks = trainingTasksRef.current.some(
        task => task.status === 'pending' || task.status === 'training'
      );
      if (hasActiveTasks) {
        loadTrainingTasks();
      }
    }, POLL_INTERVAL);

    return () => clearInterval(interval);
  }, [loadTrainingTasks]);

  // 处理文件选择
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);

      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        setFileContent(content);
      };
      reader.readAsText(file);
    }
  };

  // 创建数据集
  const handleCreateDataset = async () => {
    try {
      let parsedData;
      const dataToParse = dataInputMode === 'file' ? fileContent : newDatasetData;

      try {
        parsedData = JSON.parse(dataToParse);
      } catch {
        toast.error('数据格式错误，请输入有效的JSON');
        return;
      }

      const request: CreateDatasetRequest = {
        dataset_name: newDatasetName,
        style: newDatasetStyle || undefined,
        custom_prompt: newDatasetCustomPrompt || undefined,
        data: parsedData
      };

      const response = await api.createDataset(request);
      if (response.success) {
        toast.success('数据集创建成功');
        setCreateDatasetOpen(false);
        setPageFormData(prev => ({
          ...prev,
          newDatasetName: '',
          newDatasetStyle: '',
          newDatasetCustomPrompt: '',
          newDatasetData: '',
        }));
        setSelectedFile(null);
        setFileContent('');
        setDataInputMode('text');
        loadDatasets();
      } else {
        toast.error((response as { message?: string }).message || '创建数据集失败');
      }
    } catch (error) {
      console.error('Failed to create dataset:', error);
      toast.error('创建数据集失败');
    }
  };

  // 启动训练（由 TrainingParamsEditor 提交）
  const handleStartTraining = async (params: {
    loraName: string;
    datasetName: string;
    customConfig: Record<string, unknown>;
  }) => {
    setTrainingSubmitting(true);
    try {
      const request: StartTrainingRequest = {
        lora_name: params.loraName,
        dataset_name: params.datasetName,
        // 仍传一个默认 model_type 作为后端基础配置兜底，
        // custom_config 会覆盖/补充所有自定义字段
        model_type: selectedModelConfig,
        custom_config: params.customConfig,
      };

      const response = await api.startTraining(request);
      if (response.success) {
        toast.success('训练任务已启动');
        setStartTrainingOpen(false);
        loadTrainingTasks();
      } else {
        const msg = (response as { message?: string }).message || '启动训练失败';
        toast.error(msg, { duration: 6000 });
      }
    } catch (error: unknown) {
      console.error('Failed to start training:', error);
      const msg = error instanceof Error ? error.message : '启动训练失败';
      toast.error(msg, { duration: 6000 });
    } finally {
      setTrainingSubmitting(false);
    }
  };

  // 取消训练
  const handleCancelTraining = async (taskId: string) => {
    try {
      const response = await api.cancelTrainingTask(taskId);
      if (response.success) {
        toast.success('训练任务已取消');
        loadTrainingTasks();
      } else {
        toast.error((response as { message?: string }).message || '取消训练失败');
      }
    } catch (error) {
      console.error('Failed to cancel training:', error);
      toast.error('取消训练失败');
    }
  };

  // ========== 快速生成对话功能 ==========

  // 强制重置生成状态
  const handleForceReset = async () => {
    try {
      await api.forceResetGeneration();
      setGenerating(false);
      setGenProgress(0);
      setGenBatchInfo('');
      setGenGeneratedCount(0);
      setGeneratedDialogues([]);
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
        progressTimerRef.current = null;
      }
      if (reconnectTimerRef.current) {
        clearInterval(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      abortControllerRef.current = null;
      toast.success('生成状态已重置，可以重新生成');
    } catch {
      toast.error('重置失败');
    }
  };

  // 生成对话
  const handleGenerate = async () => {
    // 如果正在生成，点击则取消
    if (generating) {
      try {
        await api.cancelDialogueGeneration();
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }
        toast.success('已取消生成');
      } catch {
        toast.error('取消失败');
      }
      return;
    }

    if (!charDescription.trim()) {
      toast.error('请输入角色描述');
      return;
    }

    // 启动前先检查后台是否已有生成任务在运行（断线重连场景）
    try {
      const existingProgress = await api.getDialogueGenerationProgress();
      if (existingProgress.is_generating) {
        // 后台有正在进行的生成，恢复连接而非启动新任务
        reconnectToGeneration(existingProgress);
        return;
      }
    } catch {
      // 后端不可用，继续尝试启动新任务
    }
    try {
      setGenerating(true);
      setGenProgress(0);
      setGenBatchInfo('');
      setGenGeneratedCount(0);
      setGenTotal(numDialogues);
      setGeneratedDialogues([]);
      setSelectedDialogues(new Set());

      const controller = new AbortController();
      abortControllerRef.current = controller;

      // 启动进度轮询
      progressTimerRef.current = setInterval(async () => {
        try {
          const progress = await api.getDialogueGenerationProgress();
          if (progress.is_generating) {
            setGenProgress(progress.progress);
            setGenGeneratedCount(progress.generated_count);
            setGenTotal(progress.total);
            setGenBatchInfo(`批次 ${progress.batch_num}/${progress.total_batches}`);
            // 实时追加本批次新生成的对话到右侧预览
            if (progress.new_dialogues && progress.new_dialogues.length > 0) {
              setGeneratedDialogues(prev => [...prev, ...progress.new_dialogues]);
            }
          }
        } catch {
          // 忽略进度查询错误
        }
      }, 2000);

      const request: DialogueGenerateRequest = {
        character_description: charDescription,
        num_dialogues: numDialogues,
        style: charStyle || undefined,
        custom_prompt: charCustomPrompt || undefined,
      };

      const response = await api.generateDialogues(request, controller.signal);
      if (response.success) {
        // 不直接覆盖，而是合并：API返回的完整数据 + 轮询期间可能遗漏的
        setGeneratedDialogues((prev) => {
          // 如果轮询已收集到数据，以API返回的为准（更完整）
          // 但如果轮询数据比API返回的更多（不太可能），保留轮询数据
          const dialogues = response.dialogues || [];
          if (dialogues.length >= prev.length) {
            return dialogues;
          }
          return prev;
        });
        if (response.cancelled) {
          toast.success(`已取消生成，获得 ${response.total} 组对话`);
        } else {
          toast.success(`成功生成 ${response.total} 组对话，耗时 ${response.cost_time}s`);
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        toast.success('已取消生成');
      } else if (error instanceof Error && error.message.includes('已有生成任务正在运行')) {
        // 后台有残留的生成任务（断线重连场景），提示用户强制重置
        toast.error('后台仍有生成任务在运行，点击下方"强制重置"按钮清除残留状态', {
          duration: 6000,
        });
      } else {
        console.error('Failed to generate dialogues:', error);
        toast.error(error instanceof Error ? error.message : '对话生成失败');
      }
    } finally {
      setGenerating(false);
      setGenProgress(0);
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
        progressTimerRef.current = null;
      }
      if (reconnectTimerRef.current) {
        clearInterval(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      abortControllerRef.current = null;
    }
  };

  // 保存当前对话（手动）
  // 保存当前对话（手动）- 使用传入数据的内部实现
  const handleSaveDialoguesWithData = async (dialogues: DialogueConversation[]) => {
    if (dialogues.length === 0) return;
    try {
      const turnStats: Record<string, number> = {};
      const sceneStats: Record<string, number> = {};
      for (const d of dialogues) {
        const turns = String(d.conversations?.length ?? 0);
        turnStats[turns] = (turnStats[turns] || 0) + 1;
        const scene = d.scene || '未知';
        sceneStats[scene] = (sceneStats[scene] || 0) + 1;
      }

      const name = `${charDescription.slice(0, 20)}_${new Date().toLocaleString('zh-CN')}`;
      await api.saveDialogues({
        name,
        character_desc: charDescription,
        style: charStyle || undefined,
        dialogues,
        turn_stats: turnStats,
        scene_stats: sceneStats,
      });
      toast.success(`已保存 ${dialogues.length} 组对话`);
      loadSavedItems();
    } catch (error) {
      toast.error('保存失败');
      console.error('Save failed:', error);
    }
  };

  const handleSaveDialogues = async () => {
    handleSaveDialoguesWithData(generatedDialogues);
  };

  // 加载已保存列表
  const loadSavedItems = useCallback(async () => {
    try {
      setSavedLoading(true);
      const res = await api.listSavedDialogues();
      if (res.success) {
        setSavedItems(res.items);
      }
    } catch (error) {
      console.error('Failed to load saved items:', error);
      toast.error('加载已保存对话失败');
    } finally {
      setSavedLoading(false);
    }
  }, []);

  // 切换到"已保存对话"Tab时自动加载列表
  useEffect(() => {
    if (activeTab === 'saved') {
      loadSavedItems();
    }
  }, [activeTab, loadSavedItems]);

  // 生成完成后自动保存（覆盖断线重连等场景）
  const prevGeneratingRef = useRef(generating);
  const autoSaveEnabledRef = useRef(autoSaveEnabled);
  useEffect(() => {
    autoSaveEnabledRef.current = autoSaveEnabled;
    if (prevGeneratingRef.current && !generating && generatedDialogues.length > 0 && autoSaveEnabledRef.current) {
      handleSaveDialoguesWithData(generatedDialogues);
    }
    prevGeneratingRef.current = generating;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generating, generatedDialogues, autoSaveEnabled]);

  // 预览已保存对话
  const handlePreviewSaved = async (id: number) => {
    try {
      setPreviewSaveLoading(true);
      setPreviewSaveId(id);
      const res = await api.getSavedDialogue(id);
      if (res.success) {
        setPreviewSaveDialogues(res.dialogues);
      }
    } catch (error) {
      toast.error('加载对话失败');
      console.error('Preview failed:', error);
    } finally {
      setPreviewSaveLoading(false);
    }
  };

  // 删除已保存对话
  const handleDeleteSaved = async (id: number) => {
    try {
      await api.deleteSavedDialogue(id);
      toast.success('已删除');
      if (previewSaveId === id) {
        setPreviewSaveId(null);
        setPreviewSaveDialogues([]);
      }
      loadSavedItems();
    } catch (error) {
      toast.error('删除失败');
      console.error('Delete failed:', error);
    }
  };

  // 从已保存对话中删除单条对话
  const handleDeleteDialogueFromSaved = async (savedId: number, dialogueIndex: number) => {
    try {
      await api.deleteDialogueFromSaved(savedId, dialogueIndex);
      toast.success('已删除该对话');
      // 从本地预览中移除
      setPreviewSaveDialogues(prev => prev.filter((_, i) => i !== dialogueIndex));
      loadSavedItems();
    } catch (error) {
      toast.error('删除失败');
      console.error('Delete dialogue failed:', error);
    }
  };

  // 从已保存对话创建数据集
  const handleCreateDatasetFromSaved = async (id: number) => {
    try {
      const item = savedItems.find(i => i.id === id);
      const res = await api.createDatasetFromSaved(id, item?.name);
      if (res.success && res.dataset) {
        toast.success(`数据集「${res.dataset.name}」创建成功（${res.dataset.count}条数据）`);
        setActiveTab('datasets');
        loadDatasets();
      } else {
        toast.error('创建数据集失败');
      }
    } catch (error) {
      toast.error('创建数据集失败');
      console.error('Create dataset failed:', error);
    }
  };

  // 筛选对话
  const filteredDialogues = generatedDialogues.filter((d) => {
    if (!searchFilter) return true;
    return d.conversations.some(
      (c) => c.value.toLowerCase().includes(searchFilter.toLowerCase())
    );
  });

  // 删除对话
  const handleDeleteDialogue = (index: number) => {
    setGeneratedDialogues((prev) => prev.filter((_, i) => i !== index));
    // 删除后重建选中索引集合（索引前移修正）
    setSelectedDialogues((prev) => {
      const next = new Set<number>();
      for (const idx of prev) {
        if (idx < index) next.add(idx);
        else if (idx > index) next.add(idx - 1);
        // idx === index 的被删除，不加入
      }
      return next;
    });
  };

  // 删除选中的对话
  const handleDeleteSelected = () => {
    const toDelete = new Set(selectedDialogues);
    setGeneratedDialogues((prev) => prev.filter((_, i) => !toDelete.has(i)));
    setSelectedDialogues(new Set());
    toast.success('已删除选中对话');
  };

  // 编辑对话
  const handleEditDialogue = (dialogueIdx: number, convIdx: number) => {
    const conv = generatedDialogues[dialogueIdx].conversations[convIdx];
    setEditingIndex(dialogueIdx * 1000 + convIdx);
    setEditContent(conv.value);
    setEditRole(conv.from);
  };

  const handleSaveEdit = (dialogueIdx: number, convIdx: number) => {
    setGeneratedDialogues((prev) => {
      const next = [...prev];
      const dialogue = { ...next[dialogueIdx] };
      const conversations = [...dialogue.conversations];
      conversations[convIdx] = { from: editRole, value: editContent };
      dialogue.conversations = conversations;
      next[dialogueIdx] = dialogue;
      return next;
    });
    setEditingIndex(null);
    toast.success('修改已保存');
  };

  // 导出为JSON
  const handleExport = () => {
    const data = selectedDialogues.size > 0
      ? generatedDialogues.filter((_, i) => selectedDialogues.has(i))
      : generatedDialogues;

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dialogues_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('导出成功');
  };

  // 导出数据集（下载为zip，包含train.json和dataset_info.json）
  const handleExportDataset = async (datasetName: string) => {
    try {
      const res = await api.exportDataset(datasetName);
      if (!res.ok) {
        toast.error('导出数据集失败');
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetName}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success('数据集导出成功，可上传到服务器训练');
    } catch (error) {
      console.error('Export dataset failed:', error);
      toast.error('导出数据集失败');
    }
  };

  // 保存为数据集
  const handleSaveAsDataset = async () => {
    const data = selectedDialogues.size > 0
      ? generatedDialogues.filter((_, i) => selectedDialogues.has(i))
      : generatedDialogues;

    if (data.length === 0) {
      toast.error('没有可保存的对话数据');
      return;
    }

    const datasetName = `generated_${charDescription.slice(0, 10).replace(/\s/g, '_')}_${Date.now()}`;

    try {
      const request: CreateDatasetRequest = {
        dataset_name: datasetName,
        style: charStyle || undefined,
        custom_prompt: charCustomPrompt || undefined,
        data: data,
      };

      const response = await api.createDataset(request);
      if (response.success) {
        toast.success(`数据集 "${datasetName}" 创建成功`);
        loadDatasets();
      }
    } catch (error) {
      console.error('Failed to save dataset:', error);
      toast.error('保存数据集失败');
    }
  };

  // 全选/取消全选
  const handleSelectAll = () => {
    if (selectedDialogues.size === filteredDialogues.length) {
      setSelectedDialogues(new Set());
    } else {
      setSelectedDialogues(new Set(filteredDialogues.map((_, i) => generatedDialogues.indexOf(filteredDialogues[i]))));
    }
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">LoRA 训练管理</h2>
            <p className="text-muted-foreground">管理数据集、训练任务和模型配置</p>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" size="icon" onClick={() => { loadDatasets(); loadTrainingTasks(); }}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <Tabs defaultValue="datasets" value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="generate" className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              快速生成
            </TabsTrigger>
            <TabsTrigger value="datasets" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              数据集
            </TabsTrigger>
            <TabsTrigger value="training" className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              训练任务
            </TabsTrigger>
            <TabsTrigger value="saved" className="gap-2">
              <Save className="h-4 w-4" /> 已保存对话
            </TabsTrigger>
          </TabsList>

          {/* ========== 快速生成对话 Tab ========== */}
          <TabsContent value="generate" className="mt-4 space-y-6">
            <div className="grid gap-6 lg:grid-cols-5">
              {/* 左侧：输入区域 */}
              <div className="lg:col-span-2 space-y-4">
                <Card className="border-primary/20">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-primary" />
                      角色对话生成
                    </CardTitle>
                    <CardDescription>
                      输入角色描述，AI 自动生成符合角色设定的训练对话数据
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="char-description">角色描述 *</Label>
                      <Textarea
                        id="char-description"
                        placeholder="请详细描述角色的特征，例如：&#10;&#10;胡桃 - 往生堂第七十七代堂主，性格活泼开朗、古灵精怪，说话喜欢用俏皮的语气，经常开玩笑，对生死有独特的哲学观。喜欢恶作剧，但内心善良。口头禅包括「嘿嘿」「呀哈」等。"
                        className="min-h-[160px] resize-y"
                        value={charDescription}
                        onChange={(e) => updateFormField('charDescription', e.target.value)}
                      />
                      <p className="text-xs text-muted-foreground">
                        描述越详细，生成的对话越符合角色设定
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="char-style">角色风格（可选）</Label>
                      <Input
                        id="char-style"
                        placeholder="例如：活泼俏皮、古灵精怪"
                        value={charStyle}
                        onChange={(e) => updateFormField('charStyle', e.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="char-prompt">额外风格要求（可选）</Label>
                      <Textarea
                        id="char-prompt"
                        placeholder="例如：常用语气词「嘿嘿」「呀哈」，喜欢用反问句"
                        className="min-h-[80px]"
                        value={charCustomPrompt}
                        onChange={(e) => updateFormField('charCustomPrompt', e.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="num-dialogues">生成对话组数</Label>
                      <Input
                        id="num-dialogues"
                        type="number"
                        value={numDialogues}
                        min={10}
                        max={5000}
                        step={50}
                        onChange={(e) => updateFormField('numDialogues', Math.max(10, Math.min(5000, Number(e.target.value))))}
                      />
                      <p className="text-xs text-muted-foreground">建议 500-2000 组用于 LoRA 训练，最多 5000 组</p>
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm text-muted-foreground flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={autoSaveEnabled}
                          onChange={(e) => setAutoSaveEnabled(e.target.checked)}
                          className="rounded"
                        />
                        生成后自动保存
                      </Label>
                    </div>

                    <Button
                      className={`w-full h-12 text-base font-semibold ${generating ? 'bg-destructive hover:bg-destructive/90 text-white' : ''}`}
                      onClick={handleGenerate}
                      disabled={!charDescription.trim() || reconnecting}
                      size="lg"
                      variant={generating ? 'destructive' : 'default'}
                    >
                      {reconnecting ? (
                        <>
                          <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
                          正在检查后台任务...
                        </>
                      ) : generating ? (
                        <>
                          <XCircle className="mr-2 h-5 w-5" />
                          点击取消生成
                        </>
                      ) : (
                        <>
                          <Sparkles className="mr-2 h-5 w-5" />
                          生成对话数据
                        </>
                      )}
                    </Button>

                    <p className="text-center flex items-center justify-center gap-4">
                      <button
                        type="button"
                        onClick={handleForceReset}
                        className="text-xs text-muted-foreground hover:text-destructive transition-colors underline underline-offset-2"
                      >
                        强制重置生成状态
                      </button>
                      {generatedDialogues.length > 0 && (
                        <button
                          type="button"
                          onClick={handleSaveDialogues}
                          className="text-xs text-primary hover:underline underline-offset-2 transition-colors"
                        >
                          手动保存对话 ({generatedDialogues.length}组)
                        </button>
                      )}
                    </p>

                    {generating && (
                      <div className="space-y-2">
                        <Progress value={genProgress} className="h-3" />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>{genBatchInfo}</span>
                          <span>{genGeneratedCount} / {genTotal} 组 ({genProgress.toFixed(0)}%)</span>
                        </div>
                        <p className="text-xs text-muted-foreground text-center">
                          正在分批调用大模型生成对话（每批6组，覆盖12种场景类型，1-8轮多轮对话）
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* 右侧：预览区域 */}
              <div className="lg:col-span-3 space-y-4">
                {generatedDialogues.length > 0 ? (
                  <>
                    {/* 工具栏 */}
                    <Card>
                      <CardContent className="pt-4 pb-4">
                        <div className="flex flex-wrap items-center gap-3">
                          <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                            <Search className="h-4 w-4 text-muted-foreground shrink-0" />
                            <Input
                              placeholder="搜索对话内容..."
                              value={searchFilter}
                              onChange={(e) => setSearchFilter(e.target.value)}
                              className="h-8"
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary" className="text-xs">
                              {filteredDialogues.length} / {generatedDialogues.length} 组
                            </Badge>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={handleSelectAll}
                            >
                              {selectedDialogues.size === filteredDialogues.length ? '取消全选' : '全选'}
                            </Button>
                            {selectedDialogues.size > 0 && (
                              <Button
                                variant="destructive"
                                size="sm"
                                onClick={handleDeleteSelected}
                              >
                                <Trash2 className="mr-1 h-3 w-3" />
                                删除选中
                              </Button>
                            )}
                          </div>
                          <Separator orientation="vertical" className="h-6" />
                          <Button variant="outline" size="sm" onClick={handleExport}>
                            <Download className="mr-1 h-3 w-3" />
                            导出 JSON
                          </Button>
                          <Button size="sm" onClick={handleSaveAsDataset}>
                            <Database className="mr-1 h-3 w-3" />
                            保存为数据集
                          </Button>
                        </div>
                      </CardContent>
                    </Card>

                    {/* 对话列表 */}
                    <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
                      {filteredDialogues.map((dialogue) => {
                        const originalIdx = generatedDialogues.indexOf(dialogue);
                        return (
                          <Card
                            key={originalIdx}
                            className={`transition-colors ${
                              selectedDialogues.has(originalIdx) ? 'border-primary bg-primary/5' : ''
                            }`}
                          >
                            <CardHeader className="pb-2 pt-3 px-4">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <input
                                    type="checkbox"
                                    checked={selectedDialogues.has(originalIdx)}
                                    onChange={(e) => {
                                      setSelectedDialogues((prev) => {
                                        const next = new Set(prev);
                                        if (e.target.checked) next.add(originalIdx);
                                        else next.delete(originalIdx);
                                        return next;
                                      });
                                    }}
                                    className="rounded border-gray-300"
                                  />
                                  <CardTitle className="text-sm font-medium">
                                    对话 #{originalIdx + 1}
                                  </CardTitle>
                                  <Badge variant="outline" className="text-xs">
                                    {Math.ceil((dialogue.conversations?.length || 0) / 2)} 轮
                                  </Badge>
                                  {dialogue.scene && (
                                    <Badge variant="secondary" className="text-xs">
                                      {dialogue.scene}
                                    </Badge>
                                  )}
                                </div>
                                <div className="flex items-center gap-1">
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7"
                                    onClick={() => {
                                      setPreviewIndex(originalIdx);
                                      setPreviewOpen(true);
                                    }}
                                  >
                                    <Eye className="h-3.5 w-3.5" />
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 text-destructive"
                                    onClick={() => handleDeleteDialogue(originalIdx)}
                                  >
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </Button>
                                </div>
                              </div>
                            </CardHeader>
                            <CardContent className="px-4 pb-3 pt-0">
                              <div className="space-y-2">
                                {dialogue.conversations.slice(0, 4).map((conv, convIdx) => (
                                  <div
                                    key={convIdx}
                                    className={`flex gap-2 text-sm ${
                                      conv.from === 'human' ? 'justify-start' : 'justify-start'
                                    }`}
                                  >
                                    <div className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] ${
                                      conv.from === 'human'
                                        ? 'bg-blue-100 text-blue-700'
                                        : 'bg-purple-100 text-purple-700'
                                    }`}>
                                      {conv.from === 'human' ? <User className="h-3 w-3" /> : <Bot className="h-3 w-3" />}
                                    </div>
                                    <p className="text-muted-foreground line-clamp-1 flex-1">
                                      {conv.value}
                                    </p>
                                  </div>
                                ))}
                                {(dialogue.conversations?.length || 0) > 4 && (
                                  <p className="text-xs text-muted-foreground text-center">
                                    ...还有 {Math.ceil(((dialogue.conversations?.length || 0) - 4) / 2)} 轮对话
                                  </p>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <Card className="border-dashed">
                    <CardContent className="pt-6">
                      <div className="text-center py-16">
                        <MessageSquare className="h-16 w-16 mx-auto mb-4 text-muted-foreground/30" />
                        <h3 className="text-lg font-semibold mb-2 text-muted-foreground">
                          {generating ? '正在生成对话...' : '暂无生成数据'}
                        </h3>
                        <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                          在左侧输入角色描述，点击生成按钮，AI 将自动创建符合角色设定的训练对话数据
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>

            {/* 对话预览弹窗 */}
            <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
              <DialogContent className="sm:max-w-[600px] max-h-[85vh] flex flex-col">
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <MessageSquare className="h-5 w-5" />
                    对话 #{previewIndex + 1} 详情
                  </DialogTitle>
                  <DialogDescription>
                    查看和编辑对话内容，确保数据质量
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-3 py-2 overflow-y-auto flex-1 min-h-0">
                  {generatedDialogues[previewIndex]?.conversations.map((conv, convIdx) => (
                    <div key={convIdx} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                            conv.from === 'human'
                              ? 'bg-blue-100 text-blue-700'
                              : 'bg-purple-100 text-purple-700'
                          }`}>
                            {conv.from === 'human' ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
                          </div>
                          <span className="text-xs font-medium text-muted-foreground">
                            {conv.from === 'human' ? '用户' : '角色'}
                          </span>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => handleEditDialogue(previewIndex, convIdx)}
                        >
                          <Edit3 className="h-3 w-3" />
                        </Button>
                      </div>
                      {editingIndex === previewIndex * 1000 + convIdx ? (
                        <div className="ml-8 space-y-2">
                          <Textarea
                            value={editContent}
                            onChange={(e) => setEditContent(e.target.value)}
                            className="min-h-[60px] text-sm"
                          />
                          <div className="flex gap-2">
                            <Button size="sm" onClick={() => handleSaveEdit(previewIndex, convIdx)}>
                              保存
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => setEditingIndex(null)}>
                              取消
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className={`ml-8 p-3 rounded-lg text-sm ${
                          conv.from === 'human'
                            ? 'bg-blue-50 dark:bg-blue-950/20'
                            : 'bg-purple-50 dark:bg-purple-950/20'
                        }`}>
                          {conv.value}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                <DialogFooter className="shrink-0">
                  <Button variant="secondary" onClick={() => setPreviewOpen(false)}>
                    关闭
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </TabsContent>

          {/* 已保存对话 Tab */}
          <TabsContent value="saved" className="space-y-4">
            <SavedDialoguesPanel
              items={savedItems}
              loading={savedLoading}
              searchQuery={savedSearchQuery}
              previewId={previewSaveId}
              previewDialogues={previewSaveDialogues}
              previewLoading={previewSaveLoading}
              onSearchQueryChange={setSavedSearchQuery}
              onRefresh={loadSavedItems}
              onPreview={handlePreviewSaved}
              onCreateDataset={handleCreateDatasetFromSaved}
              onDeleteSaved={handleDeleteSaved}
              onClosePreview={() => {
                setPreviewSaveId(null);
                setPreviewSaveDialogues([]);
              }}
              onDeletePreviewDialogue={handleDeleteDialogueFromSaved}
            />
          </TabsContent>
          {/* 数据集管理 */}
          <TabsContent value="datasets" className="mt-4 space-y-4">
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowScanSection(!showScanSection)}>
                <Search className="mr-2 h-4 w-4" />
                {showScanSection ? '关闭扫描' : '扫描文件夹'}
              </Button>
              <Dialog open={createDatasetOpen} onOpenChange={setCreateDatasetOpen}>
                <DialogTrigger asChild>
                  <Button>
                    <Plus className="mr-2 h-4 w-4" />
                    新建数据集
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[650px] max-h-[85vh] flex flex-col">
                  <DialogHeader>
                    <DialogTitle>创建新数据集</DialogTitle>
                    <DialogDescription>
                      上传或粘贴对话数据来创建训练数据集
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-4 overflow-y-auto flex-1 min-h-0">
                    <div className="space-y-2">
                      <Label htmlFor="dataset-name">数据集名称</Label>
                      <Input
                        id="dataset-name"
                        placeholder="例如：hutao_style"
                        value={newDatasetName}
                        onChange={(e) => updateFormField('newDatasetName', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="dataset-custom-prompt">人物风格 Prompt（可选）</Label>
                      <Textarea
                        id="dataset-custom-prompt"
                        placeholder="请描述您想要的人物对话风格，例如：温柔、内向、略带犹豫的对话风格，常用语气词如「呢…」「呀…」"
                        className="min-h-[120px]"
                        value={newDatasetCustomPrompt}
                        onChange={(e) => updateFormField('newDatasetCustomPrompt', e.target.value)}
                      />
                      <p className="text-xs text-muted-foreground">
                        此 prompt 将作为大模型的参考，帮助模型学习特定的对话风格
                      </p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>训练数据 (JSON)</Label>
                        <div className="flex gap-2">
                          <Button
                            variant={dataInputMode === 'text' ? 'default' : 'secondary'}
                            size="sm"
                            onClick={() => setDataInputMode('text')}
                          >
                            文本输入
                          </Button>
                          <Button
                            variant={dataInputMode === 'file' ? 'default' : 'secondary'}
                            size="sm"
                            onClick={() => setDataInputMode('file')}
                          >
                            文件上传
                          </Button>
                        </div>
                      </div>

                      {dataInputMode === 'text' ? (
                        <Textarea
                          id="dataset-data"
                          placeholder='[{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好呀~"}], "system": "你是..."}]'
                          className="min-h-[200px] font-mono text-sm"
                          value={newDatasetData}
                          onChange={(e) => updateFormField('newDatasetData', e.target.value)}
                        />
                      ) : (
                        <div className="space-y-3">
                          <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                            <input
                              type="file"
                              accept=".json"
                              onChange={handleFileSelect}
                              className="hidden"
                              id="file-upload"
                            />
                            <label
                              htmlFor="file-upload"
                              className="cursor-pointer"
                            >
                              <Upload className="h-8 w-8 mx-auto mb-3 text-muted-foreground" />
                              <p className="text-sm font-medium">
                                {selectedFile ? selectedFile.name : '点击选择 JSON 文件'}
                              </p>
                              <p className="text-xs text-muted-foreground mt-1">
                                {selectedFile
                                  ? `已选择: ${(selectedFile.size / 1024).toFixed(2)} KB`
                                  : '支持 .json 格式文件'}
                              </p>
                            </label>
                          </div>
                          {selectedFile && fileContent && (
                            <div className="bg-muted p-3 rounded-lg">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-sm font-medium">文件预览</span>
                                <span className="text-xs text-muted-foreground">
                                  {(fileContent?.length || 0) > 100
                                    ? `${(fileContent || '').slice(0, 100)}...`
                                    : (fileContent || '')}
                                </span>
                              </div>
                              <p className="text-xs text-muted-foreground">
                                文件大小: {(selectedFile.size / 1024).toFixed(2)} KB
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  <DialogFooter className="shrink-0">
                    <Button variant="secondary" onClick={() => setCreateDatasetOpen(false)}>
                      取消
                    </Button>
                    <Button onClick={handleCreateDataset}>
                      创建数据集
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>

            {/* 扫描文件夹面板 */}
            {showScanSection && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Search className="h-5 w-5" />
                    扫描文件夹导入数据集
                  </CardTitle>
                  <CardDescription>
                    将所有数据集放在同一文件夹下，每个子文件夹为一个数据集（子文件夹名 = 数据集名称）
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      placeholder="输入文件夹路径，留空使用默认目录"
                      value={scanFolderPath}
                      onChange={(e) => setScanFolderPath(e.target.value)}
                      className="flex-1"
                    />
                    <Button onClick={handleScanFolder} disabled={scanLoading}>
                      {scanLoading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          扫描中...
                        </>
                      ) : (
                        <>
                          <Search className="mr-2 h-4 w-4" />
                          开始扫描
                        </>
                      )}
                    </Button>
                  </div>

                  {scanResults.length > 0 && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          发现 {scanResults.length} 个数据集
                        </span>
                        <Button size="sm" variant="outline" onClick={handleImportAll}>
                          <Download className="mr-2 h-3 w-3" />
                          一键导入全部
                        </Button>
                      </div>
                      <div className="space-y-2">
                        {scanResults.map((ds) => (
                          <div
                            key={ds.name}
                            className="flex items-center justify-between p-3 border rounded-lg bg-muted/30"
                          >
                            <div className="flex items-center gap-3 min-w-0">
                              <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
                              <div className="min-w-0">
                                <p className="font-medium truncate">{ds.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {ds.file_count} 个文件 · {ds.files.slice(0, 3).join(', ')}{ds.files.length > 3 ? '...' : ''}
                                </p>
                              </div>
                            </div>
                            <Button
                              size="sm"
                              onClick={() => handleImportDataset(ds.path, ds.name)}
                              disabled={importLoading === ds.name}
                            >
                              {importLoading === ds.name ? (
                                <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                              ) : (
                                <Download className="mr-1 h-3 w-3" />
                              )}
                              导入
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {datasetsLoading ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {[1, 2, 3].map((i) => (
                  <Card key={i}>
                    <CardHeader>
                      <Skeleton className="h-6 w-3/4" />
                      <Skeleton className="h-4 w-full mt-2" />
                    </CardHeader>
                    <CardContent>
                      <Skeleton className="h-4 w-full" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : datasets.length === 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-12">
                    <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                    <h3 className="text-lg font-semibold mb-2">暂无数据集</h3>
                    <p className="text-muted-foreground mb-4">
                      创建您的第一个训练数据集
                    </p>
                    <Button onClick={() => setCreateDatasetOpen(true)}>
                      <Plus className="mr-2 h-4 w-4" />
                      新建数据集
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {datasets.map((dataset) => (
                  <Card key={dataset.name}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="flex items-center gap-2">
                            <FileText className="h-5 w-5 text-muted-foreground" />
                            {dataset.name}
                          </CardTitle>
                          <CardDescription>训练数据集</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">总样本数</span>
                        <Badge variant="secondary">{dataset.stats.total || 0}</Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">训练集</span>
                        <span>{dataset.stats.train || 0}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">验证集</span>
                        <span>{dataset.stats.eval || 0}</span>
                      </div>
                    </CardContent>
                    <CardFooter className="border-t pt-4">
                      <Button
                        variant="outline"
                        className="w-full"
                        onClick={() => handleExportDataset(dataset.name)}
                      >
                        <Download className="mr-2 h-4 w-4" />
                        导出数据集
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* 训练任务 */}
          <TabsContent value="training" className="mt-4 space-y-4">
            <TrainingTasksPanel
              datasets={datasets}
              tasks={trainingTasks}
              loading={tasksLoading}
              dialogOpen={startTrainingOpen}
              submitting={trainingSubmitting}
              onDialogOpenChange={setStartTrainingOpen}
              onStartTraining={handleStartTraining}
              onCancelTraining={handleCancelTraining}
            />
          </TabsContent>

        </Tabs>
      </div>
    </AppLayout>
  );
}
