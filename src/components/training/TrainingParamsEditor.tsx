'use client';

/**
 * LoRA 训练参数编辑器
 *
 * 提供完整的训练参数配置界面：
 * - 必填参数区（基础模型 / 数据集 / 输出名 / 输出目录）
 * - 高级设置区（折叠展开，含一键预设按钮）
 * - 配置导入/导出（JSON）
 * - 数据集上传 + 前端格式校验
 * - 提交前全量合法性校验
 *
 * 通过 api.startTraining({ custom_config }) 透传所有参数到后端（后端会与基础 GPU 配置合并）。
 */

import { useState, useMemo, useCallback, useRef } from 'react';
import {
  Card, CardContent, CardHeader, CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Tooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Upload, Download, Play, ChevronDown, ChevronRight, Zap,
  CheckCircle2, XCircle, AlertCircle, Sparkles, FileJson, RotateCcw,
  Settings,
} from 'lucide-react';
import { toast } from 'sonner';

// ============================================
// 类型定义
// ============================================

/** 必填参数 */
interface RequiredParams {
  baseModel: string;
  outputModelName: string;
  outputDir: string;
  datasetName: string;        // 选用已有数据集时使用
}

/** 高级参数（完整集合，与后端 _build_config + LoRATrainingConfig 一一对应） */
interface AdvancedParams {
  // LoRA 结构
  lora_rank: number;
  lora_alpha: number;
  target_modules: string;       // "all-linear" 或 "q_proj,v_proj"
  lora_dropout: number;
  // 训练策略
  learning_rate: number;
  lr_scheduler_type: 'cosine' | 'constant' | 'linear';
  warmup_ratio: number;          // 0.05 = 5% 步数预热（与 HuggingFace TrainingArguments 对齐）
  num_train_epochs: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
  truncation_direction: 'right' | 'left';
  chat_template: boolean;
  // 正则化与早停
  weight_decay: number;
  max_grad_norm: number;
  early_stopping_patience: number;
  // 硬件与性能
  mixed_precision: 'fp16' | 'bf16' | 'no';
  use_8bit_adam: boolean;
  gradient_checkpointing: boolean;
  packing: boolean;
  neftune_noise_alpha: number;
  use_dora: boolean;
  use_rslora: boolean;
  report_to: 'tensorboard' | 'none';
}

/** 导出/导入的完整配置结构 */
export interface TrainingConfigFile {
  required: RequiredParams;
  advanced: AdvancedParams;
  __format_version: 1;
}

// ============================================
// 默认值 & 预设
// ============================================

const DEFAULT_REQUIRED: RequiredParams = {
  baseModel: 'models/Qwen3-8B-Instruct',
  outputModelName: 'my-lora-chat',
  outputDir: './lora_output',
  datasetName: '',
};

const DEFAULT_ADVANCED: AdvancedParams = {
  lora_rank: 16,
  lora_alpha: 16,
  target_modules: 'all-linear',
  lora_dropout: 0.05,
  learning_rate: 5e-5,
  lr_scheduler_type: 'cosine',
  warmup_ratio: 0.05,
  num_train_epochs: 3,
  per_device_train_batch_size: 2,
  gradient_accumulation_steps: 4,
  max_seq_length: 1024,
  truncation_direction: 'right',
  chat_template: true,
  weight_decay: 0.01,
  max_grad_norm: 0.5,
  early_stopping_patience: 3,
  mixed_precision: 'fp16',
  use_8bit_adam: true,
  gradient_checkpointing: true,
  packing: true,
  neftune_noise_alpha: 5.0,
  use_dora: false,
  use_rslora: false,
  report_to: 'tensorboard',
};

interface Preset {
  name: string;
  label: string;
  desc: string;
  patch: Partial<AdvancedParams>;
}

/** 三套显存预设 */
const PRESETS: Preset[] = [
  {
    name: 'low-vram',
    label: '⚡ 低显存模式（≤6 GB）',
    desc: '极限省显存，速度较慢',
    patch: {
      per_device_train_batch_size: 1,
      lora_rank: 8,
      mixed_precision: 'fp16',
      use_8bit_adam: true,
      gradient_checkpointing: true,
      gradient_accumulation_steps: 4,
      max_seq_length: 512,
    },
  },
  {
    name: 'balanced',
    label: '⚖️ 均衡模式（8~12 GB）',
    desc: '速度与效果的平衡点',
    patch: {
      per_device_train_batch_size: 2,
      lora_rank: 16,
      mixed_precision: 'fp16',
      use_8bit_adam: false,
      gradient_checkpointing: false,
      gradient_accumulation_steps: 2,
      max_seq_length: 1024,
    },
  },
  {
    name: 'high-perf',
    label: '🚀 高性能模式（≥16 GB）',
    desc: '高吞吐，表达力最强',
    patch: {
      per_device_train_batch_size: 4,
      lora_rank: 32,
      mixed_precision: 'bf16',
      use_8bit_adam: false,
      gradient_checkpointing: true,
      gradient_accumulation_steps: 4,
      max_seq_length: 2048,
    },
  },
];

// ============================================
// Props
// ============================================

export interface TrainingParamsEditorProps {
  /** 可选数据集列表（用于下拉选择） */
  datasets: Array<{ name: string; sampleCount: number }>;
  /** 提交训练时回调，参数为 (required, advancedCustomConfig, resolvedDatasetName) */
  onSubmit: (params: {
    loraName: string;
    datasetName: string;
    customConfig: Record<string, unknown>;
  }) => Promise<void>;
  /** 提交按钮是否禁用（如已有任务运行中） */
  submitting?: boolean;
}

// ============================================
// 校验
// ============================================

/** 单字段校验，返回错误消息（合法返回 null） */
function validateField(name: keyof AdvancedParams, value: number | string | boolean): string | null {
  switch (name) {
    case 'learning_rate':
      return typeof value === 'number' && value > 0 ? null : '学习率必须大于 0';
    case 'lora_rank':
    case 'lora_alpha':
    case 'num_train_epochs':
    case 'per_device_train_batch_size':
    case 'gradient_accumulation_steps':
    case 'early_stopping_patience':
      return Number.isInteger(value) && (value as number) >= 1 ? null : '必须为正整数';
    case 'warmup_ratio': {
      const n = value as number;
      return n >= 0 && n <= 1 ? null : '预热比例须为 0~1（如 0.05 = 5%）';
    }
    case 'max_seq_length': {
      const n = value as number;
      return Number.isInteger(n) && n >= 128 && n <= 8192 ? null : '序列长度须为 128~8192 的整数';
    }
    case 'lora_dropout': {
      const n = value as number;
      return n >= 0 && n <= 0.5 ? null : 'dropout 须在 0~0.5 之间';
    }
    case 'weight_decay': {
      const n = value as number;
      return n >= 0 && n <= 1 ? null : '权重衰减须在 0~1 之间';
    }
    case 'max_grad_norm': {
      const n = value as number;
      return n >= 0 ? null : '梯度裁剪值不能为负数';
    }
    default:
      return null;
  }
}

// ============================================
// 小组件：带提示文字的标签行
// ============================================

function FieldRow({
  label, hint, htmlFor, error, children,
}: {
  label: string;
  hint: string;
  htmlFor?: string;
  error?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={htmlFor} className="text-sm font-medium">{label}</Label>
      {children}
      {error ? (
        <p className="text-xs text-red-600 flex items-center gap-1">
          <AlertCircle className="h-3 w-3" />{error}
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">{hint}</p>
      )}
    </div>
  );
}

// ============================================
// 主组件
// ============================================

export function TrainingParamsEditor({
  datasets,
  onSubmit,
  submitting = false,
}: TrainingParamsEditorProps) {
  const [required, setRequired] = useState<RequiredParams>(DEFAULT_REQUIRED);
  const [advanced, setAdvanced] = useState<AdvancedParams>(DEFAULT_ADVANCED);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [showErrors, setShowErrors] = useState(false);
  const [validationMsg, setValidationMsg] = useState<string | null>(null);

  // 数据集上传相关
  const [datasetSource, setDatasetSource] = useState<'select' | 'upload'>('select');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'valid' | 'invalid' | 'parsing'>('idle');
  const [uploadSampleCount, setUploadSampleCount] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  // ── 校验所有高级字段 ──
  const advancedErrors = useMemo(() => {
    const errs: Partial<Record<keyof AdvancedParams, string>> = {};
    (Object.keys(advanced) as (keyof AdvancedParams)[]).forEach((k) => {
      const e = validateField(k, advanced[k]);
      if (e) errs[k] = e;
    });
    return errs;
  }, [advanced]);

  const hasAdvancedErrors = Object.keys(advancedErrors).length > 0;

  // ── 必填字段校验 ──
  const requiredErrors = useMemo(() => {
    const errs: Partial<Record<keyof RequiredParams, string>> = {};
    if (!required.baseModel.trim()) errs.baseModel = '基础模型不能为空';
    if (!required.outputModelName.trim()) errs.outputModelName = '输出模型名不能为空';
    if (!required.outputDir.trim()) errs.outputDir = '输出目录不能为空';
    if (datasetSource === 'select' && !required.datasetName) errs.datasetName = '请选择数据集';
    if (datasetSource === 'upload' && uploadStatus !== 'valid') errs.datasetName = '请上传有效数据集';
    return errs;
  }, [required, datasetSource, uploadStatus]);

  const canSubmit =
    !submitting &&
    Object.keys(requiredErrors).length === 0 &&
    (!advancedOpen || !hasAdvancedErrors);

  // ── 更新高级字段 ──
  const updateAdvanced = useCallback(<K extends keyof AdvancedParams>(key: K, value: AdvancedParams[K]) => {
    setAdvanced((prev) => ({ ...prev, [key]: value }));
    setActivePreset(null); // 手动修改后清除预设标记
  }, []);

  // ── 应用预设 ──
  const applyPreset = useCallback((preset: Preset) => {
    setAdvanced((prev) => ({ ...prev, ...preset.patch }));
    setActivePreset(preset.name);
    toast.success(`已应用 ${preset.label}`, { duration: 2000 });
  }, []);

  // ── 重置全部为默认 ──
  const resetAll = useCallback(() => {
    setRequired(DEFAULT_REQUIRED);
    setAdvanced(DEFAULT_ADVANCED);
    setActivePreset(null);
    setUploadedFile(null);
    setUploadStatus('idle');
    setUploadSampleCount(0);
    setUploadError(null);
    setValidationMsg(null);
    setShowErrors(false);
    toast.info('已重置为默认参数');
  }, []);

  // ── 数据集上传 + 即时校验 ──
  const handleFileSelect = useCallback(async (file: File) => {
    setUploadedFile(file);
    setUploadStatus('parsing');
    setUploadError(null);
    setUploadSampleCount(0);
    try {
      const text = await file.text();
      const ext = file.name.toLowerCase().split('.').pop() ?? '';
      let samples: unknown[] = [];
      if (ext === 'jsonl') {
        // 逐行解析 JSONL
        const lines = text.split(/\r?\n/).filter((l) => l.trim());
        samples = lines.map((l, i) => {
          try { return JSON.parse(l); }
          catch { throw new Error(`第 ${i + 1} 行不是合法 JSON`); }
        });
      } else if (ext === 'json') {
        const parsed = JSON.parse(text);
        samples = Array.isArray(parsed) ? parsed : [parsed];
      } else {
        throw new Error('仅支持 .json / .jsonl 文件');
      }
      // 基础结构校验：每条至少含 conversations 或 messages 字段
      const invalid = samples.findIndex(
        (s) => typeof s === 'object' && s !== null &&
          !(s as Record<string, unknown>).conversations &&
          !(s as Record<string, unknown>).messages
      );
      if (invalid !== -1) {
        throw new Error(`第 ${invalid + 1} 条样本缺少 conversations/messages 字段`);
      }
      setUploadSampleCount(samples.length);
      setUploadStatus('valid');
    } catch (e) {
      setUploadStatus('invalid');
      setUploadError(e instanceof Error ? e.message : '文件解析失败');
    }
  }, []);

  // ── 导出配置 ──
  const handleExport = useCallback(() => {
    const config: TrainingConfigFile = {
      required,
      advanced,
      __format_version: 1,
    };
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${required.outputModelName || 'training-config'}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('配置已导出');
  }, [required, advanced]);

  // ── 导入配置 ──
  const handleImport = useCallback(async (file: File) => {
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Partial<TrainingConfigFile> & {
        advanced?: { use_deepspeed?: unknown };
      };
      const { use_deepspeed = false, ...advanced } = parsed.advanced ?? {};
      if (use_deepspeed !== false) {
        throw new Error('当前不支持 DeepSpeed 训练，请关闭此选项后重新导入');
      }
      if (parsed.required) {
        setRequired({ ...DEFAULT_REQUIRED, ...parsed.required });
      }
      if (parsed.advanced) {
        setAdvanced({ ...DEFAULT_ADVANCED, ...advanced });
      }
      setActivePreset(null);
      toast.success('配置已导入');
    } catch (e) {
      toast.error(e instanceof Error ? `导入失败：${e.message}` : '导入失败：文件不是合法的配置 JSON');
    }
  }, []);

  // ── 提交训练 ──
  const handleSubmit = useCallback(async () => {
    setShowErrors(true);
    if (Object.keys(requiredErrors).length > 0) {
      setValidationMsg('请先修正必填项错误');
      return;
    }
    if (advancedOpen && hasAdvancedErrors) {
      setValidationMsg('请先修正高级参数错误');
      return;
    }
    setValidationMsg(null);

    // 解析数据集名（上传时用文件名，选择时用所选名）
    const datasetName = datasetSource === 'upload'
      ? (uploadedFile?.name.replace(/\.(json|jsonl)$/i, '') ?? 'uploaded')
      : required.datasetName;

    // 组装 custom_config（与后端 _build_config 字段一一对应 + 新增字段透传）
    const customConfig: Record<string, unknown> = {
      // 必填字段
      model_name_or_path: required.baseModel,
      output_dir: `${required.outputDir}/${required.outputModelName}`,
      // LoRA 结构
      lora_rank: advanced.lora_rank,
      lora_alpha: advanced.lora_alpha,
      lora_dropout: advanced.lora_dropout,
      target_modules: advanced.target_modules,
      // 训练策略
      learning_rate: advanced.learning_rate,
      lr_scheduler_type: advanced.lr_scheduler_type,
      warmup_ratio: advanced.warmup_ratio,
      num_train_epochs: advanced.num_train_epochs,
      per_device_train_batch_size: advanced.per_device_train_batch_size,
      gradient_accumulation_steps: advanced.gradient_accumulation_steps,
      max_seq_length: advanced.max_seq_length,
      truncation_direction: advanced.truncation_direction,
      chat_template: advanced.chat_template,
      // 正则化与早停
      weight_decay: advanced.weight_decay,
      max_grad_norm: advanced.max_grad_norm,
      early_stopping_patience: advanced.early_stopping_patience,
      // 硬件与性能
      fp16: advanced.mixed_precision === 'fp16',
      bf16: advanced.mixed_precision === 'bf16',
      use_gradient_checkpointing: advanced.gradient_checkpointing,
      use_8bit_adam: advanced.use_8bit_adam,
      packing: advanced.packing,
      neftune_noise_alpha: advanced.neftune_noise_alpha,
      use_dora: advanced.use_dora,
      use_rslora: advanced.use_rslora,
      report_to: advanced.report_to,
    };

    try {
      await onSubmit({
        loraName: required.outputModelName,
        datasetName,
        customConfig,
      });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '启动训练失败');
    }
  }, [
    requiredErrors, advancedOpen, hasAdvancedErrors, required, advanced,
    datasetSource, uploadedFile, onSubmit,
  ]);

  // ============================================
  // 渲染辅助
  // ============================================
  const errClass = (has: boolean) =>
    showErrors && has ? 'border-red-500 focus-visible:border-red-500' : '';

  return (
    <TooltipProvider delayDuration={200}>
      <div className="space-y-6">
        {/* ────────── 必填参数区 ────────── */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4 text-primary" />
              必填参数
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 基础模型 */}
            <FieldRow
              label="基础模型"
              htmlFor="base-model"
              hint="HuggingFace 模型 ID 或本地路径，例如 meta-llama/Llama-3-8B-Instruct"
              error={showErrors ? requiredErrors.baseModel : null}
            >
              <Input
                id="base-model"
                value={required.baseModel}
                onChange={(e) => setRequired((p) => ({ ...p, baseModel: e.target.value }))}
                className={errClass(!!requiredErrors.baseModel)}
              />
            </FieldRow>

            {/* 数据集：选择 or 上传 */}
            <FieldRow
              label="训练数据集"
              hint="支持 JSON/JSONL，格式见数据集。上传后自动校验。"
              error={showErrors ? requiredErrors.datasetName : null}
            >
              <div className="flex gap-2 mb-2">
                <Button
                  type="button"
                  size="sm"
                  variant={datasetSource === 'select' ? 'default' : 'outline'}
                  onClick={() => setDatasetSource('select')}
                >
                  选择已有
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant={datasetSource === 'upload' ? 'default' : 'outline'}
                  onClick={() => setDatasetSource('upload')}
                >
                  <Upload className="mr-1 h-3.5 w-3.5" />上传文件
                </Button>
              </div>

              {datasetSource === 'select' ? (
                <Select
                  value={required.datasetName}
                  onValueChange={(v) => setRequired((p) => ({ ...p, datasetName: v }))}
                >
                  <SelectTrigger className={errClass(!!requiredErrors.datasetName)}>
                    <SelectValue placeholder="选择数据集" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.map((d) => (
                      <SelectItem key={d.name} value={d.name}>
                        {d.name} ({d.sampleCount} 样本)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <div className="space-y-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json,.jsonl"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFileSelect(f);
                    }}
                  />
                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="mr-1 h-3.5 w-3.5" />选择文件
                    </Button>
                    {uploadedFile && (
                      <span className="text-xs text-muted-foreground truncate max-w-[200px]">
                        {uploadedFile.name}
                      </span>
                    )}
                  </div>
                  {/* 上传状态 */}
                  {uploadStatus === 'parsing' && (
                    <p className="text-xs text-blue-600">校验中…</p>
                  )}
                  {uploadStatus === 'valid' && (
                    <p className="text-xs text-green-600 flex items-center gap-1">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      格式正确，共 {uploadSampleCount} 条样本
                    </p>
                  )}
                  {uploadStatus === 'invalid' && (
                    <p className="text-xs text-red-600 flex items-center gap-1">
                      <XCircle className="h-3.5 w-3.5" />
                      {uploadError ?? '文件格式错误'}
                    </p>
                  )}
                </div>
              )}
            </FieldRow>

            {/* 输出模型名 */}
            <FieldRow
              label="输出模型名"
              htmlFor="output-name"
              hint="训练生成的 LoRA 文件夹名称，会保存在输出目录下。"
              error={showErrors ? requiredErrors.outputModelName : null}
            >
              <Input
                id="output-name"
                value={required.outputModelName}
                onChange={(e) => setRequired((p) => ({ ...p, outputModelName: e.target.value }))}
                className={errClass(!!requiredErrors.outputModelName)}
              />
            </FieldRow>

            {/* 输出目录 */}
            <FieldRow
              label="输出目录"
              htmlFor="output-dir"
              hint="LoRA 权重保存的根目录，训练后会在其下创建以模型名命名的子文件夹。"
              error={showErrors ? requiredErrors.outputDir : null}
            >
              <Input
                id="output-dir"
                value={required.outputDir}
                onChange={(e) => setRequired((p) => ({ ...p, outputDir: e.target.value }))}
                className={errClass(!!requiredErrors.outputDir)}
              />
            </FieldRow>
          </CardContent>
        </Card>

        {/* ────────── 高级设置区 ────────── */}
        <Card>
          <CardHeader className="pb-3">
            <button
              type="button"
              className="flex items-center gap-2 text-base font-semibold w-full text-left"
              onClick={() => setAdvancedOpen((v) => !v)}
            >
              {advancedOpen
                ? <ChevronDown className="h-4 w-4" />
                : <ChevronRight className="h-4 w-4" />}
              <Settings className="h-4 w-4 text-primary" />
              高级设置
              {activePreset && (
                <Badge variant="secondary" className="ml-2 text-xs">
                  {PRESETS.find((p) => p.name === activePreset)?.label.split('（')[0]}
                </Badge>
              )}
            </button>
          </CardHeader>

          {advancedOpen && (
            <CardContent className="space-y-5">
              {/* 一键预设按钮 */}
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">一键应用预设（覆盖相关参数）：</p>
                <div className="flex flex-wrap gap-2">
                  {PRESETS.map((p) => (
                    <Tooltip key={p.name}>
                      <TooltipTrigger asChild>
                        <Button
                          type="button"
                          size="sm"
                          variant={activePreset === p.name ? 'default' : 'outline'}
                          onClick={() => applyPreset(p)}
                        >
                          <Zap className="mr-1 h-3.5 w-3.5" />
                          {p.label}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>{p.desc}</TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              </div>

              <Separator />

              {/* LoRA 结构 */}
              <div>
                <h4 className="text-sm font-medium mb-3 text-muted-foreground">LoRA 结构</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FieldRow
                    label="秩 r"
                    hint="LoRA 秩，越大表达能力越强但文件更大。对话角色训练推荐 8~64"
                    error={showErrors ? advancedErrors.lora_rank : null}
                  >
                    <Input
                      type="number"
                      value={advanced.lora_rank}
                      onChange={(e) => updateAdvanced('lora_rank', parseInt(e.target.value || '0', 10))}
                      className={errClass(!!advancedErrors.lora_rank)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="缩放系数 alpha"
                    hint="通常设为与 r 相同，一般无需单独调整"
                    error={showErrors ? advancedErrors.lora_alpha : null}
                  >
                    <Input
                      type="number"
                      value={advanced.lora_alpha}
                      onChange={(e) => updateAdvanced('lora_alpha', parseInt(e.target.value || '0', 10))}
                      className={errClass(!!advancedErrors.lora_alpha)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="目标模块"
                    hint="all-linear 表示所有线性层，也可指定 q_proj,v_proj，多个用逗号分隔"
                  >
                    <Input
                      value={advanced.target_modules}
                      onChange={(e) => updateAdvanced('target_modules', e.target.value)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="dropout"
                    hint="防止过拟合，数据少时可调至 0.1"
                    error={showErrors ? advancedErrors.lora_dropout : null}
                  >
                    <Input
                      type="number"
                      step={0.01}
                      min={0}
                      max={0.5}
                      value={advanced.lora_dropout}
                      onChange={(e) => updateAdvanced('lora_dropout', parseFloat(e.target.value || '0'))}
                      className={errClass(!!advancedErrors.lora_dropout)}
                    />
                  </FieldRow>
                </div>
              </div>

              <Separator />

              {/* 训练策略 */}
              <div>
                <h4 className="text-sm font-medium mb-3 text-muted-foreground">训练策略</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FieldRow
                    label="学习率"
                    hint="关键参数！对话微调常用 5e-5~2e-4。loss 震荡则减小，收敛慢则增大"
                    error={showErrors ? advancedErrors.learning_rate : null}
                  >
                    <Input
                      type="number"
                      step={1e-6}
                      value={advanced.learning_rate}
                      onChange={(e) => updateAdvanced('learning_rate', parseFloat(e.target.value || '0'))}
                      className={errClass(!!advancedErrors.learning_rate)}
                    />
                  </FieldRow>
                  <FieldRow label="学习率调度器" hint="cosine 逐渐衰减，更稳定">
                    <Select
                      value={advanced.lr_scheduler_type}
                      onValueChange={(v) => updateAdvanced('lr_scheduler_type', v as AdvancedParams['lr_scheduler_type'])}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="cosine">cosine</SelectItem>
                        <SelectItem value="constant">constant</SelectItem>
                        <SelectItem value="linear">linear</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldRow>
                  <FieldRow
                    label="预热比例"
                    hint="学习率线性增加到目标值的步数比例，0.05=5% 步数（与 HuggingFace warmup_ratio 对齐）"
                    error={showErrors ? advancedErrors.warmup_ratio : null}
                  >
                    <Input
                      type="number"
                      step={0.01}
                      min={0}
                      max={1}
                      value={advanced.warmup_ratio}
                      onChange={(e) => updateAdvanced('warmup_ratio', parseFloat(e.target.value || '0'))}
                      className={errClass(!!advancedErrors.warmup_ratio)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="权重衰减"
                    hint="权重衰减正则化系数（weight_decay），防止过拟合，常用 0.01"
                    error={showErrors ? advancedErrors.weight_decay : null}
                  >
                    <Input
                      type="number"
                      step={0.001}
                      min={0}
                      max={1}
                      value={advanced.weight_decay}
                      onChange={(e) => updateAdvanced('weight_decay', parseFloat(e.target.value || '0'))}
                      className={errClass(!!advancedErrors.weight_decay)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="梯度裁剪"
                    hint="梯度范数裁剪阈值（max_grad_norm），防止梯度爆炸，常用 0.5~1.0"
                    error={showErrors ? advancedErrors.max_grad_norm : null}
                  >
                    <Input
                      type="number"
                      step={0.1}
                      min={0}
                      value={advanced.max_grad_norm}
                      onChange={(e) => updateAdvanced('max_grad_norm', parseFloat(e.target.value || '0'))}
                      className={errClass(!!advancedErrors.max_grad_norm)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="早停耐心值"
                    hint="验证指标连续不提升 N 轮后提前停止，0 表示不启用早停"
                    error={showErrors ? advancedErrors.early_stopping_patience : null}
                  >
                    <Input
                      type="number"
                      step={1}
                      min={0}
                      max={20}
                      value={advanced.early_stopping_patience}
                      onChange={(e) => updateAdvanced('early_stopping_patience', parseInt(e.target.value || '0', 10))}
                      className={errClass(!!advancedErrors.early_stopping_patience)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="训练轮数"
                    hint="数据少可增至 5~10，过多易过拟合"
                    error={showErrors ? advancedErrors.num_train_epochs : null}
                  >
                    <Input
                      type="number"
                      value={advanced.num_train_epochs}
                      onChange={(e) => updateAdvanced('num_train_epochs', parseInt(e.target.value || '0', 10))}
                      className={errClass(!!advancedErrors.num_train_epochs)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="批量大小"
                    hint="每次输入模型样本数，受显存限制，建议 1 或 2 起步"
                    error={showErrors ? advancedErrors.per_device_train_batch_size : null}
                  >
                    <Input
                      type="number"
                      value={advanced.per_device_train_batch_size}
                      onChange={(e) => updateAdvanced('per_device_train_batch_size', parseInt(e.target.value || '0', 10))}
                      className={errClass(!!advancedErrors.per_device_train_batch_size)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="梯度累积步数"
                    hint="等效增大 batch，不增加显存。实际 batch = 批量大小 × 此值"
                    error={showErrors ? advancedErrors.gradient_accumulation_steps : null}
                  >
                    <Input
                      type="number"
                      value={advanced.gradient_accumulation_steps}
                      onChange={(e) => updateAdvanced('gradient_accumulation_steps', parseInt(e.target.value || '0', 10))}
                      className={errClass(!!advancedErrors.gradient_accumulation_steps)}
                    />
                  </FieldRow>
                  <FieldRow
                    label="最大序列长度"
                    hint="超长截断，根据对话平均长度调整"
                    error={showErrors ? advancedErrors.max_seq_length : null}
                  >
                    <Input
                      type="number"
                      value={advanced.max_seq_length}
                      onChange={(e) => updateAdvanced('max_seq_length', parseInt(e.target.value || '0', 10))}
                      className={errClass(!!advancedErrors.max_seq_length)}
                    />
                  </FieldRow>
                  <FieldRow label="文本截断方向" hint="right 保留早期内容，left 保留最新内容">
                    <Select
                      value={advanced.truncation_direction}
                      onValueChange={(v) => updateAdvanced('truncation_direction', v as AdvancedParams['truncation_direction'])}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="right">right（保留早期）</SelectItem>
                        <SelectItem value="left">left（保留最新）</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldRow>
                </div>
                {/* 添加聊天模板 复选框 */}
                <div className="flex items-center gap-2 mt-3">
                  <Checkbox
                    id="chat-template"
                    checked={advanced.chat_template}
                    onCheckedChange={(v) => updateAdvanced('chat_template', v === true)}
                  />
                  <div>
                    <Label htmlFor="chat-template" className="text-sm font-medium cursor-pointer">
                      添加聊天模板
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      自动拼接模型专用对话格式，已预处理则取消
                    </p>
                  </div>
                </div>
              </div>

              <Separator />

              {/* 硬件与性能适配 */}
              <div>
                <h4 className="text-sm font-medium mb-3 text-muted-foreground">硬件与性能适配</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FieldRow label="混合精度" hint="NVIDIA 卡推荐 fp16，30 系以上可试 bf16">
                    <Select
                      value={advanced.mixed_precision}
                      onValueChange={(v) => updateAdvanced('mixed_precision', v as AdvancedParams['mixed_precision'])}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="fp16">fp16</SelectItem>
                        <SelectItem value="bf16">bf16</SelectItem>
                        <SelectItem value="no">no（禁用）</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldRow>
                </div>
                {/* 复选项 */}
                <div className="space-y-3 mt-3">
                  <div className="flex items-start gap-2">
                    <Checkbox
                      id="use-8bit-adam"
                      checked={advanced.use_8bit_adam}
                      onCheckedChange={(v) => updateAdvanced('use_8bit_adam', v === true)}
                      className="mt-0.5"
                    />
                    <div>
                      <Label htmlFor="use-8bit-adam" className="text-sm font-medium cursor-pointer">
                        使用 8bit Adam 优化器
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        大幅降低显存，效果几乎无损，低显存强烈推荐
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Checkbox
                      id="grad-checkpoint"
                      checked={advanced.gradient_checkpointing}
                      onCheckedChange={(v) => updateAdvanced('gradient_checkpointing', v === true)}
                      className="mt-0.5"
                    />
                    <div>
                      <Label htmlFor="grad-checkpoint" className="text-sm font-medium cursor-pointer">
                        启用梯度检查点
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        用时间换显存，低配用户建议开启
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 border-t pt-3">
                    <FieldRow label="NEFTune noise alpha" hint="0 disables embedding-noise regularization; 5 is the conservative default.">
                      <Input
                        type="number"
                        min={0}
                        step={0.5}
                        value={advanced.neftune_noise_alpha}
                        onChange={(e) => updateAdvanced('neftune_noise_alpha', parseFloat(e.target.value || '0'))}
                      />
                    </FieldRow>
                    <FieldRow label="Training logs" hint="TensorBoard writes runs into the LoRA output directory.">
                      <Select value={advanced.report_to} onValueChange={(v) => updateAdvanced('report_to', v as AdvancedParams['report_to'])}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="tensorboard">TensorBoard</SelectItem>
                          <SelectItem value="none">Disabled</SelectItem>
                        </SelectContent>
                      </Select>
                    </FieldRow>
                  </div>
                  <div className="flex items-start gap-2">
                    <Checkbox id="sequence-packing" checked={advanced.packing} onCheckedChange={(v) => updateAdvanced('packing', v === true)} className="mt-0.5" />
                    <Label htmlFor="sequence-packing" className="text-sm font-medium cursor-pointer">Sequence packing</Label>
                  </div>
                  <div className="flex items-start gap-2">
                    <Checkbox id="use-dora" checked={advanced.use_dora} onCheckedChange={(v) => updateAdvanced('use_dora', v === true)} className="mt-0.5" />
                    <Label htmlFor="use-dora" className="text-sm font-medium cursor-pointer">DoRA</Label>
                  </div>
                  <div className="flex items-start gap-2">
                    <Checkbox id="use-rslora" checked={advanced.use_rslora} onCheckedChange={(v) => updateAdvanced('use_rslora', v === true)} className="mt-0.5" />
                    <Label htmlFor="use-rslora" className="text-sm font-medium cursor-pointer">RSLoRA</Label>
                  </div>
                </div>
              </div>
            </CardContent>
          )}
        </Card>

        {/* ────────── 校验提示 ────────── */}
        {validationMsg && (
          <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 dark:bg-red-950/30 px-3 py-2 rounded-md">
            <AlertCircle className="h-4 w-4" />
            {validationMsg}
          </div>
        )}

        {/* ────────── 操作按钮区 ────────── */}
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={handleSubmit} disabled={!canSubmit}>
            <Play className="mr-2 h-4 w-4" />
            开始训练
          </Button>
          <Button variant="outline" onClick={handleExport}>
            <Download className="mr-2 h-4 w-4" />
            导出配置
          </Button>
          <Button variant="outline" onClick={() => importInputRef.current?.click()}>
            <FileJson className="mr-2 h-4 w-4" />
            导入配置
          </Button>
          <Button variant="ghost" onClick={resetAll}>
            <RotateCcw className="mr-2 h-4 w-4" />
            重置
          </Button>
          <input
            ref={importInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleImport(f);
            }}
          />
        </div>
      </div>
    </TooltipProvider>
  );
}
