import type {
  ActiveKnowledgeBasesResponse,
  AdapterCheckResponse,
  CreateDatasetFromSavedResponse,
  EvaluationRunRequest,
  EvaluationRunStartResponse,
  EvaluationRunsResponse,
  ExperimentStartRequest,
  ExperimentStartResponse,
  ExperimentsResponse,
  FeedbackListResponse,
  GoldSetResponse,
  IntentModelInfo,
  IntentSamplesResponse,
  IntentTaskStatus,
  IntentTrainingStatus,
  PreferenceCandidate,
  PreferenceCreateRequest,
  PreferenceExportRecord,
  PreferenceExportRequest,
  PreferencePairRecord,
  PreferenceReviewStatus,
  PreferenceSampleRequest,
  PreferenceUpdateRequest,
  RouterAdaptersResponse,
  RouterConfigRecord,
  RouterConfigResponse,
  RouterLogsResponse,
  StatsMetricsResponse,
  UserDataResponse,
} from './api-contracts';

export type * from './api-contracts';

/**
 * API 客户端模块
 *
 * 封装所有与后端 FastAPI 服务的通信接口，通过 Next.js API Route 代理转发请求。
 * 提供统一的类型定义和 ApiClient 类，涵盖健康检查、统计数据、消息记录、
 * LoRA 模型管理、训练任务管理、知识库管理等所有后端 API 调用。
 *
 * @module api
 * @example
 * import { api } from '@/lib/api';
 * const stats = await api.getStats();
 */

const API_BASE_URL = '/api';

/** 系统统计响应 - 包含今日回复数、响应时间、活跃会话、资源使用等指标 */
export interface StatsResponse {
  todayReplies: number;
  avgResponseTime: number;
  activeSessions: number;
  modelLoad: number;
  cpuUsage: number;
  gpuMemory: {
    used: number;
    total: number;
  };
  memoryUsage: {
    used: number;
    total: number;
  };
  diskUsage: {
    used: number;
    total: number;
  };
}

/** 消息记录 - 记录一次完整的对话交互（用户消息 + 机器人回复） */
export interface Message {
  id: string;
  sessionType: 'group' | 'private' | 'channel';
  sessionId: string;
  sessionName: string;
  platform?: string;
  adapter?: string;
  conversationId?: string;
  senderId?: string;
  sourceMessageId?: string;
  traceId?: string;
  userId: string;
  userName: string;
  message: string;
  reply: string;
  modelName: string;
  loraName: string;
  costTime: number;
  createdAt: string;
}

/** 消息列表分页响应 */
export interface MessagesResponse {
  messages: Message[];
  total: number;
  total_all: number;
}

/** 消息列表查询与批量删除共用的筛选条件 */
export interface MessageFilters {
  platform?: string;
  sessionType?: string;
  search?: string;
  lora?: string;
  sessionName?: string;
}

/** 会话摘要 */
export interface SessionSummary {
  sessionId: string;
  sessionType: 'group' | 'private' | 'channel';
  sessionName: string;
  platform?: string;
  adapter?: string;
  conversationId?: string;
  messageCount: number;
  lastActive: string;
  summary: string;
  botEnabled: boolean;
}

/** 会话列表响应 */
export interface SessionsResponse {
  sessions: SessionSummary[];
}

/** LoRA 微调模型 - 个性化风格定制模型 */
export interface LoraModel {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive';
  style: string;
  size: string;
  trainedSteps: number;
  totalSteps: number;
  createdAt: string;
}

/** LoRA 模型列表响应 */
export interface LorasResponse {
  loras: LoraModel[];
  total: number;
}

/** 健康检查响应 */
export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
}

/** 消息生成请求参数 */
export interface GenerateRequest {
  message: string;
  sessionType?: string;
  sessionId?: string;
  userId?: string;
  userName?: string;
  loraName?: string;
}

/** 消息生成响应 */
export interface GenerateResponse {
  reply: string;
  model: string;
  costTime: number;
}

/** 单个时间点的活动统计数据 */
export interface ActivityData {
  time: string;
  messages: number;
  replies: number;
}

/** 活动趋势响应 - 24小时内的消息与回复趋势 */
export interface ActivityResponse {
  activity: ActivityData[];
}

/** 单个服务的运行状态 */
export interface ServiceStatus {
  name: string;
  status: 'running' | 'stopped' | 'connecting';
  uptime: string;
}

/** 服务状态列表响应 */
export interface ServicesResponse {
  services: ServiceStatus[];
}

/** 训练数据集信息 */
export interface Dataset {
  name: string;
  path: string;
  stats: {
    total?: number;
    train?: number;
    eval?: number;
  };
}

/** 数据集列表响应 */
export interface DatasetsResponse {
  success: boolean;
  datasets: Dataset[];
}

/** 模型训练配置 - 针对不同 GPU 的优化参数 */
export interface ModelConfig {
  name: string;
  model_name: string;
  gpu_type: string;
  batch_size: number;
  gradient_accumulation_steps: number;
  cutoff_len: number;
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;
  learning_rate: number;
  num_train_epochs: number;
  warmup_ratio: number;
  weight_decay: number;
  bf16: boolean;
  fp16: boolean;
  load_in_4bit: boolean;
  use_gradient_checkpointing: boolean;
  description: string;
}

/** 模型配置列表响应 */
export interface ModelConfigsResponse {
  success: boolean;
  configs: ModelConfig[];
}

/** 训练任务 - 包含任务 ID、状态、进度等信息 */
export interface TrainingTask {
  task_id: string;
  lora_name: string;
  status: 'pending' | 'training' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  created_at: string;
  updated_at?: string;
  error_message?: string;
  config?: Record<string, unknown>;
}

/** 训练任务列表响应 */
export interface TrainingTasksResponse {
  success: boolean;
  tasks: TrainingTask[];
}

/** 单个训练任务响应 */
export interface TrainingTaskResponse {
  success: boolean;
  task: TrainingTask;
}

/** 人物风格预设 */
export interface CharacterStyle {
  name: string;
  display_name: string;
  description: string;
}

/** 风格列表响应 */
export interface StylesResponse {
  success: boolean;
  styles: CharacterStyle[];
}

/** 创建数据集请求参数 */
export interface CreateDatasetRequest {
  dataset_name: string;
  style?: string;
  custom_prompt?: string;
  data: unknown[];
}

/** 启动训练请求参数 */
export interface StartTrainingRequest {
  lora_name: string;
  dataset_name: string;
  model_type?: string;
  custom_config?: Record<string, unknown>;
}

/** 对话生成请求 */
export interface DialogueGenerateRequest {
  character_description: string;
  num_dialogues?: number;
  style?: string;
  custom_prompt?: string;
}

/** 单条对话 */
export interface DialogueConversation {
  conversations: Array<{
    from: 'human' | 'gpt';
    value: string;
  }>;
  system?: string;
  scene?: string;
  tags?: string[];
}

/** 对话生成响应 */
export interface DialogueGenerateResponse {
  success: boolean;
  dialogues: DialogueConversation[];
  total: number;
  cost_time: number;
  cancelled?: boolean;
}

/** 对话生成进度 */
export interface DialogueGenerationProgress {
  is_generating: boolean;
  cancel_requested: boolean;
  progress: number;
  total: number;
  batch_num: number;
  total_batches: number;
  generated_count: number;
  started_at: number | null;
  /** 本批次新增的对话，实时推送到前端 */
  new_dialogues: DialogueConversation[];
  /** 所有已生成的对话（持久累积，用于断线重连恢复） */
  all_generated_dialogues: DialogueConversation[];
}

/** 知识库文档 */
export interface KnowledgeDocument {
  id: number;
  title: string;
  content: string;
  category: string;
  knowledge_base_id: number | null;
  folder_id: number | null;
  sourceType: string;
  sourceUrl?: string;
  fileType?: string;
  fileSize?: number;
  chunkCount: number;
  createdAt: string;
  updatedAt: string;
}

/** 知识库文档分块 */
export interface KnowledgeChunk {
  id: number;
  documentId: number;
  chunkIndex: number;
  content: string;
  embedding?: number[];
  createdAt: string;
}

/** 知识库（顶层容器） */
export interface KnowledgeBase {
  id: number;
  name: string;
  description: string;
  documentCount: number;
  folderCount: number;
  created_at: string;
  updated_at: string;
}

/** 知识库文件夹 */
export interface KnowledgeFolder {
  id: number;
  knowledge_base_id: number;
  name: string;
  description: string;
  documentCount: number;
  created_at: string;
  updated_at: string;
}

/** ZIP上传响应 */
export interface ZipUploadResponse {
  success: boolean;
  message: string;
  createdFolders: string[];
  createdDocs: number;
  errors: string[];
}

/** 扫描目录树节点 */
export interface ScanDirectory {
  name: string;
  type: 'folder' | 'file';
  children?: ScanDirectory[];
  fileCount?: number;
  totalSize?: number;
  size?: number;
  extension?: string;
}

/** 知识库搜索结果项 - 包含相关性评分 */
export interface KnowledgeSearchResult {
  documentId: number;
  documentTitle: string;
  /** 后端多数分支不返回 chunkId，仅部分向量检索路径可能携带 */
  chunkId?: number;
  chunkIndex: number;
  content: string;
  score: number;
}

/** 搜索结果类型，与后端 search_knowledge 返回的 searchType 字段对应 */
export type KnowledgeSearchType = "rag_pipeline" | "hybrid" | "keyword" | "empty";

/** 知识库统计数据 */
export interface KnowledgeStats {
  totalDocuments: number;
  totalChunks: number;
  totalCharacters: number;
}

/** 知识库文档列表响应 */
export interface KnowledgeDocumentsResponse {
  success: boolean;
  documents: KnowledgeDocument[];
  stats: KnowledgeStats;
}

/** 单个知识库文档响应 */
export interface KnowledgeDocumentResponse {
  success: boolean;
  document: KnowledgeDocument;
  chunks: KnowledgeChunk[];
}

/** 创建知识库文档请求参数 */
export interface KnowledgeCreateRequest {
  title: string;
  content: string;
  category?: string;
  knowledge_base_id?: number | null;
  folder_id?: number | null;
  sourceType?: string;
  sourceUrl?: string;
  fileType?: string;
  fileSize?: number;
}

/** 更新知识库文档请求参数 */
export interface KnowledgeUpdateRequest {
  title?: string;
  content?: string;
  category?: string;
  knowledge_base_id?: number | null;
  folder_id?: number | null;
  sourceType?: string;
  sourceUrl?: string;
  fileType?: string;
  fileSize?: number;
}

/** 知识库搜索请求参数 */
export interface KnowledgeSearchRequest {
  query: string;
  topK?: number;
  /** 按知识库名称过滤检索结果，用于意图分类器路由后的精准检索 */
  knowledgeBaseName?: string;
}

/** 知识库搜索响应 */
export interface KnowledgeSearchResponse {
  success: boolean;
  query: string;
  results: KnowledgeSearchResult[];
  /** 检索路径：rag_pipeline | hybrid | keyword | empty
   * 后端 _ensure_vector_index 失败或不存在的 KB 时会降级 */
  searchType: KnowledgeSearchType;
}

/** 知识库统计响应 */
export interface KnowledgeStatsResponse {
  success: boolean;
  stats: KnowledgeStats;
}

/** 系统配置 - 键值对结构，值类型为 string | number | boolean */
export interface SystemConfig {
  [key: string]: string | number | boolean;
}

/** 用户信息 */
export interface User {
  id: number;
  username: string;
  created_at: string;
  /** M3 fix: 用户角色，后端 login/register/me 均已返回该字段。
   * 缺省为 'user'，首个注册用户为 'admin'。用于前端按角色过滤侧栏导航项。
   * 类型层强制必填，反映后端契约；AuthContext 会在解析时补齐缺省值。 */
  role: 'admin' | 'user';
}

/** 注册请求 */
export interface RegisterRequest {
  username: string;
  password: string;
}

/** 登录请求 */
export interface LoginRequest {
  username: string;
  password: string;
}

/** 配置获取响应 */
export interface ConfigResponse {
  config: SystemConfig;
}

/** 配置更新响应 */
export interface ConfigUpdateResponse {
  success: boolean;
  message: string;
  config: SystemConfig;
}

/** 可用模型信息 */
export interface AvailableModel {
  name: string;
  display_name: string;
  repo_id: string;
  size: string;
  description: string;
  downloaded: boolean;
}

/** 模型列表响应 */
export interface ModelsResponse {
  success: boolean;
  models: AvailableModel[];
}

/** 已保存对话列表项 */
export interface SavedDialogueItem {
  id: number;
  name: string;
  character_desc: string;
  style: string | null;
  dialogue_count: number;
  created_at: string;
  updated_at: string;
}

/** 已保存对话完整数据 */
export interface SavedDialogueFull {
  success: boolean;
  id: number;
  name: string;
  character_desc: string;
  style: string | null;
  dialogue_count: number;
  dialogues: DialogueConversation[];
  turn_stats: Record<string, number> | null;
  scene_stats: Record<string, number> | null;
  created_at: string;
  updated_at: string;
}

/** 保存对话请求 */
export interface SaveDialoguesRequest {
  name: string;
  character_desc: string;
  style?: string;
  dialogues: DialogueConversation[];
  turn_stats?: Record<string, number>;
  scene_stats?: Record<string, number>;
}

/** 角色长期记忆隔离范围（与后端 UserScope 一致） */
export interface CharacterMemoryScope {
  platform: string;
  adapter: string;
  sender_id: string;
  conversation_type: string;
  conversation_id: string;
}

/** 角色关系记录 */
export interface CharacterRelationshipRecord {
  character_id: string;
  platform: string;
  adapter: string;
  sender_id: string;
  conversation_type: string;
  conversation_id: string;
  relationship_stage: string;
  preferred_address: string;
  summary: string;
  interaction_count: number;
  created_at: string;
  updated_at: string;
}

/** 角色长期记忆记录 */
export interface CharacterMemoryRecord {
  id: number;
  character_id: string;
  memory_type: string;
  memory_key: string;
  content: string;
  importance: number;
  source_message_id: string | null;
  created_at: string;
  updated_at: string;
}

/** 把隔离范围序列化为 URL 查询串 */
function buildScopeQuery(scope: CharacterMemoryScope): string {
  const params = new URLSearchParams({
    platform: scope.platform,
    adapter: scope.adapter,
    sender_id: scope.sender_id,
    conversation_type: scope.conversation_type,
    conversation_id: scope.conversation_id,
  });
  return params.toString();
}

/**
 * API 客户端类
 *
 * 封装对后端 FastAPI 服务的所有 HTTP 请求，提供类型安全的接口调用。
 * 导出为单例实例 `api`，全局共享使用。
 */
class ApiClient {
  /**
   * 通用请求方法 - 封装 fetch 调用，统一处理认证、请求头和错误
   *
   * 统一特性：
   * - credentials: 'include' 确保 httpOnly Cookie 随请求发送（Edge浏览器兼容）
   * - 自动注入 Content-Type: application/json（有body时）
   * - 统一错误处理：尝试提取后端错误信息
   *
   * @param {string} path - API 路径（相对于 /api）
   * @param {RequestInit} [options] - fetch 选项
   * @returns {Promise<T>} 解析后的 JSON 响应
   */
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const hasBody = options?.body !== undefined;
    const headers: Record<string, string> = {};

    // 有请求体时自动设置 Content-Type（FormData 除外，让浏览器自动设置 boundary）
    if (hasBody && !(options?.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    // 合并自定义 headers
    if (options?.headers) {
      if (options.headers instanceof Headers) {
        options.headers.forEach((v, k) => { headers[k] = v; });
      } else if (Array.isArray(options.headers)) {
        for (const [k, v] of options.headers) { headers[k] = v; }
      } else {
        Object.assign(headers, options.headers);
      }
    }

    const response = await fetch(`${API_BASE_URL}${path}`, {
      credentials: 'include',
      ...options,
      headers,
    });

    if (!response.ok) {
      // 401 时自动清理用户状态并跳转登录页
      if (response.status === 401) {
        this._handle401Redirect();
      }
      // 尝试提取后端错误详情
      let detail = '';
      try {
        const errBody = await response.json();
        detail = errBody?.detail || errBody?.message || JSON.stringify(errBody);
      } catch {
        detail = response.statusText;
      }
      throw new Error(`Request failed: ${response.status} - ${detail}`);
    }

    // 204 No Content
    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  }

  /**
   * 统一的 401 重定向处理：清除本地用户状态、调用后端清 Cookie、跳转登录页。
   * 从 request() 提取出来，供 health()/exportDataset() 等返回特殊类型的方法复用，
   * 避免绕过统一 401 处理导致 token 过期时用户卡在错误状态。
   */
  private _handle401Redirect(): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem('qq_assistant_user');
    if (window.location.pathname.includes('/login')) return;
    // C6 fix: httpOnly Cookie 无法被 JS 清除，必须调用后端 /api/auth/logout。
    // 否则 middleware 仍会看到 Cookie 存在，把 /login 重定向回 /，形成死锁。
    // 使用同步 fetch（带 keepalive 以支持 unload 期间完成）触发后端清 Cookie，
    // 然后带 ?expired=1 跳转，middleware 识别此参数放行到 /login。
    try {
      fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
        keepalive: true,
      }).catch(() => { /* 尽力而为，跳转仍需进行 */ });
    } catch { /* 忽略，继续跳转 */ }
    // 略微延迟跳转让 logout 请求发出，带 expired 参数绕过 middleware 死锁
    setTimeout(() => {
      window.location.href = '/login?expired=1';
    }, 50);
  }

  /**
   * 批量删除消息（基于筛选条件）
   */
  async deleteMessagesBatch(filters: MessageFilters): Promise<{ success: boolean; deleted: number; message: string }> {
    return this.request<{ success: boolean; deleted: number; message: string }>('/messages/batch', {
      method: 'DELETE',
      body: JSON.stringify(filters),
    });
  }

  /**
   * 健康检查 - 检测后端服务是否正常运行
   * @returns {Promise<HealthResponse>} 服务健康状态及时间戳
   */
  async health(): Promise<HealthResponse> {
    // Me3 fix: 走 Next.js 代理路由 /api/health 而非直连 /health。
    // 原先直连 /health 在纯 Next.js dev 模式（无 nginx）下会 404，
    // 因为 Next.js 根路径无对应的 page.tsx/route.ts。
    // src/app/api/health/route.ts 已存在并正确代理到后端 /health。
    const response = await fetch(`${API_BASE_URL}/health`, { credentials: 'include' });
    if (!response.ok) {
      // 复用统一 401 处理（虽然 /health 通常公开，但部署中可能被中间件拦截）
      if (response.status === 401) this._handle401Redirect();
      throw new Error('Health check failed');
    }
    return response.json();
  }

  /**
   * 获取系统统计数据
   * @returns {Promise<StatsResponse>} 包含今日回复数、响应时间、活跃会话、CPU/内存/GPU 等指标
   */
  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>('/stats');
  }

  /**
   * 获取消息记录（分页）
   * @param {number} [limit] - 返回条数限制
   * @param {number} [offset] - 分页偏移量
   * @returns {Promise<MessagesResponse>} 消息列表及总数
   */
  async getMessages(limit?: number, offset?: number, filters?: MessageFilters): Promise<MessagesResponse> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    if (filters?.platform && filters.platform !== 'all') params.append('platform', filters.platform);
    if (filters?.search) params.append('search', filters.search);
    if (filters?.lora && filters.lora !== 'all') params.append('lora', filters.lora);
    if (filters?.sessionName) params.append('sessionName', filters.sessionName);
    if (filters?.sessionType && filters.sessionType !== 'all') params.append('sessionType', filters.sessionType);
    const qs = params.toString();
    return this.request<MessagesResponse>(`/messages${qs ? '?' + qs : ''}`);
  }

  /**
   * 删除单条消息记录
   * @param {string} id - 消息 ID
   */
  async deleteMessage(id: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/messages/${id}`, {
      method: 'DELETE',
    });
  }

  /**
   * 获取会话列表（聚合统计）
   */
  async getSessionSummaries(): Promise<SessionsResponse> {
    return this.request<SessionsResponse>('/sessions');
  }

  /**
   * 设置会话机器人开关
   */
  async toggleSessionBot(
    sessionId: string,
    enabled: boolean,
    platform = 'qq',
    conversationId?: string,
    conversationType: SessionSummary['sessionType'] = 'private',
  ): Promise<{ success: boolean; sessionId: string; botEnabled: boolean }> {
    return this.request<{ success: boolean; sessionId: string; botEnabled: boolean }>('/sessions/bot-toggle', {
      method: 'PUT',
      body: JSON.stringify({ sessionId, enabled, platform, conversationId, conversationType }),
    });
  }

  /**
   * 获取 LoRA 模型列表
   * @returns {Promise<LorasResponse>} LoRA 模型列表及总数
   */
  async getLoras(): Promise<LorasResponse> {
    const data = await this.request<{ loras: LoraModel[] }>('/loras');
    return {
      loras: data.loras,
      total: data.loras.length
    };
  }

  /**
   * 切换 LoRA 模型的激活状态
   * @param {string} id - LoRA 模型 ID
   * @param {string} currentStatus - 当前状态（active 或 inactive）
   * @returns {Promise<LoraModel>} 更新后的 LoRA 模型信息
   */
  async toggleLoraStatus(id: string, currentStatus: string): Promise<LoraModel> {
    const newStatus = currentStatus === 'active' ? 'inactive' : 'active';
    const result = await this.request<{ lora: LoraModel }>(`/loras/${id}/status`, {
      method: 'PUT',
      body: JSON.stringify({ status: newStatus }),
    });
    return result.lora;
  }

  /**
   * 删除 LoRA 模型
   * @param {string} id - 要删除的 LoRA 模型 ID
   * @returns {Promise<{success: boolean; message: string}>} 删除结果
   */
  async deleteLora(id: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/loras/${id}`, {
      method: 'DELETE',
    });
  }

  /**
   * 扫描 loras 目录，自动发现并注册新的 LoRA 适配器
   */
  async scanLoras(): Promise<{ success: boolean; message: string; new_count: number; updated_count: number; failed_count: number; failed_names: string[] }> {
    return this.request<{ success: boolean; message: string; new_count: number; updated_count: number; failed_count: number; failed_names: string[] }>('/loras/scan', {
      method: 'POST',
    });
  }

  /**
   * 测试消息生成 - 向机器人发送消息并获取回复
   * @param {GenerateRequest} request - 消息内容、会话类型、用户信息
   * @returns {Promise<GenerateResponse>} 机器人回复内容及耗时
   */
  async generateReply(request: GenerateRequest): Promise<GenerateResponse> {
    return this.request<GenerateResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * 获取 24 小时活动趋势数据
   * @returns {Promise<ActivityResponse>} 每小时的接收消息数和回复消息数
   */
  async getActivity(): Promise<ActivityResponse> {
    return this.request<ActivityResponse>('/stats/activity');
  }

  /**
   * 获取各服务运行状态
   * @returns {Promise<ServicesResponse>} 各服务名称、状态、运行时长
   */
  async getServices(): Promise<ServicesResponse> {
    return this.request<ServicesResponse>('/stats/services');
  }

  /**
   * 获取统计指标（AstrBot 网关状态、各平台连接状态等）
   *
   * M4 fix: 此前 integrations 页面用裸 fetch('/api/stats/metrics') 调用，
   * 绕过了 request 助手，导致 401（token 过期）时不跳转登录页，只显示“获取平台指标失败”。
   * 现统一走 request 助手，401 时自动清理用户状态并跳转 /login。
   */
  async getMetrics(): Promise<StatsMetricsResponse> {
    return this.request<StatsMetricsResponse>('/stats/metrics');
  }


  /**
   * 列出所有训练数据集
   * @returns {Promise<DatasetsResponse>} 数据集列表
   */
  async listDatasets(): Promise<DatasetsResponse> {
    return this.request<DatasetsResponse>('/training/datasets');
  }

  /**
   * 创建训练数据集
   * @param {CreateDatasetRequest} request - 数据集名称、风格、自定义提示和原始数据
   * @returns {Promise<{success: boolean; message: string; dataset_name: string}>} 创建结果
   */
  async createDataset(request: CreateDatasetRequest): Promise<{ success: boolean; message: string; dataset_name: string }> {
    return this.request<{ success: boolean; message: string; dataset_name: string }>('/training/datasets', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * 列出可用的模型训练配置（支持 RTX 4060 / RTX 3090 等多种 GPU）
   * @returns {Promise<ModelConfigsResponse>} GPU 优化训练配置列表
   */
  async listModelConfigs(): Promise<ModelConfigsResponse> {
    return this.request<ModelConfigsResponse>('/training/models');
  }

  /**
   * 导出数据集为 ZIP 文件（用于上传到服务器训练）
   * @param {string} datasetName - 数据集名称
   * @returns {Promise<Response>} ZIP 文件的 Response 对象
   */
  async exportDataset(datasetName: string): Promise<Response> {
    const response = await fetch(`${API_BASE_URL}/training/datasets/${encodeURIComponent(datasetName)}/export`, {
      credentials: 'include',
    });
    if (!response.ok && response.status === 401) {
      // 复用统一 401 处理：导出数据集需要登录，token 过期时引导用户重新登录
      this._handle401Redirect();
    }
    return response;
  }

  /**
   * 扫描数据集文件夹
   * @param folder 要扫描的文件夹路径，为空时使用默认目录
   */
  async scanDatasets(folder?: string): Promise<{ success: boolean; scan_path: string; datasets: Array<{ name: string; path: string; file_count: number; files: string[]; valid: boolean }>; count: number }> {
    const params = folder ? `?folder=${encodeURIComponent(folder)}` : '';
    return this.request<{ success: boolean; scan_path: string; datasets: Array<{ name: string; path: string; file_count: number; files: string[]; valid: boolean }>; count: number }>(`/training/datasets/scan${params}`);
  }

  /**
   * 从扫描结果导入数据集
   */
  async importDataset(sourcePath: string, datasetName?: string): Promise<{ success: boolean; name: string; path: string; stats: { total: number; train: number; eval: number } }> {
    return this.request<{ success: boolean; name: string; path: string; stats: { total: number; train: number; eval: number } }>('/training/datasets/scan/import', {
      method: 'POST',
      body: JSON.stringify({ source_path: sourcePath, dataset_name: datasetName }),
    });
  }

  /**
   * 启动 LoRA 训练任务
   * @param {StartTrainingRequest} request - LoRA 名称、数据集名称、模型类型、自定义配置
   * @returns {Promise<{success: boolean; task_id: string; message: string}>} 启动的训练任务信息
   */
  async startTraining(request: StartTrainingRequest): Promise<{ success: boolean; task_id: string; message: string }> {
    return this.request<{ success: boolean; task_id: string; message: string }>('/training/start', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * 列出所有训练任务
   * @returns {Promise<TrainingTasksResponse>} 训练任务列表
   */
  async listTrainingTasks(): Promise<TrainingTasksResponse> {
    return this.request<TrainingTasksResponse>('/training/tasks');
  }

  /**
   * 获取单个训练任务的详细信息
   * @param {string} taskId - 训练任务 ID
   * @returns {Promise<TrainingTaskResponse>} 任务详情（状态、进度等）
   */
  async getTrainingTask(taskId: string): Promise<TrainingTaskResponse> {
    return this.request<TrainingTaskResponse>(`/training/tasks/${taskId}`);
  }

  /**
   * 取消正在运行的训练任务
   * @param {string} taskId - 训练任务 ID
   * @returns {Promise<{success: boolean; message: string}>} 取消结果
   */
  async cancelTrainingTask(taskId: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/training/tasks/${taskId}/cancel`, {
      method: 'POST',
    });
  }

  /**
   * 列出可用的预定义人物风格
   * @returns {Promise<StylesResponse>} 风格列表
   */
  async listStyles(): Promise<StylesResponse> {
    return this.request<StylesResponse>('/training/styles');
  }

  /**
   * 获取知识库文档列表（分页）
   * @param {number} [limit] - 返回条数限制
   * @param {number} [offset] - 分页偏移量
   * @returns {Promise<KnowledgeDocumentsResponse>} 文档列表及统计信息
   */
  async getKnowledgeDocuments(limit?: number, offset?: number, category?: string, knowledgeBaseId?: number, folderId?: number): Promise<KnowledgeDocumentsResponse> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    if (category && category !== '全部') params.append('category', category);
    if (knowledgeBaseId) params.append('knowledge_base_id', knowledgeBaseId.toString());
    if (folderId) params.append('folder_id', folderId.toString());
    const qs = params.toString();
    return this.request<KnowledgeDocumentsResponse>(`/knowledge/documents${qs ? '?' + qs : ''}`);
  }

  /**
   * 获取单个知识库文档及其分块
   * @param {number} docId - 文档 ID
   * @returns {Promise<KnowledgeDocumentResponse>} 文档详情及内容分块
   */
  async getKnowledgeDocument(docId: number): Promise<KnowledgeDocumentResponse> {
    return this.request<KnowledgeDocumentResponse>(`/knowledge/documents/${docId}`);
  }

  /**
   * 创建知识库文档（自动分块和向量化）
   * @param {KnowledgeCreateRequest} request - 文档标题、内容、来源类型等
   * @returns {Promise<{success: boolean; document: KnowledgeDocument}>} 创建结果
   */
  async createKnowledgeDocument(request: KnowledgeCreateRequest): Promise<{ success: boolean; document: KnowledgeDocument }> {
    return this.request<{ success: boolean; document: KnowledgeDocument }>('/knowledge/documents', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * 更新知识库文档
   * @param {number} docId - 文档 ID
   * @param {KnowledgeUpdateRequest} request - 需要更新的字段
   * @returns {Promise<{success: boolean; document: KnowledgeDocument}>} 更新结果
   */
  async updateKnowledgeDocument(docId: number, request: KnowledgeUpdateRequest): Promise<{ success: boolean; document: KnowledgeDocument }> {
    return this.request<{ success: boolean; document: KnowledgeDocument }>(`/knowledge/documents/${docId}`, {
      method: 'PUT',
      body: JSON.stringify(request),
    });
  }

  /**
   * 删除知识库文档及其所有分块
   * @param {number} docId - 文档 ID
   * @returns {Promise<{success: boolean; message: string}>} 删除结果
   */
  async deleteKnowledgeDocument(docId: number): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/knowledge/documents/${docId}`, {
      method: 'DELETE',
    });
  }

  /**
   * 知识库语义搜索 - 结合向量检索和关键词匹配
   * @param {KnowledgeSearchRequest} request - 搜索关键词和返回条数
   * @returns {Promise<KnowledgeSearchResponse>} 按照相关性排序的搜索结果
   */
  async searchKnowledge(request: KnowledgeSearchRequest): Promise<KnowledgeSearchResponse> {
    return this.request<KnowledgeSearchResponse>('/knowledge/search', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * 获取知识库统计数据
   * @returns {Promise<KnowledgeStatsResponse>} 文档总数、分块总数、总字符数
   */
  async getKnowledgeStats(): Promise<KnowledgeStatsResponse> {
    return this.request<KnowledgeStatsResponse>('/knowledge/stats');
  }

  // ============================================
  // 知识库管理 API
  // ============================================

  /** 获取所有知识库 */
  async getKnowledgeBases(): Promise<{ success: boolean; bases: KnowledgeBase[] }> {
    return this.request<{ success: boolean; bases: KnowledgeBase[] }>('/knowledge/bases');
  }

  /** 创建知识库 */
  async createKnowledgeBase(name: string, description: string = ''): Promise<{ success: boolean; base: KnowledgeBase }> {
    return this.request<{ success: boolean; base: KnowledgeBase }>('/knowledge/bases', {
      method: 'POST',
      body: JSON.stringify({ name, description }),
    });
  }

  /** 更新知识库 */
  async updateKnowledgeBase(kbId: number, data: { name?: string; description?: string }): Promise<{ success: boolean; base: KnowledgeBase }> {
    return this.request<{ success: boolean; base: KnowledgeBase }>(`/knowledge/bases/${kbId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  /** 删除知识库 */
  async deleteKnowledgeBase(kbId: number): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/knowledge/bases/${kbId}`, {
      method: 'DELETE',
    });
  }

  /** 获取知识库下的文件夹 */
  async getKnowledgeFolders(kbId: number): Promise<{ success: boolean; folders: KnowledgeFolder[] }> {
    return this.request<{ success: boolean; folders: KnowledgeFolder[] }>(`/knowledge/bases/${kbId}/folders`);
  }

  /** 创建文件夹 */
  async createKnowledgeFolder(kbId: number, name: string, description: string = ''): Promise<{ success: boolean; folder: KnowledgeFolder }> {
    return this.request<{ success: boolean; folder: KnowledgeFolder }>(`/knowledge/bases/${kbId}/folders`, {
      method: 'POST',
      body: JSON.stringify({ name, description }),
    });
  }

  /** 删除文件夹 */
  async deleteKnowledgeFolder(folderId: number): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/knowledge/folders/${folderId}`, {
      method: 'DELETE',
    });
  }

  /** 上传ZIP文件到知识库 */
  async uploadKnowledgeZip(kbId: number, file: File): Promise<ZipUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.request<ZipUploadResponse>(`/knowledge/bases/${kbId}/upload-zip`, {
      method: 'POST',
      body: formData,
    });
  }

  /** 扫描知识库文件夹目录 */
  async scanKnowledgeDirs(): Promise<{ success: boolean; directories: ScanDirectory[]; message?: string }> {
    return this.request<{ success: boolean; directories: ScanDirectory[]; message?: string }>('/knowledge/scan');
  }

  /** 导入扫描到的目录 */
  async importScannedDir(directoryName: string, kbId?: number): Promise<ZipUploadResponse & { knowledgeBase?: KnowledgeBase }> {
    const params = new URLSearchParams();
    params.append('directory_name', directoryName);
    if (kbId) params.append('kb_id', kbId.toString());
    return this.request<ZipUploadResponse & { knowledgeBase?: KnowledgeBase }>(`/knowledge/scan/import?${params.toString()}`, {
      method: 'POST',
    });
  }

  /**
   * 获取系统配置
   * @returns {Promise<ConfigResponse>} 系统配置键值对
   */
  async getConfig(): Promise<ConfigResponse> {
    return this.request<ConfigResponse>('/config');
  }

  /**
   * 更新系统配置
   * @param {SystemConfig} config - 要更新的配置键值对
   * @returns {Promise<ConfigUpdateResponse>} 更新结果
   */
  async updateConfig(config: SystemConfig): Promise<ConfigUpdateResponse> {
    return this.request<ConfigUpdateResponse>('/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  /**
   * 切换模型提供商（admin 端点）
   * H1/H2 fix: 此前 settings/page.tsx 使用裸 fetch 调用，不处理 401 重定向。
   * 现统一通过 api 助手调用，确保 token 过期时自动跳转登录页。
   */
  async setModelProvider(provider: string): Promise<{ success: boolean; provider: string }> {
    return this.request<{ success: boolean; provider: string }>('/model/provider', {
      method: 'PUT',
      body: JSON.stringify({ provider }),
    });
  }

  /**
   * 获取 vLLM 连接状态
   * H1 fix: 此前 settings/page.tsx 使用裸 fetch 调用，既不检查 response.ok
   * 也不处理 401 重定向。现统一通过 api 助手调用。
   */
  async getVllmStatus(): Promise<{
    enabled: boolean;
    summary?: { total: number; healthy: number; all_healthy: boolean };
  }> {
    return this.request('/vllm/status');
  }

  /**
   * 获取可用模型列表
   * @returns {Promise<ModelsResponse>} 模型列表
   */
  async getModels(): Promise<ModelsResponse> {
    return this.request<ModelsResponse>('/models');
  }

  /**
   * 生成对话数据
   * @param {DialogueGenerateRequest} request - 角色描述、对话数量等
   * @returns {Promise<DialogueGenerateResponse>} 生成的对话数据
   */
  async generateDialogues(request: DialogueGenerateRequest, signal?: AbortSignal): Promise<DialogueGenerateResponse> {
    return this.request<DialogueGenerateResponse>('/training/generate-dialogues', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  /** 取消对话生成 */
  async cancelDialogueGeneration(): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>('/training/generate-dialogues/cancel', {
      method: 'POST',
    });
  }

  /** 强制重置生成状态（用于断线重连后清理残留状态） */
  async forceResetGeneration(): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>('/training/generate-dialogues/force-reset', {
      method: 'POST',
    });
  }

  /** 获取对话生成进度 */
  async getDialogueGenerationProgress(): Promise<DialogueGenerationProgress> {
    return this.request<DialogueGenerationProgress>('/training/generate-dialogues/progress');
  }

  /** 用户注册 */
  async register(request: RegisterRequest): Promise<{ success: boolean; user: User }> {
    return this.request<{ success: boolean; user: User }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /** 用户登录 */
  async login(request: LoginRequest): Promise<{ success: boolean; user: User }> {
    return this.request<{ success: boolean; user: User }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /** 获取当前用户信息 */
  async getCurrentUser(): Promise<{ success: boolean; user: User }> {
    return this.request<{ success: boolean; user: User }>('/auth/me');
  }

  /** 保存用户表单数据 */
  async saveUserData(pageKey: string, dataJson: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>('/user/data', {
      method: 'PUT',
      body: JSON.stringify({ page_key: pageKey, data_json: dataJson }),
    });
  }

  /** 获取用户表单数据 */
  async getUserData(pageKey?: string): Promise<UserDataResponse> {
    const params = new URLSearchParams();
    if (pageKey) params.set('page_key', pageKey);
    const qs = params.toString();
    return this.request<UserDataResponse>(`/user/data${qs ? '?' + qs : ''}`);
  }

  /** 保存对话数据 */
  async saveDialogues(request: SaveDialoguesRequest): Promise<{ success: boolean; id: number; message: string }> {
    return this.request<{ success: boolean; id: number; message: string }>('/training/saved-dialogues', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /** 列出所有已保存对话 */
  async listSavedDialogues(): Promise<{ success: boolean; items: SavedDialogueItem[] }> {
    return this.request<{ success: boolean; items: SavedDialogueItem[] }>('/training/saved-dialogues');
  }

  /** 获取单条已保存对话 */
  async getSavedDialogue(id: number): Promise<SavedDialogueFull> {
    return this.request<SavedDialogueFull>(`/training/saved-dialogues/${id}`);
  }

  /** 删除已保存对话 */
  async deleteSavedDialogue(id: number): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/training/saved-dialogues/${id}`, {
      method: 'DELETE',
    });
  }

  // ── Claw 工具管理 ──

  /**
   * 列出所有 Claw 工具
   */
  async listClawTools(): Promise<{ success: boolean; tools: Array<{ name: string; description: string; code: string; enabled: boolean; builtin: boolean; created_at?: string; updated_at?: string }>; total: number }> {
    return this.request<{ success: boolean; tools: Array<{ name: string; description: string; code: string; enabled: boolean; builtin: boolean; created_at?: string; updated_at?: string }>; total: number }>('/claw/tools');
  }

  /**
   * 保存（创建/更新）Claw 工具
   */
  async saveClawTool(data: { name: string; description: string; code: string; enabled: boolean }): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>('/claw/tools', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * 删除 Claw 工具
   */
  async deleteClawTool(name: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/claw/tools/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
  }

  /**
   * 测试执行 Claw 工具代码
   */
  async executeClawTool(code: string, args?: Record<string, unknown>): Promise<{ success: boolean; output: string; error: string; result: string }> {
    return this.request<{ success: boolean; output: string; error: string; result: string }>('/claw/tools/execute', {
      method: 'POST',
      body: JSON.stringify({ code, args: args || {} }),
    });
  }

  /** 从已保存对话中删除单条对话 */
  async deleteDialogueFromSaved(id: number, index: number): Promise<{ success: boolean; message: string; remaining_count: number }> {
    return this.request<{ success: boolean; message: string; remaining_count: number }>(`/training/saved-dialogues/${id}/dialogues/${index}`, {
      method: 'DELETE',
    });
  }

  /** 从已保存对话创建数据集 */
  async createDatasetFromSaved(id: number, datasetName?: string): Promise<CreateDatasetFromSavedResponse> {
    const params = new URLSearchParams();
    if (datasetName) params.set('dataset_name', datasetName);
    return this.request<CreateDatasetFromSavedResponse>(`/training/saved-dialogues/${id}/create-dataset?${params.toString()}`, {
      method: 'POST',
    });
  }

  /** 删除样本 */
  async deleteIntentSample(label: string, index: number): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(`/knowledge/train-intent/samples?label=${encodeURIComponent(label)}&index=${index}`, {
      method: 'DELETE',
    });
  }

  /** 获取训练样本 */
  async getIntentSamples(): Promise<IntentSamplesResponse> {
    return this.request<IntentSamplesResponse>('/knowledge/train-intent/samples');
  }

  /** 保存全部样本（批量覆盖） */
  async saveIntentSamples(samples: Record<string, string[]>): Promise<{ success: boolean; stats: Record<string, number> }> {
    return this.request<{ success: boolean; stats: Record<string, number> }>('/knowledge/train-intent/samples', {
      method: 'PATCH',
      body: JSON.stringify({ samples }),
    });
  }

  /** 编辑样本 */
  async updateIntentSample(label: string, index: number, text: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('/knowledge/train-intent/samples', {
      method: 'PUT',
      body: JSON.stringify({ label, index, text }),
    });
  }

  /** 添加样本 */
  async addIntentSample(label: string, text: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('/knowledge/train-intent/samples', {
      method: 'POST',
      body: JSON.stringify({ label, text }),
    });
  }

  /** 生成训练样本（LLM基于知识库文档） */
  async generateIntentSamples(params: { kb_ids: number[]; samples_per_kb?: number; negative_count?: number; lora_name?: string }): Promise<{ success: boolean; message?: string; error?: string }> {
    return this.request<{ success: boolean; message?: string; error?: string }>('/knowledge/train-intent/generate', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  /** 查询样本生成进度 */
  async getGenerationStatus(): Promise<{ success: boolean; status: IntentTaskStatus }> {
    return this.request<{ success: boolean; status: IntentTaskStatus }>('/knowledge/train-intent/generate/status');
  }

  /** 启动意图分类器训练 */
  async trainIntentClassifier(params?: { kb_ids?: number[] }): Promise<{ success: boolean; message?: string; error?: string }> {
    return this.request<{ success: boolean; message?: string; error?: string }>('/knowledge/train-intent', {
      method: 'POST',
      body: JSON.stringify(params || {}),
    });
  }

  /** 查询训练状态 */
  async getIntentTrainingStatus(): Promise<{ success: boolean; status: IntentTrainingStatus }> {
    return this.request<{ success: boolean; status: IntentTrainingStatus }>('/knowledge/train-intent/status');
  }

  /** 取消训练/生成 */
  async cancelIntentTraining(): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>('/knowledge/train-intent/cancel', {
      method: 'POST',
    });
  }

  /** 获取意图分类模型信息 */
  async getIntentModelInfo(): Promise<{ success: boolean; model: IntentModelInfo }> {
    return this.request<{ success: boolean; model: IntentModelInfo }>('/knowledge/train-intent/model');
  }

  /** 获取活跃知识库 */
  async getActiveKnowledgeBases(): Promise<ActiveKnowledgeBasesResponse> {
    return this.request<ActiveKnowledgeBasesResponse>('/knowledge/train-intent/active-kbs');
  }

  // ============================================
  // 研究与评估 API（LLM Research Enhancement）
  // ============================================

  /** Evaluation and research APIs */
  async getGoldSet(): Promise<GoldSetResponse> {
    return this.request<GoldSetResponse>('/evaluation/gold-set');
  }

  async runEvaluation(req: EvaluationRunRequest): Promise<EvaluationRunStartResponse> {
    return this.request<EvaluationRunStartResponse>('/evaluation/run', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  async getEvaluationRuns(): Promise<EvaluationRunsResponse> {
    return this.request<EvaluationRunsResponse>('/evaluation/runs');
  }

  async getFeedback(rating?: string): Promise<FeedbackListResponse> {
    const query = rating ? `?rating=${encodeURIComponent(rating)}` : '';
    return this.request<FeedbackListResponse>(`/feedback${query}`);
  }

  async getExperiments(): Promise<ExperimentsResponse> {
    return this.request<ExperimentsResponse>('/experiments/');
  }

  async startExperiment(type: string, req: ExperimentStartRequest): Promise<ExperimentStartResponse> {
    return this.request<ExperimentStartResponse>(`/experiments/${type}`, {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  async getExperimentReport(expId: string): Promise<{ success: boolean; report: string }> {
    return this.request<{ success: boolean; report: string }>(`/experiments/${expId}/report`);
  }

  async getRouterConfig(): Promise<RouterConfigResponse> {
    return this.request<RouterConfigResponse>('/router/config');
  }

  async updateRouterConfig(req: Partial<RouterConfigRecord>): Promise<RouterConfigResponse> {
    return this.request<RouterConfigResponse>('/router/config', {
      method: 'PUT',
      body: JSON.stringify(req),
    });
  }

  async getRouterAdapters(): Promise<RouterAdaptersResponse> {
    return this.request<RouterAdaptersResponse>('/router/adapters');
  }

  async getRouterLogs(): Promise<RouterLogsResponse> {
    return this.request<RouterLogsResponse>('/router/logs');
  }

  async checkAdapter(adapterName: string): Promise<AdapterCheckResponse> {
    return this.request<AdapterCheckResponse>(`/router/check/${encodeURIComponent(adapterName)}`, {
      method: 'POST',
    });
  }

  async getPreferences(reviewStatus?: PreferenceReviewStatus): Promise<{ success: boolean; preferences: PreferencePairRecord[]; total: number }> {
    const query = reviewStatus ? `?review_status=${encodeURIComponent(reviewStatus)}` : '';
    return this.request<{ success: boolean; preferences: PreferencePairRecord[]; total: number }>(`/preferences/${query}`);
  }

  async createPreference(req: PreferenceCreateRequest): Promise<{ success: boolean; id: string }> {
    return this.request<{ success: boolean; id: string }>('/preferences/', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  async updatePreference(id: string, req: PreferenceUpdateRequest): Promise<{ success: boolean; id: string; note?: string }> {
    return this.request<{ success: boolean; id: string; note?: string }>(`/preferences/${id}`, {
      method: 'PUT',
      body: JSON.stringify(req),
    });
  }

  async exportPreferences(req: PreferenceExportRequest): Promise<{ success: boolean; format: string; count: number; data: PreferenceExportRecord[] }> {
    return this.request<{ success: boolean; format: string; count: number; data: PreferenceExportRecord[] }>('/preferences/export', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  async sampleFromHistory(req: PreferenceSampleRequest): Promise<{ success: boolean; candidates: PreferenceCandidate[]; total: number }> {
    return this.request<{ success: boolean; candidates: PreferenceCandidate[]; total: number }>('/preferences/sample-from-history', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  // ═══ 角色与长期记忆管理（admin） ═══

  async listCharacters(): Promise<{ success: boolean; characters: Array<{ character_id: string; display_name: string; version: string }> }> {
    return this.request('/characters');
  }

  async getCharacterRelationship(
    characterId: string,
    scope: CharacterMemoryScope,
  ): Promise<{ success: boolean; relationship: CharacterRelationshipRecord | null }> {
    return this.request(`/characters/${encodeURIComponent(characterId)}/relationship?${buildScopeQuery(scope)}`);
  }

  async updateCharacterRelationship(
    characterId: string,
    scope: CharacterMemoryScope,
    data: { stage: string; preferred_address?: string; summary?: string },
  ): Promise<{ success: boolean; relationship: CharacterRelationshipRecord }> {
    return this.request(`/characters/${encodeURIComponent(characterId)}/relationship?${buildScopeQuery(scope)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async listCharacterMemories(
    characterId: string,
    scope: CharacterMemoryScope,
  ): Promise<{ success: boolean; memories: CharacterMemoryRecord[] }> {
    return this.request(`/characters/${encodeURIComponent(characterId)}/memories?${buildScopeQuery(scope)}`);
  }

  async updateCharacterMemory(
    characterId: string,
    memoryId: number,
    scope: CharacterMemoryScope,
    data: { content: string; importance: number },
  ): Promise<{ success: boolean }> {
    return this.request(`/characters/${encodeURIComponent(characterId)}/memories/${memoryId}?${buildScopeQuery(scope)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteCharacterMemory(
    characterId: string,
    memoryId: number,
    scope: CharacterMemoryScope,
  ): Promise<{ success: boolean; message: string }> {
    return this.request(`/characters/${encodeURIComponent(characterId)}/memories/${memoryId}?${buildScopeQuery(scope)}`, {
      method: 'DELETE',
    });
  }

  async clearCharacterMemories(
    characterId: string,
    scope: CharacterMemoryScope,
  ): Promise<{ success: boolean; deleted: number; message: string }> {
    return this.request(`/characters/${encodeURIComponent(characterId)}/memories?${buildScopeQuery(scope)}`, {
      method: 'DELETE',
    });
  }

}

// 导出单例实例
export const api = new ApiClient();

export default api;
