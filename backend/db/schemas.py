"""
Pydantic 请求/响应模型
从 main.py 提取，集中管理所有 API 数据模型

注意：本文件仅包含 Pydantic 模型（API 契约）。
SQLAlchemy ORM 模型（数据库 schema）定义在 db/models.py。
"""

from pydantic import BaseModel, Field, StringConstraints
from typing import Optional, List, Dict, Any, Literal, Annotated

# ============================================
# 核心消息模型
# ============================================


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    sessionType: str = "private"
    conversationType: str = ""
    sessionId: str = ""
    sessionName: str = ""
    userId: str = ""
    userName: str = ""
    senderName: str = ""
    loraName: str = ""
    platform: str = "qq"
    adapter: str = "nonebot"
    conversationId: str = ""
    senderId: str = ""
    sourceMessageId: str = ""
    traceId: str = ""


class GenerateResponse(BaseModel):
    reply: str
    model: str = "Qwen/Qwen3-8B"
    costTime: float
    citations: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    abstained: bool = False


class StatsResponse(BaseModel):
    todayMessages: int = 0
    todayReplies: int = 0
    avgResponseTime: float = 0.0
    p95ResponseTime: float = 0.0
    p99ResponseTime: float = 0.0
    modelFailureRate: float = 0.0
    ragFailureRate: float = 0.0
    activeSessions: int = 0
    modelLoad: int = 0
    cpuUsage: int = 0
    gpuMemory: Dict[str, float] = {}
    memoryUsage: Dict[str, float] = {}
    diskUsage: Dict[str, float] = {}
    queueLength: int = 0
    currentInferenceConcurrency: int = 0
    astrBotGateway: Dict[str, Any] = {}
    platformStatus: Dict[str, Any] = {}


# ============================================
# LoRA 训练管理模型
# ============================================


class DatasetUploadRequest(BaseModel):
    """数据集上传请求"""
    dataset_name: str
    style: Optional[str] = None
    custom_prompt: Optional[str] = None
    data: List[Dict[str, Any]]


class TrainingStartRequest(BaseModel):
    """训练启动请求"""
    lora_name: str
    dataset_name: str
    model_type: str = "qwen3-8b"
    custom_config: Optional[Dict[str, Any]] = None


class ModelDownloadRequest(BaseModel):
    """模型下载请求"""
    model_name: str
    force: bool = False


# ============================================
# 知识库管理模型
# ============================================


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求"""
    name: Optional[str] = None
    description: Optional[str] = None


class KnowledgeFolderCreate(BaseModel):
    """创建知识库文件夹请求"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""


class KnowledgeDocumentCreate(BaseModel):
    """创建知识库文档请求"""
    title: str
    content: str
    category: str = "未分类"
    knowledge_base_id: Optional[int] = None
    folder_id: Optional[int] = None
    sourceType: str = "text"
    sourceUrl: Optional[str] = None
    fileType: Optional[str] = None
    fileSize: Optional[int] = None


class KnowledgeDocumentUpdate(BaseModel):
    """更新知识库文档请求"""
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    knowledge_base_id: Optional[int] = None
    folder_id: Optional[int] = None
    sourceType: Optional[str] = None
    sourceUrl: Optional[str] = None
    fileType: Optional[str] = None
    fileSize: Optional[int] = None


class KnowledgeSearchRequest(BaseModel):
    """知识库搜索请求"""
    # 去除首尾空白（strip_whitespace），最小长度 1（拒绝空查询/纯空白），
    # 最大长度 2000 防止滥用。Pydantic v2 用 StringConstraints 替代 Field 的
    # strip_whitespace 参数（后者已在 v2 废弃）。
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000, strip_whitespace=True)]
    topK: int = Field(default=5, ge=1, le=100)
    knowledgeBaseName: Optional[str] = None  # 按知识库名称过滤检索结果


class KnowledgeSearchResult(BaseModel):
    """知识库搜索结果"""
    documentId: int
    documentTitle: str
    # 多数检索分支（rag_pipeline/keyword/empty）不返回 chunkId，仅部分向量路径可能携带
    chunkId: Optional[int] = None
    chunkIndex: int
    content: str
    score: float


# ============================================
# 用户认证模型
# ============================================


class RegisterRequest(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)


class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str
    password: str


class UserDataRequest(BaseModel):
    """用户数据保存请求"""
    page_key: str = Field(..., min_length=1, max_length=100)
    data_json: str = Field(..., min_length=1)


# ============================================
# 对话数据保存模型
# ============================================


class SaveDialoguesRequest(BaseModel):
    """保存对话数据请求"""
    name: str = Field(..., min_length=1, max_length=200)
    character_desc: str = Field(..., min_length=1)
    style: Optional[str] = None
    dialogues: list
    turn_stats: Optional[Dict[str, int]] = None
    scene_stats: Optional[Dict[str, int]] = None


class SavedDialoguesListItem(BaseModel):
    """已保存对话列表项（不含完整数据）"""
    id: int
    name: str
    character_desc: str
    style: Optional[str] = None
    dialogue_count: int
    created_at: str
    updated_at: str


# ============================================
# 对话生成模型
# ============================================


class DialogueGenerateRequest(BaseModel):
    """对话生成请求"""
    character_description: str
    num_dialogues: int = 10
    style: Optional[str] = None
    custom_prompt: Optional[str] = None


# ============================================
# 研究与评估模型（LLM Research Enhancement Roadmap）
# ============================================


class GoldPromptSchema(BaseModel):
    """Gold 评估集单条提示词"""
    id: str
    prompt: str
    expected_behavior: str = ""
    category: str = "persona"
    tags: List[str] = Field(default_factory=list)
    persona: Optional[str] = None
    expected_refs: Optional[List[str]] = None
    split: str = "eval"


class EvalRunRequest(BaseModel):
    """评估运行请求"""
    adapter_name: Optional[str] = None
    model_label: Optional[str] = None
    categories: Optional[List[str]] = None
    split: str = "eval"
    max_prompts: Optional[int] = None
    mock: bool = False


class ExperimentStartRequest(BaseModel):
    """实验启动请求"""
    hypothesis: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    mock: bool = False


class PreferencePairCreate(BaseModel):
    """偏好对创建请求"""
    prompt: str
    chosen: str
    rejected: str
    rubric: Optional[Dict[str, float]] = None
    annotator: str = "manual"
    metadata: Optional[Dict[str, Any]] = None
    review_status: str = "pending"


class PreferencePairUpdate(BaseModel):
    """偏好对更新请求"""
    review_status: Optional[str] = None
    rubric: Optional[Dict[str, float]] = None
    annotator: Optional[str] = None


class PreferenceExportRequest(BaseModel):
    """偏好数据导出请求"""
    review_status: str = "approved"
    format: str = "jsonl"


class SampleFromHistoryRequest(BaseModel):
    """从消息历史采样偏好对"""
    limit: int = 20
    session_id: Optional[str] = None
    min_length: int = 10


class RouterConfigUpdate(BaseModel):
    """路由配置更新请求"""
    enabled: Optional[bool] = None
    default_adapter: Optional[str] = None
    mode: Optional[Literal["manual", "rule", "intent"]] = None
    persona_adapters: Optional[Dict[str, str]] = None
    rag_confidence_threshold: Optional[float] = None
    persona_keywords: Optional[Dict[str, List[str]]] = None


class FeedbackCreate(BaseModel):
    """用户反馈创建请求"""
    trace_id: Optional[str] = None
    message_id: Optional[str] = None
    rating: str  # "thumbs_up" | "thumbs_down"
    reason: Optional[str] = None
    adapter_name: Optional[str] = None
    kb_revision: Optional[str] = None
    prompt_version: Optional[str] = None
    detail: Optional[str] = None


class RetrievalEvalQuestionCreate(BaseModel):
    """检索评估问题创建请求"""
    id: Optional[str] = None
    question: str
    expected_doc_ids: List[str] = Field(default_factory=list)
    expected_doc_titles: List[str] = Field(default_factory=list)
    gold_answer: Optional[str] = None
    category: Optional[str] = None
