"""
Pydantic 请求/响应模型
从 main.py 提取，集中管理所有 API 数据模型

注意：本文件仅包含 Pydantic 模型（API 契约）。
SQLAlchemy ORM 模型（数据库 schema）定义在 db/models.py。
"""

from pydantic import BaseModel, Field, StringConstraints, field_validator
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
    history: list[dict[str, str]] = Field(default_factory=list, max_length=40)

    @field_validator("history")
    @classmethod
    def _validate_history(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        total_chars = 0
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("history items must be objects")
            if item.get("role") not in {"user", "assistant"}:
                raise ValueError("history role must be 'user' or 'assistant'")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("history content must be a non-empty string")
            if len(content) > 4000:
                raise ValueError("history content exceeds 4000 characters")
            total_chars += len(content)
        if total_chars > 24000:
            raise ValueError("history total content exceeds 24000 characters")
        return value


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
    model_name: str = Field(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
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

    @field_validator("password")
    @classmethod
    def validate_bcrypt_password_length(cls, password: str) -> str:
        if len(password.encode("utf-8")) > 72:
            raise ValueError("password must not exceed 72 UTF-8 bytes")
        return password


class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=100)

    @field_validator("password")
    @classmethod
    def validate_bcrypt_password_length(cls, password: str) -> str:
        if len(password.encode("utf-8")) > 72:
            raise ValueError("password must not exceed 72 UTF-8 bytes")
        return password


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
    dataset_id: Literal["kisaki_v21", "kisaki_v3", "legacy_general"] = "kisaki_v21"
    adapter_name: Optional[str] = Field(default=None, max_length=256)
    model_label: Optional[str] = Field(default=None, max_length=256)
    categories: Optional[List[str]] = Field(default=None, max_length=20)
    split: Literal["eval", "held_out"] = "eval"
    max_prompts: Optional[int] = Field(default=None, ge=1, le=50)
    mock: bool = False


class ExperimentStartRequest(BaseModel):
    """实验启动请求"""
    hypothesis: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    mock: bool = False


PreferenceReviewStatus = Literal["pending", "approved", "rejected"]


class PreferencePairCreate(BaseModel):
    """偏好对创建请求"""
    prompt: str = Field(..., min_length=1, max_length=20_000)
    chosen: str = Field(..., min_length=1, max_length=20_000)
    rejected: str = Field(..., min_length=1, max_length=20_000)
    rubric: Optional[Dict[str, float]] = Field(default=None, max_length=20)
    annotator: str = Field(default="manual", min_length=1, max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(default=None, max_length=100)
    review_status: PreferenceReviewStatus = "pending"


class PreferencePairUpdate(BaseModel):
    """偏好对更新请求"""
    review_status: Optional[PreferenceReviewStatus] = None
    rubric: Optional[Dict[str, float]] = Field(default=None, max_length=20)
    annotator: Optional[str] = Field(default=None, min_length=1, max_length=100)


class PreferenceExportRequest(BaseModel):
    """偏好数据导出请求"""
    review_status: PreferenceReviewStatus = "approved"
    format: Literal["jsonl", "json"] = "jsonl"
    limit: int = Field(default=10_000, ge=1, le=10_000)


class SampleFromHistoryRequest(BaseModel):
    """从消息历史采样偏好对"""
    limit: int = Field(default=20, ge=1, le=200)
    session_id: Optional[str] = Field(default=None, max_length=256)
    min_length: int = Field(default=10, ge=1, le=8_000)


class RouterConfigUpdate(BaseModel):
    """路由配置更新请求"""
    enabled: Optional[bool] = None
    default_adapter: Optional[str] = Field(default=None, min_length=1, max_length=128)
    mode: Optional[Literal["manual", "rule", "intent"]] = None
    persona_adapters: Optional[Dict[str, str]] = Field(default=None, max_length=20)
    rag_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    persona_keywords: Optional[Dict[str, List[str]]] = Field(default=None, max_length=20)

    @field_validator("persona_adapters")
    @classmethod
    def validate_persona_adapters(cls, value: Optional[Dict[str, str]]):
        if value is None:
            return value
        if any(not key or len(key) > 64 or not adapter or len(adapter) > 128 for key, adapter in value.items()):
            raise ValueError("persona adapter names exceed allowed bounds")
        return value

    @field_validator("persona_keywords")
    @classmethod
    def validate_persona_keywords(cls, value: Optional[Dict[str, List[str]]]):
        if value is None:
            return value
        for persona, keywords in value.items():
            if not persona or len(persona) > 64 or len(keywords) > 100:
                raise ValueError("persona keyword groups exceed allowed bounds")
            if any(not keyword or len(keyword) > 100 for keyword in keywords):
                raise ValueError("persona keywords exceed allowed bounds")
        return value


class IntentSampleGenerateRequest(BaseModel):
    """Bounded request for generating intent-classifier samples."""
    kb_ids: List[int] = Field(default_factory=list, max_length=8)
    samples_per_kb: int = Field(default=100, ge=1, le=500)
    negative_count: int = Field(default=200, ge=0, le=1000)
    lora_name: Optional[str] = Field(default=None, max_length=256)


class IntentTrainRequest(BaseModel):
    """Request for training the lightweight intent classifier."""
    kb_ids: Optional[List[int]] = Field(default=None, max_length=8)


class ApiKeyCreateRequest(BaseModel):
    """Create one managed API key with bounded metadata."""
    role: Literal["admin", "operator", "viewer", "api_user"] = "api_user"
    description: Optional[str] = Field(default=None, max_length=500)
    rate_limit: Optional[int] = Field(default=None, ge=1, le=10000)


class FeedbackCreate(BaseModel):
    """用户反馈创建请求"""
    trace_id: Optional[str] = Field(default=None, max_length=128)
    message_id: Optional[str] = Field(default=None, max_length=256)
    rating: Literal["positive", "negative", "thumbs_up", "thumbs_down"]
    reason: Optional[str] = Field(default=None, max_length=2000)
    adapter_name: Optional[str] = Field(default=None, max_length=256)
    kb_revision: Optional[str] = Field(default=None, max_length=256)
    prompt_version: Optional[str] = Field(default=None, max_length=256)
    detail: Optional[str] = Field(default=None, max_length=10000)


class RetrievalEvalQuestionCreate(BaseModel):
    """检索评估问题创建请求"""
    id: Optional[str] = Field(default=None, max_length=128)
    question: str = Field(..., min_length=1, max_length=2000)
    expected_doc_ids: List[Annotated[str, StringConstraints(max_length=256)]] = Field(
        default_factory=list,
        max_length=50,
    )
    expected_doc_titles: List[Annotated[str, StringConstraints(max_length=500)]] = Field(
        default_factory=list,
        max_length=50,
    )
    gold_answer: Optional[str] = Field(default=None, max_length=10000)
    category: Optional[str] = Field(default=None, max_length=128)
