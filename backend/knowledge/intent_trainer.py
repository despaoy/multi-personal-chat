"""
RAG意图分类器训练模块 - 多分类 + LLM生成样本 + 样本审查 + 轻量分类器训练

核心能力：
- 多分类模型：预测查询应路由到哪个知识库（或不需要RAG）
- LLM生成训练样本：先读取知识库文档内容，再让LLM基于文档生成样本
- 样本审查：生成后可查看、编辑、删除样本，人工修正后再训练
- 任务取消：通过 _cancel_flag 支持中断
- 知识库关联：训练时选择参与的知识库，模型自动学习路由
"""

import asyncio
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent.parent
def _runtime_dir_from_env(name: str, default: Path) -> Path:
    configured = os.getenv(name, "").strip()
    path = Path(configured).expanduser() if configured else default
    return path if path.is_absolute() else BACKEND_DIR / path


MODEL_DIR = _runtime_dir_from_env("INTENT_MODEL_PATH", BACKEND_DIR / "intent_classifier_model")
SAMPLES_DIR = _runtime_dir_from_env("INTENT_SAMPLES_PATH", BACKEND_DIR / "intent_samples")

# 最大可选知识库数量（性能与准确度平衡）
MAX_KB_COUNT = 8

# 训练状态
_training_status: Dict[str, Any] = {
    "running": False,
    "progress": 0,
    "stage": "",
    "message": "",
    "result": None,
    "started_at": None,
    "logs": [],
}

# 样本生成状态
_generation_status: Dict[str, Any] = {
    "running": False,
    "progress": 0,
    "stage": "",
    "message": "",
}

# 取消标志
_cancel_flag = threading.Event()

# 训练日志
_training_logs: List[str] = []


def get_training_status() -> Dict[str, Any]:
    status = _training_status.copy()
    status["logs"] = _training_logs[-50:]
    return status


def get_generation_status() -> Dict[str, Any]:
    return _generation_status.copy()


def cancel_training() -> bool:
    if not _training_status["running"] and not _generation_status["running"]:
        return False
    _cancel_flag.set()
    _add_log("用户请求取消...")
    return True


def _add_log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    _training_logs.append(f"[{timestamp}] {msg}")
    if len(_training_logs) > 200:
        _training_logs.pop(0)
    logger.info(f"[IntentTrainer] {msg}")


def _check_cancelled():
    if _cancel_flag.is_set():
        raise RuntimeError("任务已被用户取消")


# ═══════════════════════════════════════════════════════════
# 知识库配置
# ═══════════════════════════════════════════════════════════

def get_active_knowledge_bases() -> List[Dict[str, Any]]:
    config_path = MODEL_DIR / "active_kbs.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def set_active_knowledge_bases(kb_ids: List[int]) -> Dict[str, Any]:
    from db.adapter import db
    all_bases = db.get_knowledge_bases()
    selected = [b for b in all_bases if b["id"] in kb_ids]
    if len(selected) > MAX_KB_COUNT:
        return {"error": f"最多选择 {MAX_KB_COUNT} 个知识库"}
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "active_kbs.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    return {"success": True, "active_kbs": selected}


# ═══════════════════════════════════════════════════════════
# 样本管理（生成 / 查看 / 编辑 / 删除）
# ═══════════════════════════════════════════════════════════

def _get_samples_path() -> Path:
    SAMPLES_DIR.mkdir(exist_ok=True)
    return SAMPLES_DIR / "samples.json"


def get_samples() -> Dict[str, Any]:
    """获取当前所有样本"""
    path = _get_samples_path()
    if not path.exists():
        return {"samples": {}, "stats": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 计算统计
    stats = {}
    for label, items in data.items():
        stats[label] = len(items)
    return {"samples": data, "stats": stats}


def save_samples(samples: Dict[str, List[str]]) -> Dict[str, Any]:
    """保存样本（编辑后）"""
    path = _get_samples_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    stats = {label: len(items) for label, items in samples.items()}
    return {"success": True, "stats": stats}


def update_sample(label: str, index: int, new_text: str) -> Dict[str, Any]:
    """编辑单条样本"""
    data = get_samples()["samples"]
    if label not in data or index < 0 or index >= len(data[label]):
        return {"error": "样本不存在"}
    data[label][index] = new_text
    save_samples(data)
    return {"success": True}


def delete_sample(label: str, index: int) -> Dict[str, Any]:
    """删除单条样本"""
    data = get_samples()["samples"]
    if label not in data or index < 0 or index >= len(data[label]):
        return {"error": "样本不存在"}
    data[label].pop(index)
    save_samples(data)
    return {"success": True}


def add_sample(label: str, text: str) -> Dict[str, Any]:
    """添加单条样本"""
    data = get_samples()["samples"]
    if label not in data:
        data[label] = []
    data[label].append(text)
    save_samples(data)
    return {"success": True}


# ═══════════════════════════════════════════════════════════
# 样本生成（LLM + 知识库文档）
# ═══════════════════════════════════════════════════════════

async def generate_samples(
    kb_ids: List[int],
    samples_per_kb: int = 100,
    negative_count: int = 200,
    lora_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    异步生成训练样本（不训练，仅生成供审查）
    LLM会先接收知识库文档内容，再基于文档生成样本

    生成策略：
    - 正例：每个KB基于自身文档生成具体问题
    - 通用负例：日常闲聊、社交消息（不需要RAG）
    - 硬负例：每个KB生成"看起来像但实际属于其他KB"的跨域混淆问题
    """
    global _generation_status

    if _generation_status["running"]:
        return {"error": "样本生成正在进行中"}

    _cancel_flag.clear()
    _generation_status = {
        "running": True,
        "progress": 0,
        "stage": "init",
        "message": "初始化...",
    }

    try:
        # ── 1. 获取知识库信息 ──
        _generation_status.update(stage="init", message="加载知识库信息...", progress=3)
        _check_cancelled()

        from db.adapter import db
        all_bases = db.get_knowledge_bases()
        selected_kbs = [b for b in all_bases if b["id"] in kb_ids] if kb_ids else all_bases[:MAX_KB_COUNT]

        if not selected_kbs:
            raise RuntimeError("没有可用的知识库")

        if len(selected_kbs) > MAX_KB_COUNT:
            raise RuntimeError(f"最多选择 {MAX_KB_COUNT} 个知识库")

        # ── 2. 读取知识库文档内容 ──
        _generation_status.update(stage="load_docs", message="读取知识库文档...", progress=8)
        _add_log("读取知识库文档内容...")
        kb_docs = _load_kb_documents(selected_kbs)
        for kb_name, doc_text in kb_docs.items():
            _add_log(f"知识库「{kb_name}」: 文档摘要 {len(doc_text)} 字符")

        # ── 3. 获取vLLM客户端 ──
        _check_cancelled()
        _generation_status.update(stage="connect", message="连接vLLM服务...", progress=10)
        vllm_client = _get_vllm_client()
        if not vllm_client:
            raise RuntimeError("vLLM服务不可用")
        _add_log("vLLM服务连接成功")

        try:
            # C11 fix: 用嵌套 try/finally 确保 vllm_client 在所有退出路径上被关闭，
            # 避免每次样本生成泄漏一个 httpx.AsyncClient 连接池
            return await _generate_samples_inner(
                vllm_client, selected_kbs, kb_docs,
                samples_per_kb, negative_count, lora_name,
            )
        finally:
            try:
                await vllm_client.close()
            except Exception as close_err:
                logger.warning(f"关闭 vLLM 客户端失败: {close_err}")

    except RuntimeError as e:
        if "取消" in str(e):
            _generation_status.update(running=False, stage="cancelled", message="已取消", progress=0)
        else:
            _generation_status.update(running=False, stage="error", message=str(e), progress=0)
        return {"error": str(e)}
    except Exception as e:
        _generation_status.update(running=False, stage="error", message=str(e), progress=0)
        return {"error": str(e)}


async def _generate_samples_inner(
    vllm_client, selected_kbs, kb_docs,
    samples_per_kb: int, negative_count: int, lora_name: Optional[str],
) -> Dict[str, Any]:
    """样本生成主逻辑（由 generate_samples 调用，client 由外层负责关闭）"""
    # ── 4. 为每个知识库生成正例（基于文档内容） ──
    kb_samples: Dict[str, List[str]] = {}
    total_kbs = len(selected_kbs)
    for i, kb in enumerate(selected_kbs):
        _check_cancelled()
        progress = 12 + int(40 * i / total_kbs)
        _generation_status.update(
            stage="generate_kb",
            message=f"为「{kb['name']}」生成正例样本 ({i+1}/{total_kbs})...",
            progress=progress,
        )
        _add_log(f"生成知识库「{kb['name']}」的正例样本...")

        doc_context = kb_docs.get(kb["name"], "")
        samples = await _generate_kb_samples_with_docs(
            vllm_client, kb["name"], doc_context, samples_per_kb, lora_name
        )
        kb_samples[kb["name"]] = samples
        _add_log(f"知识库「{kb['name']}」: 生成 {len(samples)} 条正例")

    # ── 5. 生成通用负例（日常闲聊） ──
    _check_cancelled()
    _generation_status.update(stage="generate_neg", message="生成通用负例样本...", progress=60)
    _add_log("生成通用负例样本...")
    general_negative_count = max(negative_count // 2, 50)
    negative_samples = await _generate_negative_samples(vllm_client, general_negative_count, lora_name)
    _add_log(f"通用负例生成完成: {len(negative_samples)} 条")

    # ── 6. 为每个KB生成硬负例（跨域混淆问题） ──
    _check_cancelled()
    _generation_status.update(stage="generate_hard_neg", message="生成跨KB硬负例...", progress=70)
    _add_log("生成跨KB硬负例样本...")
    hard_neg_per_kb = max(negative_count // (total_kbs * 2), 10)
    hard_negatives = await _generate_hard_negatives_per_kb(
        vllm_client, selected_kbs, kb_docs, hard_neg_per_kb, lora_name
    )
    for kb_name, hard_negs in hard_negatives.items():
        # 硬负例归入对应KB的负例池（标签为"none"，但内容是与该KB混淆的跨域问题）
        negative_samples.extend(hard_negs)
        _add_log(f"知识库「{kb_name}」: 生成 {len(hard_negs)} 条硬负例")

    # ── 7. 保存样本 ──
    kb_samples["none"] = list(set(negative_samples))
    save_samples(kb_samples)

    total = sum(len(v) for v in kb_samples.values())
    _add_log(f"样本生成完成，共 {total} 条（含硬负例）")

    _generation_status.update(
        running=False, stage="done", message="样本生成完成", progress=100,
    )
    return {"success": True, "total": total, "stats": {k: len(v) for k, v in kb_samples.items()}}


def _load_kb_documents(kbs: list) -> Dict[str, str]:
    """读取知识库文档内容，拼接为上下文"""
    from db.adapter import db
    result = {}
    for kb in kbs:
        try:
            docs = db.get_knowledge_documents(knowledge_base_id=kb["id"], limit=200)
            parts = []
            for doc in docs[:30]:  # 最多30篇文档
                title = doc.get("title", "")
                content = doc.get("content", "")
                if title:
                    parts.append(f"【{title}】")
                if content:
                    # 截取前200字，避免过长
                    parts.append(content[:200])
            result[kb["name"]] = "\n".join(parts)[:3000]  # 最多3000字符
        except Exception as e:
            logger.warning(f"读取知识库「{kb['name']}」文档失败: {e}")
            result[kb["name"]] = ""
    return result


def _get_vllm_client():
    """通过 inference 层获取 vLLM 客户端，不依赖 api 层"""
    vllm_enabled = (
        os.getenv("VLLM_ENABLED", "").lower() == "true"
        or bool(os.getenv("VLLM_BASE_URLS", "").strip())
        or bool(os.getenv("VLLM_BASE_URL", "").strip())
    )
    if not vllm_enabled:
        logger.warning("vLLM 未启用（缺少 VLLM_ENABLED 或 VLLM_BASE_URLS 环境变量）")
        return None
    try:
        from inference.vllm_client import VLLMClient
        client = VLLMClient()
        logger.info("vLLM 客户端初始化成功（通过 inference 层）")
        return client
    except Exception as e:
        logger.warning(f"vLLM 客户端初始化失败: {e}")
        return None


async def _generate_with_vllm(client, prompt: str, lora_name: Optional[str] = None) -> str:
    _check_cancelled()
    messages = [
        {"role": "system", "content": "你是一个数据生成助手。严格按照用户要求生成数据，每条一行，不要添加额外说明或编号。"},
        {"role": "user", "content": prompt},
    ]
    reply = await client.generate(
        messages=messages, lora_name=lora_name, temperature=0.9, max_tokens=1024,
    )
    return reply


def _parse_samples(text: str) -> list:
    samples = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        cleaned = line
        if len(line) > 2 and line[0].isdigit() and line[1] in '.、:：':
            cleaned = line[2:].strip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in '.、:：':
            cleaned = line[3:].strip()
        if 2 <= len(cleaned) <= 100:
            samples.append(cleaned)
    return samples


async def _generate_kb_samples_with_docs(
    client, kb_name: str, doc_context: str, count: int, lora_name: Optional[str] = None
) -> list:
    """基于知识库文档内容生成查询样本"""
    batch_size = max(count // 3, 10)

    if doc_context:
        # 有文档内容：基于文档生成具体问题
        prompts = [
            (
                f"以下是知识库的部分文档内容：\n\n{doc_context}\n\n"
                f"请根据以上文档内容，生成{batch_size}条用户可能会问的具体问题。"
                f"问题必须直接涉及文档中提到的角色、技能、数值、机制等具体内容，"
                f"例如「胡桃的二命效果是什么」「雷电将军的元素爆发倍率多少」这样的具体问题。"
                f"不要生成关于知识库本身的元问题（如「是否包含」「有没有」等）。"
                f"每条一行，使用自然的中文口语，不要编号。"
            ),
            (
                f"以下是知识库的部分文档内容：\n\n{doc_context}\n\n"
                f"请根据以上文档内容，生成{batch_size}条玩家常问的具体问题。"
                f"包括具体的数据、效果、获取方式、对比等。"
                f"问题必须引用文档中的具体内容，例如「钟离的护盾吸收量怎么算」「甘雨的蓄力箭倍率」等。"
                f"不要生成泛泛的问题。每条一行，用日常对话语气，不要编号。"
            ),
            (
                f"以下是知识库的部分文档内容：\n\n{doc_context}\n\n"
                f"请根据以上文档内容，生成{batch_size}条进阶问题。"
                f"包括机制细节、策略搭配、数值对比等深度内容。"
                f"问题必须具体，例如「行秋的雨帘剑减伤比例是多少」「温迪大招聚怪范围多大」等。"
                f"每条一行，用自然语气，不要编号。"
            ),
        ]
    else:
        # 无文档内容：用知识库名称推断领域
        prompts = [
            f"假设有一个关于「{kb_name}」的知识库，生成{batch_size}条用户可能会问的具体问题。"
            f"问题应该涉及具体的角色、技能、数值、机制等内容。"
            f"每条一行，使用自然的中文口语，不要编号。",
            f"生成{batch_size}条关于「{kb_name}」领域的常见具体问题。"
            f"包括数据查询、效果说明、获取方式等。每条一行，用日常对话语气，不要编号。",
            f"生成{batch_size}条关于「{kb_name}」领域的进阶问题。"
            f"包括机制、策略、对比等深度内容。每条一行，用自然语气，不要编号。",
        ]

    all_samples = []
    for i, prompt in enumerate(prompts):
        try:
            reply = await _generate_with_vllm(client, prompt, lora_name)
            samples = _parse_samples(reply)
            all_samples.extend(samples)
        except Exception as e:
            logger.warning(f"知识库「{kb_name}」样本生成批次 {i+1} 失败: {e}")

    return list(set(all_samples))[:count]


async def _generate_negative_samples(
    client, count: int, lora_name: Optional[str] = None
) -> list:
    batch_size = max(count // 4, 10)
    prompts = [
        f"生成{batch_size}条日常闲聊消息，每条一行。包括问候、感叹、情感表达等，不涉及任何知识查询意图。用自然的中文口语，不要编号。",
        f"生成{batch_size}条游戏玩家的日常聊天消息，每条一行。包括分享游戏状态、表达喜好、社交互动等，不涉及具体知识查询。用自然语气，不要编号。",
        f"生成{batch_size}条QQ群里的日常闲聊消息，每条一行。包括表情、短句、日常话题等，不涉及知识查询。用口语化表达，不要编号。",
        f"生成{batch_size}条非知识查询的社交消息，每条一行。包括道谢、道歉、确认、告别等，不涉及任何信息查询。用日常口语，不要编号。",
    ]

    all_samples = []
    for i, prompt in enumerate(prompts):
        try:
            reply = await _generate_with_vllm(client, prompt, lora_name)
            samples = _parse_samples(reply)
            all_samples.extend(samples)
        except Exception as e:
            logger.warning(f"负例生成批次 {i+1} 失败: {e}")

    template_neg = [
        "你好", "晚安", "好的", "谢谢", "哈哈", "嗯嗯",
        "好无聊", "好累", "吃饭了吗", "在干嘛",
        "太棒了", "我刚上线", "随便聊聊", "来玩吧",
        "好开心", "真的吗", "知道了", "666",
    ]
    all_samples.extend(template_neg)
    return list(set(all_samples))[:count]


async def _generate_hard_negatives_per_kb(
    client,
    selected_kbs: List[Dict[str, Any]],
    kb_docs: Dict[str, str],
    count_per_kb: int,
    lora_name: Optional[str] = None,
) -> Dict[str, List[str]]:
    """为每个KB生成硬负例：看起来像该KB但实际属于其他KB的跨域混淆问题

    策略：对每个KB-A，告知LLM「这是KB-A的领域，同时存在其他KB-B/C...」，
    让LLM生成「提到KB-A的内容但问题实际属于其他KB」的混淆问题。
    这些问题标签为"none"（不需要检索KB-A），用于强化多分类边界。

    Args:
        client: vLLM客户端
        selected_kbs: 所有选中的知识库列表
        kb_docs: KB名→文档内容
        count_per_kb: 每个KB生成的硬负例数量
        lora_name: 可选LoRA名称

    Returns:
        {kb_name: [硬负例问题列表]}
    """
    result: Dict[str, List[str]] = {}
    total_kbs = len(selected_kbs)

    for i, kb in enumerate(selected_kbs):
        _check_cancelled()
        kb_name = kb["name"]
        kb_doc = kb_docs.get(kb_name, "")

        # 其他KB的名称和领域摘要
        other_kbs = [k for k in selected_kbs if k["name"] != kb_name]
        if not other_kbs:
            # 只有一个KB时，生成"看起来像知识查询但实际是闲聊"的伪硬负例
            other_desc = "日常闲聊、社交对话"
        else:
            other_parts = []
            for okb in other_kbs:
                other_doc = kb_docs.get(okb["name"], "")[:200]
                other_parts.append(f"「{okb['name']}」: {other_doc}")
            other_desc = "\n".join(other_parts)

        # 构造提示：生成跨域混淆问题
        if kb_doc:
            prompt = (
                f"以下是知识库「{kb_name}」的文档内容：\n\n{kb_doc[:1500]}\n\n"
                f"同时存在其他知识库：\n{other_desc}\n\n"
                f"请生成{count_per_kb}条「看起来像属于「{kb_name}」但实际应该检索其他知识库」的混淆问题。\n"
                f"这些问题应该提到「{kb_name}」领域的内容（如角色名、术语），"
                f"但问题核心实际属于其他知识库的范畴。\n"
                f"例如：如果「{kb_name}」是角色库、其他是武器库，"
                f"生成「胡桃的护摩之杖属性怎么样」「钟离用什么圣遗物」这类跨域问题。\n"
                f"每条一行，用自然中文口语，不要编号。"
            )
        else:
            prompt = (
                f"假设有一个关于「{kb_name}」的知识库，"
                f"同时存在其他知识库：{', '.join(k['name'] for k in other_kbs)}。\n"
                f"请生成{count_per_kb}条「提到「{kb_name}」相关内容但实际属于其他知识库」的混淆问题。\n"
                f"每条一行，用自然中文口语，不要编号。"
            )

        try:
            reply = await _generate_with_vllm(client, prompt, lora_name)
            samples = _parse_samples(reply)
            result[kb_name] = samples
        except Exception as e:
            logger.warning(f"知识库「{kb_name}」硬负例生成失败: {e}")
            result[kb_name] = []

    return result


# ═══════════════════════════════════════════════════════════
# 训练（使用已审查的样本）
# ═══════════════════════════════════════════════════════════

async def train_intent_classifier(
    kb_ids: List[int] = None,
    samples_per_kb: int = 100,
    negative_count: int = 200,
    lora_name: Optional[str] = None,
) -> Dict[str, Any]:
    """使用已保存的样本训练多分类模型"""
    global _training_status, _training_logs

    if _training_status["running"]:
        return {"error": "训练正在进行中"}

    _cancel_flag.clear()
    _training_logs = []
    _training_status = {
        "running": True, "progress": 0, "stage": "init",
        "message": "加载样本数据...", "result": None, "started_at": time.time(),
    }

    try:
        # ── 1. 加载已保存的样本 ──
        _training_status.update(stage="load_samples", message="加载样本数据...", progress=5)
        _add_log("加载已保存的样本数据")
        _check_cancelled()

        samples_data = get_samples()
        all_samples = samples_data["samples"]

        if not all_samples:
            raise RuntimeError("没有可用的样本数据，请先生成样本")

        # 如果指定了kb_ids，只使用对应知识库的样本
        if kb_ids:
            from db.adapter import db
            all_bases = db.get_knowledge_bases()
            kb_names = [b["name"] for b in all_bases if b["id"] in kb_ids]
            filtered = {}
            for name, items in all_samples.items():
                if name in kb_names or name == "none":
                    filtered[name] = items
            all_samples = filtered

        # 保存活跃知识库配置
        if kb_ids:
            set_active_knowledge_bases(kb_ids)

        kb_names = [k for k in all_samples.keys() if k != "none"]
        negative_samples = all_samples.get("none", [])
        kb_samples = {k: v for k, v in all_samples.items() if k != "none"}

        # 构建 KB名称→ID 映射（供预测时RAG过滤使用）
        from db.adapter import db
        all_bases = db.get_knowledge_bases()
        kb_name_to_id = {b["name"]: b["id"] for b in all_bases if b["name"] in kb_names}

        total = sum(len(v) for v in all_samples.values())
        _add_log(f"样本统计: 总计 {total} 条, 知识库 {kb_names}, 负例 {len(negative_samples)} 条")

        if not kb_samples:
            raise RuntimeError("没有知识库样本，请先生成样本")

        # ── 2. 编码 + 训练 ──
        _training_status.update(stage="train", message="编码文本 + 训练多分类模型...", progress=15)
        _add_log("开始编码和训练...")
        result = await asyncio.to_thread(_train_multiclass_model, kb_samples, negative_samples, kb_name_to_id)

        if not result.get("success"):
            raise RuntimeError(result.get("error", "训练失败"))

        # ── 3. 重载意图检测器 ──
        _check_cancelled()
        _training_status.update(stage="reload", message="重载意图检测器...", progress=95)
        _reload_intent_detector()

        _training_status.update(
            running=False, stage="done", message="训练完成！", progress=100, result=result,
        )
        _add_log(f"训练完成！准确率={result.get('accuracy', 'N/A')}")
        return result

    except RuntimeError as e:
        if "取消" in str(e):
            _training_status.update(running=False, stage="cancelled", message="训练已取消", progress=0, result={"cancelled": True})
        else:
            _training_status.update(running=False, stage="error", message=str(e), progress=0, result={"error": str(e)})
        _add_log(str(e))
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        _training_status.update(running=False, stage="error", message=str(e), progress=0, result={"error": str(e)})
        _add_log(f"训练失败: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# 多分类模型训练
# ═══════════════════════════════════════════════════════════

def _train_multiclass_model(
    kb_samples: Dict[str, List[str]],
    negative_samples: List[str],
    kb_name_to_id: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("请安装 sentence-transformers")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import joblib

    _check_cancelled()

    # 标签：每个知识库一个类别 + "none"
    label_names = list(kb_samples.keys()) + ["none"]
    label_map = {name: i for i, name in enumerate(label_names)}

    texts = []
    labels = []
    for kb_name, samples in kb_samples.items():
        for s in samples:
            texts.append(s)
            labels.append(label_map[kb_name])
    for s in negative_samples:
        texts.append(s)
        labels.append(label_map["none"])

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined) if combined else ([], [])
    texts, labels = list(texts), list(labels)

    _add_log(f"训练数据: {len(texts)} 条, 类别: {label_names}")
    _add_log(f"各类别样本数: {dict(zip(label_names, [labels.count(i) for i in range(len(label_names))]))}")

    embed_model = _load_embed_model()
    if not embed_model:
        raise RuntimeError("无法加载嵌入模型")

    _add_log(f"编码 {len(texts)} 条文本...")
    X = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
    y = np.array(labels)

    _check_cancelled()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _add_log("交叉验证中...")
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="lbfgs",
    )
    try:
        scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(set(labels))), scoring="accuracy")
        cv_acc, cv_std = float(scores.mean()), float(scores.std())
        _add_log(f"交叉验证准确率: {cv_acc:.4f} (+/- {cv_std:.4f})")
    except Exception as e:
        _add_log(f"交叉验证失败: {e}")
        cv_acc, cv_std = 0.0, 0.0

    _check_cancelled()

    _add_log("训练最终模型...")
    clf.fit(X_scaled, y)
    train_acc = float(clf.score(X_scaled, y))
    _add_log(f"训练集准确率: {train_acc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    config = {
        "model_type": "multiclass_logistic_regression",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "label_names": label_names,
        "label_map": label_map,
        "kb_name_to_id": kb_name_to_id or {},
        "version": "4.1-multiclass-routed",
        "training_samples": len(texts),
        "samples_per_class": {name: labels.count(i) for name, i in label_map.items()},
        "cv_accuracy_mean": cv_acc,
        "cv_accuracy_std": cv_std,
        "train_accuracy": train_acc,
        "sample_source": "llm+vllm+docs+hard_negatives",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _add_log(f"模型已保存: {MODEL_DIR}")

    return {
        "success": True,
        "model_type": "multiclass",
        "training_samples": len(texts),
        "label_names": label_names,
        "samples_per_class": {name: labels.count(i) for name, i in label_map.items()},
        "accuracy": round(train_acc, 4),
        "cv_f1_mean": round(cv_acc, 4),
        "cv_f1_std": round(cv_std, 4),
        "model_path": str(MODEL_DIR),
    }


def _load_embed_model():
    from sentence_transformers import SentenceTransformer
    candidates = [
        BACKEND_DIR.parent / "RAG" / "paraphrase-multilingual-MiniLM-L12-v2",
        BACKEND_DIR / "models" / "paraphrase-multilingual-MiniLM-L12-v2",
        Path(os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("EMBED_MODEL_PATH", "")),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                model = SentenceTransformer(str(candidate))
                logger.info(f"使用本地嵌入模型: {candidate}")
                return model
            except Exception:
                continue
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("使用在线嵌入模型")
        return model
    except Exception as e:
        logger.error(f"无法加载嵌入模型: {e}")
        return None


def _reload_intent_detector():
    try:
        import knowledge.intent_detector as module
        module._ML_AVAILABLE = False
        module._ml_encoder = None
        module._ml_classifier = None
        module._ml_config = None
        from knowledge.intent_detector import needs_rag
        needs_rag("测试")
        _add_log("意图检测器已重载")
    except Exception as e:
        _add_log(f"意图检测器重载失败（下次启动时自动加载）: {e}")


def get_model_info() -> Dict[str, Any]:
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        return {"exists": False}
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {
        "exists": True,
        "model_type": config.get("model_type", "unknown"),
        "version": config.get("version", "unknown"),
        "label_names": config.get("label_names", []),
        "training_samples": config.get("training_samples", 0),
        "samples_per_class": config.get("samples_per_class", {}),
        "accuracy": config.get("train_accuracy", config.get("cv_accuracy_mean", 0)),
        "cv_f1_mean": config.get("cv_accuracy_mean", 0),
        "trained_at": config.get("trained_at", "unknown"),
    }
