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
from tempfile import TemporaryDirectory
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


class IntentTaskCancelled(RuntimeError):
    """Cooperative cancellation, distinct from a dependency or generation error."""


def _check_cancelled():
    if _cancel_flag.is_set():
        raise IntentTaskCancelled("任务已被用户取消")


# ═══════════════════════════════════════════════════════════
# 知识库配置
# ═══════════════════════════════════════════════════════════


def get_active_knowledge_bases() -> List[Dict[str, Any]]:
    config_path = MODEL_DIR / "active_kbs.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def set_active_knowledge_bases(kb_ids: List[int], *, database) -> Dict[str, Any]:
    db = database
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
    """Prepare the complete payload before atomically publishing the sample file."""
    stats = {label: len(items) for label, items in samples.items()}
    payload = json.dumps(samples, ensure_ascii=False, indent=2)
    path = _get_samples_path()
    with TemporaryDirectory(prefix=".samples-", dir=path.parent) as folder:
        candidate = Path(folder) / path.name
        candidate.touch(mode=0o600)
        candidate.write_text(payload, encoding="utf-8")
        os.replace(candidate, path)
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


class SampleGenerationError(RuntimeError):
    """A safe, actionable failure of the sample-generation contract."""


async def generate_samples(
    kb_ids: List[int],
    samples_per_kb: int = 100,
    negative_count: int = 200,
    lora_name: Optional[str] = None,
    *,
    database,
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
    _generation_status = {"running": True, "progress": 0, "stage": "init", "message": "初始化..."}
    try:
        _generation_status.update(stage="init", message="加载知识库信息...", progress=3)
        _check_cancelled()
        db = database
        all_bases = db.get_knowledge_bases()
        selected_kbs = [b for b in all_bases if b["id"] in kb_ids] if kb_ids else all_bases[:MAX_KB_COUNT]
        if not selected_kbs:
            raise SampleGenerationError("没有可用的知识库")
        if kb_ids and len(selected_kbs) != len(set(kb_ids)):
            raise SampleGenerationError("部分选中的知识库不存在，请刷新后重新选择")
        if len(selected_kbs) > MAX_KB_COUNT:
            raise SampleGenerationError(f"最多选择 {MAX_KB_COUNT} 个知识库")
        _generation_status.update(stage="load_docs", message="读取知识库文档...", progress=8)
        _add_log("读取知识库文档内容...")
        kb_docs = _load_kb_documents(selected_kbs, database=database)
        for kb_name, doc_text in kb_docs.items():
            _add_log(f"知识库「{kb_name}」: 文档摘要 {len(doc_text)} 字符")
        _check_cancelled()
        _generation_status.update(stage="connect", message="连接vLLM服务...", progress=10)
        vllm_client = await _get_vllm_client()
        _add_log("vLLM服务连接成功")
        return await _generate_samples_inner(
            vllm_client, selected_kbs, kb_docs, samples_per_kb, negative_count, lora_name
        )
    except asyncio.CancelledError:
        _generation_status.update(running=False, stage="cancelled", message="已取消", progress=0)
        raise
    except IntentTaskCancelled:
        _generation_status.update(running=False, stage="cancelled", message="已取消", progress=0)
        return {"error": "已取消"}
    except Exception as exc:
        stage = _generation_status["stage"]
        reason = str(exc) if isinstance(exc, SampleGenerationError) else type(exc).__name__
        message = f"样本生成失败（{stage}：{reason}），请检查知识库、vLLM服务和请求数量后重试"
        _generation_status.update(running=False, stage="error", message=message, progress=0)
        logger.error("样本生成失败，阶段=%s，类型=%s", stage, type(exc).__name__)
        raise SampleGenerationError(message) from exc


async def _generate_samples_inner(
    vllm_client,
    selected_kbs,
    kb_docs,
    samples_per_kb: int,
    negative_count: int,
    lora_name: Optional[str],
) -> Dict[str, Any]:
    """完成全部生成与数量检查后才保存；共享客户端由应用生命周期关闭。"""
    # ── 4. 为每个知识库生成正例（基于文档内容） ──
    kb_samples: Dict[str, List[str]] = {}
    total_kbs = len(selected_kbs)
    for i, kb in enumerate(selected_kbs):
        _check_cancelled()
        progress = 12 + int(40 * i / total_kbs)
        _generation_status.update(
            stage="generate_kb",
            message=f"为「{kb['name']}」生成正例样本 ({i + 1}/{total_kbs})...",
            progress=progress,
        )
        _add_log(f"生成知识库「{kb['name']}」的正例样本...")

        doc_context = kb_docs[kb["name"]]
        samples = await _generate_kb_samples_with_docs(vllm_client, kb["name"], doc_context, samples_per_kb, lora_name)
        kb_samples[kb["name"]] = samples
        _add_log(f"知识库「{kb['name']}」: 生成 {len(samples)} 条正例")

    _check_cancelled()
    _generation_status.update(stage="generate_neg", message="生成通用负例样本...", progress=60)
    general_count = (negative_count + 1) // 2
    negative_samples = await _generate_negative_samples(vllm_client, general_count, lora_name)

    _check_cancelled()
    _generation_status.update(stage="generate_hard_neg", message="生成跨KB硬负例...", progress=70)
    hard_count = (negative_count - general_count + total_kbs - 1) // total_kbs
    if hard_count:
        hard_negatives = await _generate_hard_negatives_per_kb(
            vllm_client, selected_kbs, kb_docs, hard_count, lora_name
        )
        for samples in hard_negatives.values():
            negative_samples.extend(samples)

    _check_cancelled()
    kb_samples["none"] = _take_samples(negative_samples, negative_count)
    _generation_status.update(stage="save", message="保存样本...", progress=90)
    save_samples(kb_samples)

    total = sum(len(v) for v in kb_samples.values())
    _add_log(f"样本生成完成，共 {total} 条（含硬负例）")

    _generation_status.update(
        running=False,
        stage="done",
        message="样本生成完成",
        progress=100,
    )
    return {"success": True, "total": total, "stats": {k: len(v) for k, v in kb_samples.items()}}


def _load_kb_documents(kbs: list, *, database) -> Dict[str, str]:
    """Load bounded source text; missing evidence cannot become a guessed topic."""
    db = database
    result = {}
    for kb in kbs:
        docs = db.get_knowledge_documents(knowledge_base_id=kb["id"], limit=30)
        parts = []
        for doc in docs:
            content = doc.get("content", "").strip()
            if content:
                title = doc.get("title", "")
                if title:
                    parts.append(f"【{title}】")
                parts.append(content[:200])
        text = "\n".join(parts)[:3000]
        if not text:
            raise SampleGenerationError(f"知识库 {kb['id']} 没有可用文档内容，请先补全文档")
        result[kb["name"]] = text
    return result


async def _get_vllm_client():
    """Use the application's shared vLLM client without substituting a provider."""
    from app.config import is_vllm_enabled
    from inference.vllm_client import get_vllm_client

    if not is_vllm_enabled():
        raise SampleGenerationError("样本生成需要启用并配置 vLLM 服务")
    return await get_vllm_client()


async def _generate_with_vllm(client, prompt: str, lora_name: Optional[str] = None) -> str:
    _check_cancelled()
    messages = [
        {
            "role": "system",
            "content": "你是一个数据生成助手。严格按照用户要求生成数据，每条一行，不要添加额外说明或编号。",
        },
        {"role": "user", "content": prompt},
    ]
    reply = await client.generate(
        messages=messages,
        lora_name=lora_name,
        temperature=0.9,
        max_tokens=1024,
    )
    return reply


def _parse_samples(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        raise SampleGenerationError("模型未返回非空文本样本")
    samples = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        cleaned = line
        if len(line) > 2 and line[0].isdigit() and line[1] in ".、:：":
            cleaned = line[2:].strip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in ".、:：":
            cleaned = line[3:].strip()
        if 2 <= len(cleaned) <= 100:
            samples.append(cleaned)
    return samples


def _take_samples(samples: list[str], count: int) -> list[str]:
    unique = list(dict.fromkeys(samples))
    if len(unique) < count:
        raise SampleGenerationError(f"有效去重样本不足：需要 {count} 条，实际 {len(unique)} 条")
    return unique[:count]


async def _generate_sample_batches(client, prompts: list[str], count: int, lora_name: Optional[str]) -> list[str]:
    samples = []
    for prompt in prompts:
        reply = await _generate_with_vllm(client, prompt, lora_name)
        samples.extend(_parse_samples(reply))
    return _take_samples(samples, count)


async def _generate_kb_samples_with_docs(
    client, kb_name: str, doc_context: str, count: int, lora_name: Optional[str] = None
) -> list:
    """基于知识库文档内容生成查询样本"""
    batch_size = (count + 2) // 3

    if not doc_context.strip():
        raise SampleGenerationError("生成正例需要知识库文档内容")
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

    return await _generate_sample_batches(client, prompts, count, lora_name)


async def _generate_negative_samples(client, count: int, lora_name: Optional[str] = None) -> list:
    if count == 0:
        return []
    batch_size = (count + 3) // 4
    prompts = [
        f"生成{batch_size}条日常闲聊消息，每条一行。包括问候、感叹、情感表达等，不涉及任何知识查询意图。用自然的中文口语，不要编号。",
        f"生成{batch_size}条游戏玩家的日常聊天消息，每条一行。包括分享游戏状态、表达喜好、社交互动等，不涉及具体知识查询。用自然语气，不要编号。",
        f"生成{batch_size}条QQ群里的日常闲聊消息，每条一行。包括表情、短句、日常话题等，不涉及知识查询。用口语化表达，不要编号。",
        f"生成{batch_size}条非知识查询的社交消息，每条一行。包括道谢、道歉、确认、告别等，不涉及任何信息查询。用日常口语，不要编号。",
    ]

    return await _generate_sample_batches(client, prompts, count, lora_name)


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

    for kb in selected_kbs:
        _check_cancelled()
        kb_name = kb["name"]
        kb_doc = kb_docs[kb_name]

        # 其他KB的名称和领域摘要
        other_kbs = [k for k in selected_kbs if k["name"] != kb_name]
        if not other_kbs:
            # 只有一个KB时，生成"看起来像知识查询但实际是闲聊"的伪硬负例
            other_desc = "日常闲聊、社交对话"
        else:
            other_parts = []
            for okb in other_kbs:
                other_doc = kb_docs[okb["name"]][:200]
                other_parts.append(f"「{okb['name']}」: {other_doc}")
            other_desc = "\n".join(other_parts)

        # 构造提示：生成跨域混淆问题
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

        result[kb_name] = await _generate_sample_batches(client, [prompt], count_per_kb, lora_name)

    return result


# ═══════════════════════════════════════════════════════════
# 训练（使用已审查的样本）
# ═══════════════════════════════════════════════════════════


async def train_intent_classifier(
    kb_ids: List[int] = None,
    samples_per_kb: int = 100,
    negative_count: int = 200,
    lora_name: Optional[str] = None,
    *,
    database,
) -> Dict[str, Any]:
    """使用已保存的样本训练多分类模型"""
    global _training_status, _training_logs
    if _training_status["running"]:
        return {"error": "训练正在进行中"}
    _cancel_flag.clear()
    _training_logs = []
    _training_status = {
        "running": True,
        "progress": 0,
        "stage": "init",
        "message": "加载样本数据...",
        "result": None,
        "started_at": time.time(),
    }
    result = None
    try:
        _training_status.update(stage="load_samples", message="加载样本数据...", progress=5)
        _add_log("加载已保存的样本数据")
        _check_cancelled()
        samples_data = get_samples()
        all_samples = samples_data["samples"]
        if not all_samples:
            raise RuntimeError("没有可用的样本数据，请先生成样本")
        if kb_ids:
            db = database
            all_bases = db.get_knowledge_bases()
            kb_names = [b["name"] for b in all_bases if b["id"] in kb_ids]
            filtered = {}
            for name, items in all_samples.items():
                if name in kb_names or name == "none":
                    filtered[name] = items
            all_samples = filtered
        if kb_ids:
            set_active_knowledge_bases(kb_ids, database=database)
        kb_names = [k for k in all_samples.keys() if k != "none"]
        negative_samples = all_samples.get("none", [])
        kb_samples = {k: v for k, v in all_samples.items() if k != "none"}
        total = sum((len(v) for v in all_samples.values()))
        _add_log(f"样本统计: 总计 {total} 条, 知识库 {kb_names}, 负例 {len(negative_samples)} 条")
        if not kb_samples:
            raise RuntimeError("没有知识库样本，请先生成样本")
        _training_status.update(stage="train", message="编码文本 + 训练多分类模型...", progress=15)
        _add_log("开始编码和训练...")
        result = await asyncio.to_thread(_train_multiclass_model, kb_samples, negative_samples)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "训练失败"))
        _check_cancelled()
        _training_status.update(stage="reload", message="重载意图检测器...", progress=95)
        await _reload_intent_detector()
        result = {**result, "artifacts_saved": True, "reloaded": True}
        _training_status.update(running=False, stage="done", message="训练完成！", progress=100, result=result)
        _add_log(f"训练完成！准确率={result.get('accuracy', 'N/A')}")
        return result
    except asyncio.CancelledError:
        cancelled = {"cancelled": True}
        if result is not None and result.get("success"):
            cancelled.update(result, success=False, artifacts_saved=True, reloaded=False, cancelled=True)
        _training_status.update(running=False, stage="cancelled", message="训练已取消", progress=0, result=cancelled)
        raise
    except Exception as exc:
        failure = {"error": str(exc)}
        message = str(exc)
        if result is not None and result.get("success"):
            failure = {**result, **failure, "success": False, "artifacts_saved": True, "reloaded": False}
            message = "训练产物已保存，但重载失败：" + message
        cancelled = isinstance(exc, RuntimeError) and "取消" in str(exc)
        if cancelled:
            failure["cancelled"] = True
        _training_status.update(
            running=False,
            stage="cancelled" if cancelled else "error",
            message="训练已取消" if cancelled else message,
            progress=0,
            result=failure,
        )
        _add_log(message)
        return failure


# ═══════════════════════════════════════════════════════════
# 多分类模型训练
# ═══════════════════════════════════════════════════════════


def _train_multiclass_model(
    kb_samples: Dict[str, List[str]],
    negative_samples: List[str],
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

    embed_model, embedding_source = _load_embed_model()

    _add_log(f"编码 {len(texts)} 条文本...")
    X = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
    y = np.array(labels)

    _check_cancelled()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _add_log("交叉验证中...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(set(labels))), scoring="accuracy", error_score="raise")
    cv_acc, cv_std = float(scores.mean()), float(scores.std())
    _add_log(f"交叉验证准确率: {cv_acc:.4f} (+/- {cv_std:.4f})")

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
        "embedding_model": embedding_source,
        "label_names": label_names,
        "label_map": label_map,
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

    source = (os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("EMBED_MODEL_PATH", "")).strip()
    if not source:
        raise ValueError("Intent training requires EMBEDDING_MODEL_PATH")
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = BACKEND_DIR / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Intent embedding directory does not exist: {path}")
    return SentenceTransformer(str(path)), str(path)


async def _reload_intent_detector():
    from knowledge import intent_detector

    intent_detector._set_ml_model(None)
    artifacts = await asyncio.to_thread(intent_detector._read_ml_model, MODEL_DIR)
    _check_cancelled()
    intent_detector._set_ml_model(artifacts)
    _add_log("意图检测器已重载")


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
