"""偏好对齐数据模式 - 现用偏好对(pairwise)数据结构与 JSONL 导出。

遵循路线图 guardrail：
- 偏好数据存为不可变记录，含标注者 rubric 和分歧元数据
- 支持 JSONL 格式用于 DPO/ORPO 训练
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, Field


class PreferencePair(BaseModel):
    """单条偏好对 (prompt, chosen, rejected)。"""
    id: str = Field(default_factory=lambda: f"pref_{uuid.uuid4().hex[:12]}")
    prompt: str
    chosen: str
    rejected: str
    rubric: Dict[str, float] = Field(default_factory=dict)
    annotator: str = "manual"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    review_status: str = "pending"  # pending | approved | rejected
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """转换为 JSONL 格式（适配 trl DPO/ORPO）。"""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "rubric": self.rubric,
            "annotator": self.annotator,
            "metadata": self.metadata,
            "review_status": self.review_status,
            "created_at": self.created_at,
        }


def save_jsonl(pairs: List[PreferencePair], path: Path) -> Path:
    """将偏好对列表保存为 JSONL 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_jsonl_dict(), ensure_ascii=False) + "\n")
    return path
