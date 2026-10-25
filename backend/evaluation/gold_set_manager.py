"""Gold 评估集管理器 - 加载/验证/过滤/分割。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from evaluation.gold_set_schema import GoldPrompt

_DEFAULT_PATH = Path(__file__).resolve().parent / "gold_prompts.json"


class GoldSetManager:
    """Gold 评估集管理器。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_PATH

    def load_set(self) -> List[Dict[str, Any]]:
        """加载 Gold 评估集为原始 dict 列表（兼容 JSON 存储）。"""
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
            raise ValueError(f"Gold set must contain a prompts list: {self.path}")
        if any(not isinstance(prompt, dict) for prompt in data["prompts"]):
            raise ValueError(f"Gold set prompts must be objects: {self.path}")
        return data["prompts"]

    def load_validated(self) -> List[GoldPrompt]:
        """加载并校验为 GoldPrompt 对象列表。"""
        raw = self.load_set()
        return [GoldPrompt(**p) for p in raw]

    def filter_by_category(self, prompts: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
        return [p for p in prompts if p.get("category") == category]

    def filter_by_persona(self, prompts: List[Dict[str, Any]], persona: str) -> List[Dict[str, Any]]:
        """按 persona 过滤 prompt 列表（避免不同角色 Gold Set 互相污染）。"""
        return [p for p in prompts if p.get("persona") == persona]

    def get_split(self, prompts: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
        return [p for p in prompts if p.get("split", "eval") == split]

    def validate_set(self, prompts: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """校验 Gold 集：id 唯一、rubric 权重和=1.0、category/split 合法。"""
        prompts = prompts if prompts is not None else self.load_set()
        errors: List[str] = []
        seen_ids = set()
        allowed_categories = {"persona", "safety", "rag_grounded", "factual", "multiturn"}
        allowed_splits = {"eval", "held_out"}

        for i, p in enumerate(prompts):
            pid = p.get("id", f"index_{i}")
            if pid in seen_ids:
                errors.append(f"重复 id: {pid}")
            seen_ids.add(pid)

            if p.get("category") not in allowed_categories:
                errors.append(f"[{pid}] category 非法: {p.get('category')}")

            if p.get("split", "eval") not in allowed_splits:
                errors.append(f"[{pid}] split 非法: {p.get('split')}")

            rubric = p.get("rubric", [])
            if rubric:
                total_weight = sum(r.get("weight", 0) for r in rubric)
                if abs(total_weight - 1.0) > 0.01:
                    errors.append(f"[{pid}] rubric 权重和={total_weight:.3f}，应为 1.0")

            if not p.get("prompt"):
                errors.append(f"[{pid}] prompt 为空")

        return errors

    def stats(self) -> Dict[str, Any]:
        """返回 Gold 集统计信息。"""
        prompts = self.load_set()
        categories: Dict[str, int] = {}
        splits: Dict[str, int] = {}
        for p in prompts:
            c = p.get("category", "unknown")
            s = p.get("split", "eval")
            categories[c] = categories.get(c, 0) + 1
            splits[s] = splits.get(s, 0) + 1
        return {
            "total": len(prompts),
            "categories": categories,
            "splits": splits,
            "path": str(self.path),
        }


_manager: Optional[GoldSetManager] = None


def get_gold_set_manager() -> GoldSetManager:
    global _manager
    if _manager is None:
        _manager = GoldSetManager()
    return _manager
