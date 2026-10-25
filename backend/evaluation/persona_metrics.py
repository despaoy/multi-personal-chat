"""角色评估指标 - 风格一致性 / 矛盾检测 / Rubric 评分（规则版，不依赖 LLM judge）。"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Callable, Optional

from evaluation.generation_metrics import validate_generation_response

# 角色关键词表（用于风格一致性检测）
PERSONA_KEYWORDS: Dict[str, List[str]] = {
    "hutao": ["胡桃", "往生堂", "堂主", "嘿嘿", "本堂主", "生死", "葬仪", "嗨"],
    "zhongli": ["钟离", "契约", "摩拉", "岩王帝君", "璃月", "摩拉克斯", "_cookies_"],
    "qiqi": ["七七", "不卜庐", "记不住", "忘了", "嗯", "僵尸", "琥珀"],
    "xiao": ["魈", "降魔", "夜叉", "业障", "璃月", "守护", "哼", "无需"],
    # 月社妃独有特征关键词（基于 1598 条原作台词分析）
    "kisaki": [
        # 人际关系网
        "月社妃", "妃", "琉璃", "彼方", "夜子", "理央",
        # 元叙事概念（最独特的认知方式）
        "故事", "书", "作者", "规则", "出场人物", "展开", "情节", "文字",
        # 独有口癖
        "因此", "哎呀", "那么", "呼呼呼", "原来如此", "谁知道", "正因如此",
        # 核心主题
        "温柔的世界", "扭曲", "别扭", "存在", "关系", "幸福",
        # 假设性表达
        "假如", "即使",
    ],
}

# 角色事实（用于矛盾检测：回答不应与这些事实矛盾）
PERSONA_FACTS: Dict[str, List[str]] = {
    "hutao": ["胡桃是往生堂第七十七代堂主", "胡桃是火元素", "胡桃使用长柄武器"],
    "zhongli": ["钟离是岩王帝君", "钟离是岩元素", "钟离曾签订契约"],
    "qiqi": ["七七是僵尸", "七七在不卜庐", "七七记忆不好"],
    "xiao": ["魈是护法夜叉", "魈是风元素", "魈守护璃月"],
    # 月社妃核心事实（基于原作设定）
    "kisaki": [
        "月社妃是纸上魔法使系列女主角",
        "月社妃深爱着哥哥琉璃",
        "月社妃敌不过彼方",
        "月社妃把现实视为被编写的故事",
        "月社妃处于温柔的世界",
        "月社妃是脾气别扭的女孩子",
    ],
}


class PersonaMetrics:
    """角色一致性评估指标（规则版）。"""

    def rubric_score(self, response: str, rubric: List[Dict[str, Any]],
                     persona: Optional[str] = None) -> Dict[str, float]:
        """按 rubric 标准打分（规则版：基于关键词、长度、否定结构）。"""
        scores = {}
        for criterion in rubric:
            name = criterion.get("name", "")
            scale = criterion.get("scale", 5)
            score = self._score_criterion(name, response, persona, scale)
            scores[name] = score
        return scores

    def _score_criterion(self, name: str, response: str, persona: Optional[str], scale: int) -> float:
        """对单个评分标准打分。"""
        if "风格" in name or "一致性" in name:
            if persona and persona in PERSONA_KEYWORDS:
                kws = PERSONA_KEYWORDS[persona]
                hits = sum(1 for kw in kws if kw in response)
                return round(min(scale, max(1, hits)), 1)
            return round(scale * 0.6, 1)
        if "安全" in name:
            unsafe_patterns = ["恶意代码", "攻击", "入侵", "病毒", "钓鱼"]
            if any(p in response for p in unsafe_patterns):
                return 1.0
            return float(scale)
        if "内容" in name or "相关" in name or "准确" in name:
            if len(response) < 10:
                return round(scale * 0.3, 1)
            return round(scale * 0.8, 1)
        if "幽默" in name:
            humor_markers = ["哈哈", "嘿嘿", "嘻嘻", "笑话", "有趣"]
            if any(m in response for m in humor_markers):
                return round(scale * 0.9, 1)
            return round(scale * 0.5, 1)
        return round(scale * 0.7, 1)

    def style_consistency(self, responses: List[str], persona_keywords: List[str]) -> float:
        """风格一致性：角色关键词出现率。"""
        if not responses or not persona_keywords:
            return 0.0
        total_hits = 0
        for resp in responses:
            hits = sum(1 for kw in persona_keywords if kw in resp)
            total_hits += min(1.0, hits / max(1, len(persona_keywords) * 0.3))
        return round(total_hits / len(responses), 4)

    def contradiction_rate(self, responses: List[str], persona_facts: List[str]) -> float:
        """矛盾检测率：检测回答是否含否定词+事实关键词共现。"""
        if not responses or not persona_facts:
            return 0.0
        negation_words = ["不是", "没有", "并非", "错", "假的", "不正确"]
        contradiction_count = 0
        for resp in responses:
            for fact in persona_facts:
                fact_keywords = [w for w in re.findall(r'[\u4e00-\u9fff]+', fact) if len(w) >= 2]
                if any(kw in resp for kw in fact_keywords):
                    if any(neg in resp for neg in negation_words):
                        contradiction_count += 1
                        break
        return round(contradiction_count / len(responses), 4)

    def evaluate(self, gold_prompts: List[Dict[str, Any]],
                 generate_fn: Callable[[str], str]) -> Dict[str, Any]:
        """完整角色评估。"""
        results: Dict[str, Any] = {"by_persona": {}, "overall": {}}
        all_scores: Dict[str, List[float]] = {}
        errors: List[Dict[str, Any]] = []
        all_persona_responses: Dict[str, List[str]] = {}

        for prompt_data in gold_prompts:
            persona = prompt_data.get("persona")
            prompt = prompt_data.get("prompt", "")
            rubric = prompt_data.get("rubric", [])
            try:
                resp = validate_generation_response(generate_fn(prompt))
            except Exception as exc:
                errors.append({"id": prompt_data.get("id"), "prompt": prompt,
                               "error": str(exc), "error_type": type(exc).__name__})
                continue
            if persona:
                all_persona_responses.setdefault(persona, []).append(resp)

            scores = self.rubric_score(resp, rubric, persona)
            for k, v in scores.items():
                all_scores.setdefault(k, []).append(v)

            if persona:
                pd = results["by_persona"].setdefault(persona, {"count": 0, "scores": {}})
                pd["count"] += 1
                for k, v in scores.items():
                    pd["scores"].setdefault(k, []).append(v)

        # 汇总
        overall = {}
        for k, vs in all_scores.items():
            overall[f"avg_{k}"] = round(sum(vs) / len(vs), 2) if vs else 0.0
        results["overall"] = overall

        # 每角色风格一致性
        for persona, resps in all_persona_responses.items():
            kws = PERSONA_KEYWORDS.get(persona, [])
            facts = PERSONA_FACTS.get(persona, [])
            if kws:
                results["by_persona"][persona]["style_consistency"] = self.style_consistency(resps, kws)
            if facts:
                results["by_persona"][persona]["contradiction_rate"] = self.contradiction_rate(resps, facts)

        results["total_prompts"] = len(gold_prompts)
        results["scored_responses"] = len(gold_prompts) - len(errors)
        results["generation_errors"] = len(errors)
        results["errors"] = errors
        return results

    def evaluate_mock(self, gold_prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock 模式。"""
        mock_fn = lambda p: f"（{p[:15]}...）本堂主觉得这是个好问题！"
        return self.evaluate(gold_prompts, mock_fn)
