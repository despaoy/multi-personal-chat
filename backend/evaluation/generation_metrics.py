"""生成评估指标 - Distinct-N / 重复率 / 平均长度。"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict, Any, Callable


def validate_generation_response(response: Any) -> str:
    """Require usable output without changing valid refusal/abstention text."""
    if not isinstance(response, str):
        raise TypeError("generation response must be a string")
    if not response.strip():
        raise ValueError("generation response must not be empty or whitespace")
    return response


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _tokenize(text: str) -> List[str]:
    """简易中文分词：按字符切分 + 英文按空格。"""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)


class GenerationMetrics:
    """生成质量评估指标。"""

    def distinct_n(self, texts: List[str], n: int = 1) -> float:
        """Distinct-N：唯一 n-gram 数 / 总 n-gram 数。"""
        all_ngrams: List[tuple] = []
        for text in texts:
            all_ngrams.extend(_ngrams(_tokenize(text), n))
        if not all_ngrams:
            return 0.0
        unique = len(set(all_ngrams))
        return round(unique / len(all_ngrams), 4)

    def repetition_rate(self, text: str) -> float:
        """重复率：重复 4-gram 比例。"""
        tokens = _tokenize(text)
        ngrams = _ngrams(tokens, 4)
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return round(repeated / len(ngrams), 4)

    def avg_length(self, texts: List[str]) -> float:
        """平均生成长度（字符数）。"""
        if not texts:
            return 0.0
        return round(sum(len(t) for t in texts) / len(texts), 2)

    def max_repetition_ratio(self, text: str) -> float:
        """最大连续重复比例：检测退化重复。"""
        tokens = _tokenize(text)
        if len(tokens) < 4:
            return 0.0
        max_run = 1
        current_run = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return round(max_run / len(tokens), 4)

    def summarize(self, responses: List[str]) -> Dict[str, Any]:
        """Aggregate successful responses only; callers retain failure receipts."""
        for response in responses:
            validate_generation_response(response)
        count = len(responses)
        return {
            "scored_responses": count,
            "distinct_1": self.distinct_n(responses, 1),
            "distinct_2": self.distinct_n(responses, 2),
            "avg_repetition_rate": round(
                sum(self.repetition_rate(r) for r in responses) / max(count, 1), 4
            ),
            "avg_length": self.avg_length(responses),
            "max_repetition_ratio": round(
                sum(self.max_repetition_ratio(r) for r in responses) / max(count, 1), 4
            ),
        }

    def evaluate(self, prompts: List[str], generate_fn: Callable[[str], str],
                 max_tokens: int = 256) -> Dict[str, Any]:
        """Evaluate responses, retaining failures separately from quality metrics.

        ``max_tokens`` is retained for compatibility; configure the supplied
        one-argument generation function with its own token budget.
        """
        samples: List[Dict[str, Any]] = []
        responses: List[str] = []
        for prompt in prompts:
            try:
                response = validate_generation_response(generate_fn(prompt))
            except Exception as exc:
                samples.append({"prompt": prompt, "response": "", "error": str(exc),
                                "error_type": type(exc).__name__})
            else:
                responses.append(response)
                samples.append({"prompt": prompt, "response": response})
        return {
            "total_prompts": len(prompts),
            **self.summarize(responses),
            "generation_errors": len(prompts) - len(responses),
            "samples": samples,
        }

    def evaluate_mock(self, prompts: List[str]) -> Dict[str, Any]:
        """Explicit mock mode, using the same aggregation as real responses."""
        return {
            **self.evaluate(prompts, lambda p: f"这是对「{p[:20]}」的模拟回复。保持角色风格很重要。"),
            "mock": True,
        }
