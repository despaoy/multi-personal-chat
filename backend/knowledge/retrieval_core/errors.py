"""Safe failure details for required retrieval indexes."""


class RetrievalIndexError(RuntimeError):
    def __init__(self, channel: str, reason: str):
        self.channel, self.reason = channel, reason
        label = {"sparse": "BM25", "vector": "Vector", "documents": "Document"}[channel]
        problem = {
            "not_initialized": "not initialized",
            "row_count_mismatch": "inconsistent with documents",
            "invalid_row": "referencing an invalid document row",
        }[reason]
        super().__init__(f"{label} index is {problem}; load or rebuild the complete index")

    def public_detail(self) -> dict[str, str]:
        label = {"sparse": "关键词", "vector": "向量", "documents": "文档"}[self.channel]
        problem = {
            "not_initialized": "未初始化",
            "row_count_mismatch": "与文档数量不一致",
            "invalid_row": "引用了不存在的文档行",
        }[self.reason]
        return {
            "stage": "retrieval_index",
            "channel": self.channel,
            "reason": self.reason,
            "message": f"{label}索引{problem}，本轮未返回回复。请加载或重建完整索引后重试。",
        }
