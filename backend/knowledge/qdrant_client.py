"""Qdrant 向量数据库客户端（阶段2.6 预留模块）

当前使用 Faiss 进行向量检索，Qdrant 作为后续升级选项。
Qdrant 优势：
- 内置 CRUD API（无需手动管理索引）
- 支持过滤搜索（按知识库ID、分类等）
- 支持分布式部署
- 持久化存储，无需重建索引

迁移路径：
1. docker run qdrant/qdrant
2. pip install qdrant-client
3. 重写 knowledge/vector_db.py 使用 Qdrant
4. 批量迁移现有向量到 Qdrant

H2 fix: 此前文件只有 docstring，无任何代码实现。若被误导入使用，
不会报错但也无法工作。现添加占位类，导入时正常，实例化时抛出
NotImplementedError，明确告知调用方此模块尚未实现。
"""


class QdrantClient:
    """Qdrant 向量数据库客户端占位类。

    此模块为预留实现，当前尚未完成。实例化时会抛出 NotImplementedError，
    避免被误用为可工作的客户端。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "QdrantClient 尚未实现。当前向量检索使用 Faiss（见 knowledge/vector_db.py）。"
            "如需启用 Qdrant，请先完成迁移路径中描述的实现步骤。"
        )

    def add_documents(self, *args, **kwargs):
        raise NotImplementedError("QdrantClient 尚未实现")

    def search(self, *args, **kwargs):
        raise NotImplementedError("QdrantClient 尚未实现")

    def delete_collection(self, *args, **kwargs):
        raise NotImplementedError("QdrantClient 尚未实现")
