"""
模块接口定义 - 依赖倒置原则

高层模块定义接口，低层模块实现接口。
模块间通过接口通信，而非直接依赖具体实现。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ============================================
# 数据库接口
# ============================================

@runtime_checkable
class DatabaseInterface(Protocol):
    """数据库访问接口 - SQLite 和 PostgreSQL 的统一抽象"""

    @property
    def config(self) -> Dict[str, Any]:
        """获取已解析持久化标量的配置字典；缓存和调用方直接使用这些值。"""
        ...

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """获取消息列表"""
        ...

    @property
    def loras(self) -> List[Dict[str, Any]]:
        """获取 LoRA 模型列表"""
        ...

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新系统配置"""
        ...

    def execute_sql(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行 SQL 查询"""
        ...

    def execute_sql_insert(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """执行 SQL 插入/更新"""
        ...

    def iter_chunks_with_document(self, batch_size: int = 500):
        """分页迭代 chunk 及其所属文档（LEFT JOIN），避免 N+1 查询。

        每条记录包含 chunk 字段和文档字段（doc_title / doc_category /
        doc_kb_id）。孤儿 chunk 的 doc_title 为 None。
        SQLiteDB / SyncPgAdapter 必须实现此方法以支持向量索引重建。
        """
        ...


# ============================================
# 熔断器接口
# ============================================

@runtime_checkable
class CircuitBreakerInterface(Protocol):
    """熔断器接口"""

    @property
    def state(self) -> str:
        """获取当前状态"""
        ...

    async def call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """通过熔断器调用函数"""
        ...

    async def reset(self) -> None:
        """重置熔断器"""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        ...
