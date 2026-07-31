"""基础设施模块

提供：
- 熔断器 (circuit_breaker)
- 访问控制 (access_control)
- 备份管理 (backup_manager)
- 数据加密 (encryption)
- 故障转移 (failover)
- 输入验证 (input_validator)
- 资源池 (resource_pool)
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState, global_registry

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    "global_registry",
]
