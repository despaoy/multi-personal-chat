"""配置值类型解析公共工具（M-3 fix）。

消除 database.py、pg_database.py、model_manager.py 三处重复实现。
统一异常处理为 (ValueError, TypeError)，避免裸 except 吞掉 SystemExit/KeyboardInterrupt。
"""

import logging

logger = logging.getLogger(__name__)


def coerce_config_value(value):
    """将持久化的配置字符串转换为原始 Python 类型。

    解析规则：
    - "true"/"false"（不区分大小写）→ bool
    - 含 "." 的字符串 → float
    - 不含 "." 的字符串 → int
    - 转换失败 → 保留原字符串

    Args:
        value: 配置值（通常是字符串，也可能是已转换的类型）。

    Returns:
        转换后的值（bool/int/float/str），或原始值（如果非字符串类型）。
    """
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = str(value)
    lowered = text.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    try:
        if '.' in text:
            return float(text)
        return int(text)
    except (ValueError, TypeError):
        # P1-M1 fix: 不记录配置值内容，避免 API Key/token 等敏感信息进入日志。
        # 仅记录类型和长度，便于排查又不泄露敏感数据。
        logger.debug("config value 转换失败（type=%s, len=%d），保留字符串",
                     type(text).__name__, len(text) if text else 0)
        return value
