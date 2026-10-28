"""输入验证规则与 API 共用的 Schema。

API 在协议边界直接调用 InputValidator.validate(data, schema)，
验证字符串、类型、范围和已配置的安全规则；失败交由各入口转换为响应。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 字段类型枚举
# ---------------------------------------------------------------------------

class FieldType(Enum):
    """支持的字段类型。"""
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    LIST = "list"


# ---------------------------------------------------------------------------
# 字段规则定义
# ---------------------------------------------------------------------------

@dataclass
class FieldRule:
    """单个字段的验证规则。

    Attributes:
        field_type: 字段期望类型
        required: 是否必填
        min_length: 字符串最小长度
        max_length: 字符串最大长度
        min_value: 数值最小值
        max_value: 数值最大值
        allowed_values: 允许的枚举值列表
        check_xss: 是否检测XSS
        check_path_traversal: 是否检测路径遍历
        sanitize: 是否拒绝非法控制字符（不改写原文）
    """

    field_type: FieldType = FieldType.STRING
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list[Any]] = None
    check_xss: bool = False
    check_path_traversal: bool = False
    sanitize: bool = True


@dataclass
class Schema:
    """验证模式，定义一组字段的规则。

    Attributes:
        fields: 字段名到规则的映射
        allow_extra: 是否允许模式中未定义的额外字段
    """

    fields: dict[str, FieldRule] = field(default_factory=dict)
    allow_extra: bool = False


# ---------------------------------------------------------------------------
# 安全检测模式
# ---------------------------------------------------------------------------

# XSS检测模式
_XSS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"\bon\w+\s*=", re.IGNORECASE),  # onerror=, onload=, etc.
    re.compile(r"<\s*iframe", re.IGNORECASE),
    re.compile(r"<\s*object", re.IGNORECASE),
    re.compile(r"<\s*embed", re.IGNORECASE),
    re.compile(r"<\s*form", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
    re.compile(r"expression\s*\(", re.IGNORECASE),
    re.compile(r"vbscript\s*:", re.IGNORECASE),
    re.compile(r"data\s*:\s*text/html", re.IGNORECASE),
]

# 路径遍历模式
_PATH_TRAVERSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.\./"),
    re.compile(r"\.\.\\"),
    re.compile(r"\.\.%2[fF]"),
    re.compile(r"%2[eE]%2[eE]/"),
    re.compile(r"%2[eE]%2[eE]\\"),
    re.compile(r"\.\.%5[cC]"),
    re.compile(r"~/" ),
    re.compile(r"~\\"),
]

# 控制字符模式（\x00-\x1f，排除\t\n\r）
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ---------------------------------------------------------------------------
# 预定义Schema
# ---------------------------------------------------------------------------

KNOWLEDGE_DOCUMENT_SCHEMA = Schema(
    fields={
        "title": FieldRule(
            field_type=FieldType.STRING,
            required=True,
            min_length=1,
            max_length=200,
            check_xss=True,
            sanitize=True,
        ),
        "content": FieldRule(
            field_type=FieldType.STRING,
            required=True,
            min_length=1,
            max_length=50000,
            check_xss=True,
            sanitize=True,
        ),
        "category": FieldRule(
            field_type=FieldType.STRING,
            required=False,
            max_length=100,
            sanitize=True,
        ),
        "tags": FieldRule(
            field_type=FieldType.LIST,
            required=False,
        ),
    },
    allow_extra=True,
)

TRAINING_SCHEMA = Schema(
    fields={
        "lora_name": FieldRule(
            field_type=FieldType.STRING,
            required=True,
            min_length=1,
            max_length=200,
            check_path_traversal=True,
            sanitize=True,
        ),
        "dataset_name": FieldRule(
            field_type=FieldType.STRING,
            required=True,
            min_length=1,
            max_length=100,
            check_path_traversal=True,
            sanitize=True,
        ),
        "model_type": FieldRule(
            field_type=FieldType.STRING,
            required=True,
            min_length=1,
            max_length=200,
            check_path_traversal=True,
            sanitize=True,
        ),
        "style": FieldRule(
            field_type=FieldType.STRING,
            required=False,
            max_length=100,
            sanitize=True,
        ),
        "epochs": FieldRule(
            field_type=FieldType.INTEGER,
            required=False,
            min_value=1,
            max_value=100,
        ),
        "batch_size": FieldRule(
            field_type=FieldType.INTEGER,
            required=False,
            min_value=1,
            max_value=32,
        ),
        "learning_rate": FieldRule(
            field_type=FieldType.FLOAT,
            required=False,
            min_value=1e-7,
            max_value=1.0,
        ),
        "lora_rank": FieldRule(
            field_type=FieldType.INTEGER,
            required=False,
            min_value=1,
            max_value=256,
        ),
    },
    allow_extra=True,
)

CONFIG_SCHEMA = Schema(
    fields={
        "key": FieldRule(
            field_type=FieldType.STRING,
            required=True,
            min_length=1,
            max_length=100,
            sanitize=True,
        ),
        "value": FieldRule(
            field_type=FieldType.STRING,
            required=True,
            max_length=10000,
            check_xss=True,
            sanitize=True,
        ),
        "provider": FieldRule(
            field_type=FieldType.STRING,
            required=False,
            max_length=50,
            allowed_values=["local", "openai", "zhipu", "deepseek", "ollama"],
        ),
    },
    allow_extra=True,
)


# ---------------------------------------------------------------------------
# 核心验证器
# ---------------------------------------------------------------------------

class InputValidator:
    """集中管理所有API输入验证规则。

    提供静态方法用于验证数据、检测安全威胁和清理输入字符串。
    所有验证方法均为无状态的，可安全并发调用。
    """

    @staticmethod
    def validate(data: dict[str, Any], schema: Schema) -> tuple[bool, list[dict[str, str]]]:
        """根据给定的Schema验证数据。

        Args:
            data: 待验证的数据字典
            schema: 验证模式定义

        Returns:
            (is_valid, errors) 元组：
            - is_valid: 验证是否通过
            - errors: 错误列表，每项包含 field 和 message
        """
        errors: list[dict[str, str]] = []

        if not isinstance(data, dict):
            errors.append({"field": "_root", "message": "输入数据必须是字典类型"})
            return False, errors

        # 检查必填字段
        for field_name, rule in schema.fields.items():
            if rule.required and field_name not in data:
                errors.append({"field": field_name, "message": "此字段为必填项"})
                continue

        # 验证每个存在的字段
        for field_name, value in data.items():
            if field_name not in schema.fields:
                if not schema.allow_extra:
                    errors.append({"field": field_name, "message": "未定义的字段"})
                continue

            rule = schema.fields[field_name]
            field_errors = InputValidator._validate_field(field_name, value, rule)
            errors.extend(field_errors)

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning("输入验证失败: %s", errors)
        return is_valid, errors

    @staticmethod
    def _validate_field(field_name: str, value: Any, rule: FieldRule) -> list[dict[str, str]]:
        """验证单个字段。

        Args:
            field_name: 字段名称
            value: 字段值
            rule: 字段验证规则

        Returns:
            错误列表
        """
        errors: list[dict[str, str]] = []

        # 允许非必填字段为None
        if value is None:
            if rule.required:
                errors.append({"field": field_name, "message": "此字段不能为None"})
            return errors

        # 类型验证
        type_error = InputValidator._validate_type(field_name, value, rule.field_type)
        if type_error:
            errors.append(type_error)
            return errors  # 类型不对，后续验证无意义

        # 字符串类型验证
        if rule.field_type == FieldType.STRING and isinstance(value, str):
            str_errors = InputValidator._validate_string_field(field_name, value, rule)
            errors.extend(str_errors)

        # 数值类型验证
        if rule.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            num_errors = InputValidator._validate_numeric_field(field_name, value, rule)
            errors.extend(num_errors)

        # 列表类型验证
        if rule.field_type == FieldType.LIST and isinstance(value, list):
            list_errors = InputValidator._validate_list_field(field_name, value, rule)
            errors.extend(list_errors)

        # 枚举值验证
        if rule.allowed_values is not None and value not in rule.allowed_values:
            errors.append({
                "field": field_name,
                "message": f"值不在允许范围内，允许值: {rule.allowed_values}",
            })

        return errors

    @staticmethod
    def _validate_type(field_name: str, value: Any, expected_type: FieldType) -> Optional[dict[str, str]]:
        """验证字段类型。"""
        type_checks: dict[FieldType, tuple[type, ...]] = {
            FieldType.STRING: (str,),
            FieldType.INTEGER: (int,),  # 注意：bool是int的子类，需要额外排除
            FieldType.FLOAT: (int, float),
            FieldType.LIST: (list,),
        }

        expected = type_checks[expected_type]

        # 特殊处理：bool不是int
        if expected_type == FieldType.INTEGER and isinstance(value, bool):
            return {"field": field_name, "message": f"期望类型 {expected_type.value}，实际为 bool"}

        if not isinstance(value, expected):
            actual_type = type(value).__name__
            return {"field": field_name, "message": f"期望类型 {expected_type.value}，实际为 {actual_type}"}

        return None

    @staticmethod
    def _validate_string_field(field_name: str, value: str, rule: FieldRule) -> list[dict[str, str]]:
        """验证字符串类型字段的各项规则。"""
        errors: list[dict[str, str]] = []

        # 长度验证
        if rule.min_length is not None and len(value) < rule.min_length:
            errors.append({
                "field": field_name,
                "message": f"长度不能少于 {rule.min_length} 个字符",
            })
        if rule.max_length is not None and len(value) > rule.max_length:
            errors.append({
                "field": field_name,
                "message": f"长度不能超过 {rule.max_length} 个字符",
            })

        # 安全检测

        if rule.check_xss:
            xss_error = InputValidator._check_xss(field_name, value)
            if xss_error:
                errors.append(xss_error)

        if rule.check_path_traversal:
            path_error = InputValidator._check_path_traversal(field_name, value)
            if path_error:
                errors.append(path_error)


        # 控制字符检测
        if rule.sanitize and _CONTROL_CHAR_PATTERN.search(value):
            errors.append({
                "field": field_name,
                "message": "包含非法控制字符",
            })

        return errors

    @staticmethod
    def _validate_numeric_field(field_name: str, value: Any, rule: FieldRule) -> list[dict[str, str]]:
        """验证数值类型字段。"""
        errors: list[dict[str, str]] = []

        if rule.min_value is not None and value < rule.min_value:
            errors.append({
                "field": field_name,
                "message": f"值不能小于 {rule.min_value}",
            })
        if rule.max_value is not None and value > rule.max_value:
            errors.append({
                "field": field_name,
                "message": f"值不能大于 {rule.max_value}",
            })

        return errors

    @staticmethod
    def _validate_list_field(field_name: str, value: list, rule: FieldRule) -> list[dict[str, str]]:
        """验证列表类型字段。"""
        errors: list[dict[str, str]] = []

        if rule.min_length is not None and len(value) < rule.min_length:
            errors.append({
                "field": field_name,
                "message": f"列表元素数量不能少于 {rule.min_length}",
            })
        if rule.max_length is not None and len(value) > rule.max_length:
            errors.append({
                "field": field_name,
                "message": f"列表元素数量不能超过 {rule.max_length}",
            })

        return errors

    # -----------------------------------------------------------------------
    # 安全检测方法
    # -----------------------------------------------------------------------


    @staticmethod
    def _check_xss(field_name: str, value: str) -> Optional[dict[str, str]]:
        """检测XSS攻击模式。"""
        for pattern in _XSS_PATTERNS:
            if pattern.search(value):
                logger.warning("XSS检测触发: field=%s, pattern=%s", field_name, pattern.pattern)
                return {
                    "field": field_name,
                    "message": "输入包含潜在的XSS攻击内容",
                }
        return None

    @staticmethod
    def _check_path_traversal(field_name: str, value: str) -> Optional[dict[str, str]]:
        """检测路径遍历模式。"""
        for pattern in _PATH_TRAVERSAL_PATTERNS:
            if pattern.search(value):
                logger.warning("路径遍历检测触发: field=%s, pattern=%s", field_name, pattern.pattern)
                return {
                    "field": field_name,
                    "message": "输入包含潜在的路径遍历内容",
                }
        return None
