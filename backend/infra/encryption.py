"""
敏感数据加密存储模块 - AES-256-GCM加密

提供敏感数据的加密和解密功能，包括：
- AES-256-GCM加密算法
- 密钥管理（环境变量读取、自动生成、密钥轮换）
- 字段级加密/解密
- 批量加密/解密
- 敏感字段自动识别
- 数据库集成（写入前加密、读取后解密）

使用方式：
    from encryption import EncryptionManager

    manager = EncryptionManager()

    # 加密/解密
    encrypted = manager.encrypt("sensitive data")
    decrypted = manager.decrypt(encrypted)

    # 字段级加密
    encrypted_value = manager.encrypt_field("my_api_key", "api_key")

    # 批量加密
    items = [{"api_key": "key1", "name": "test"}]
    encrypted_items = manager.encrypt_batch(items, ["api_key"])
"""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 加密数据前缀标识
_ENCRYPTION_PREFIX = "ENC:AES256GCM:"

# 敏感字段名称模式（不区分大小写匹配）
_SENSITIVE_FIELD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:api[_-]?key|apikey)", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"passwd", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"private[_-]?key", re.IGNORECASE),
    re.compile(r"access[_-]?key", re.IGNORECASE),
    re.compile(r"auth[_-]?key", re.IGNORECASE),
    re.compile(r"encryption[_-]?key", re.IGNORECASE),
    re.compile(r"database[_-]?url", re.IGNORECASE),
    re.compile(r"connection[_-]?string", re.IGNORECASE),
]

# 密钥长度：AES-256需要32字节
_KEY_LENGTH = 32

# GCM nonce长度
_NONCE_LENGTH = 12

# 环境变量名
_ENV_KEY_NAME = "ENCRYPTION_KEY"

# 后端只使用 backend/.env。旧版本曾误写到 backend/infra/.env，保留只读迁移入口。
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE_PATH = _BACKEND_ROOT / ".env"
_LEGACY_ENV_FILE_PATH = Path(__file__).resolve().parent / ".env"
_PRODUCTION_ENVIRONMENTS = {"production", "prod"}


# ---------------------------------------------------------------------------
# 异常类
# ---------------------------------------------------------------------------

class EncryptionError(Exception):
    """加密操作失败时抛出的异常。"""
    pass


class DecryptionError(Exception):
    """解密操作失败时抛出的异常。"""
    pass


class KeyManagementError(Exception):
    """密钥管理操作失败时抛出的异常。"""
    pass


# ---------------------------------------------------------------------------
# 加密管理器
# ---------------------------------------------------------------------------

class EncryptionManager:
    """管理敏感数据的加密和解密。

    使用AES-256-GCM加密算法，密钥从环境变量读取或自动生成。
    支持字段级加密、批量操作和密钥轮换。

    Attributes:
        _key: 当前使用的加密密钥（32字节）
        _aesgcm: AESGCM实例
    """

    def __init__(self, key: Optional[bytes] = None) -> None:
        """初始化加密管理器。

        Args:
            key: 可选的加密密钥（32字节）。如果未提供，
                 将从环境变量ENCRYPTION_KEY读取，或自动生成新密钥。

        Raises:
            KeyManagementError: 密钥初始化失败
        """
        if key is not None:
            if len(key) != _KEY_LENGTH:
                raise KeyManagementError(
                    f"密钥长度必须为 {_KEY_LENGTH} 字节，实际为 {len(key)} 字节"
                )
            self._key = key
        else:
            self._key = self._load_or_generate_key()

        self._aesgcm = AESGCM(self._key)
        logger.info("EncryptionManager 初始化完成")

    # -------------------------------------------------------------------
    # 密钥管理
    # -------------------------------------------------------------------

    @staticmethod
    def _decode_key(encoded_key: str, source: str) -> bytes:
        """Decode and validate one URL-safe base64 AES-256 key."""
        try:
            key = base64.urlsafe_b64decode(encoded_key.encode("ascii"))
        except Exception as exc:
            raise KeyManagementError(f"解析 {source} 中的 {_ENV_KEY_NAME} 失败") from exc
        if len(key) != _KEY_LENGTH:
            raise KeyManagementError(
                f"{source} 中的 {_ENV_KEY_NAME} 必须解码为 {_KEY_LENGTH} 字节"
            )
        return key

    @staticmethod
    def _read_key_from_file(path: Path) -> Optional[str]:
        """Read ENCRYPTION_KEY without mutating the process environment."""
        if not path.exists():
            return None
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                name, separator, value = line.partition("=")
                if separator and name.strip() == _ENV_KEY_NAME:
                    return value.strip().strip('"').strip("'") or None
        except OSError as exc:
            raise KeyManagementError(f"读取密钥文件 {path} 失败") from exc
        return None

    def _load_or_generate_key(self) -> bytes:
        """Load the canonical key, migrate the legacy path, or create a dev key."""
        candidates = (
            ("环境变量", os.environ.get(_ENV_KEY_NAME)),
            (str(_ENV_FILE_PATH), self._read_key_from_file(_ENV_FILE_PATH)),
        )
        for source, encoded_key in candidates:
            if not encoded_key:
                continue
            key = self._decode_key(encoded_key, source)
            os.environ[_ENV_KEY_NAME] = encoded_key
            logger.info("从%s加载加密密钥成功", source)
            return key

        legacy_key = self._read_key_from_file(_LEGACY_ENV_FILE_PATH)
        if legacy_key:
            key = self._decode_key(legacy_key, str(_LEGACY_ENV_FILE_PATH))
            self._save_key_to_env(key)
            logger.warning(
                "已将旧路径 %s 中的加密密钥迁移到 %s；确认服务正常后可手动删除旧文件",
                _LEGACY_ENV_FILE_PATH,
                _ENV_FILE_PATH,
            )
            return key

        environment = os.getenv("ENVIRONMENT", "development").strip().lower()
        if environment in _PRODUCTION_ENVIRONMENTS:
            raise KeyManagementError(
                "生产环境必须显式配置 ENCRYPTION_KEY，禁止自动生成"
            )

        logger.warning("未找到加密密钥，正在为开发环境生成并持久化新密钥")
        key = AESGCM.generate_key(bit_length=256)
        self._save_key_to_env(key)
        return key

    def _save_key_to_env(self, key: bytes) -> None:
        """将密钥保存到.env文件。

        Args:
            key: 要保存的密钥

        Raises:
            KeyManagementError: 保存失败
        """
        encoded_key = base64.urlsafe_b64encode(key).decode("ascii")

        temp_path = _ENV_FILE_PATH.with_name(
            f".{_ENV_FILE_PATH.name}.{os.getpid()}.tmp"
        )
        try:
            env_lines: list[str] = []
            key_written = False

            if _ENV_FILE_PATH.exists():
                env_lines = _ENV_FILE_PATH.read_text(encoding="utf-8").splitlines(
                    keepends=True
                )
                for index, line in enumerate(env_lines):
                    name, separator, _ = line.strip().partition("=")
                    if separator and name.strip() == _ENV_KEY_NAME:
                        env_lines[index] = f"{_ENV_KEY_NAME}={encoded_key}\n"
                        key_written = True
                        break

            if env_lines and not env_lines[-1].endswith(("\n", "\r")):
                env_lines[-1] += "\n"
            if not key_written:
                env_lines.append(f"{_ENV_KEY_NAME}={encoded_key}\n")

            _ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text("".join(env_lines), encoding="utf-8")
            os.replace(temp_path, _ENV_FILE_PATH)
            if os.name != "nt":
                os.chmod(_ENV_FILE_PATH, 0o600)

            os.environ[_ENV_KEY_NAME] = encoded_key
            logger.info("加密密钥已保存到 %s", _ENV_FILE_PATH)
        except Exception as exc:
            raise KeyManagementError(
                f"保存密钥到 .env 文件失败: {exc}"
            ) from exc
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("无法清理临时密钥文件 %s", temp_path)

    def rotate_key(self, new_key: Optional[bytes] = None) -> None:
        """密钥轮换：使用新密钥替换当前密钥。

        注意：轮换后，使用旧密钥加密的数据将无法解密。
        需要先解密所有数据，再使用新密钥重新加密。

        Args:
            new_key: 新的加密密钥（32字节）。如果未提供，将自动生成。

        Raises:
            KeyManagementError: 密钥轮换失败
        """
        if new_key is not None and len(new_key) != _KEY_LENGTH:
            raise KeyManagementError(
                f"新密钥长度必须为 {_KEY_LENGTH} 字节，实际为 {len(new_key)} 字节"
            )

        old_key = self._key

        try:
            if new_key is None:
                new_key = AESGCM.generate_key(bit_length=256)

            self._key = new_key
            self._aesgcm = AESGCM(self._key)
            self._save_key_to_env(new_key)

            logger.warning(
                "密钥轮换完成。旧密钥已失效，使用旧密钥加密的数据需要重新加密。"
            )
        except Exception as exc:
            # 回滚到旧密钥
            self._key = old_key
            self._aesgcm = AESGCM(self._key)
            raise KeyManagementError(f"密钥轮换失败: {exc}") from exc

    # -------------------------------------------------------------------
    # 加密/解密核心方法
    # -------------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """加密明文字符串。

        使用AES-256-GCM算法加密，返回格式化的加密字符串。

        Args:
            plaintext: 待加密的明文

        Returns:
            加密后的字符串，格式为 ENC:AES256GCM:{iv}:{ciphertext}:{tag}

        Raises:
            EncryptionError: 加密失败
        """
        if not isinstance(plaintext, str):
            raise EncryptionError("加密输入必须是字符串类型")

        if not plaintext:
            return ""

        try:
            # 生成随机nonce
            nonce = os.urandom(_NONCE_LENGTH)

            # 加密（AESGCM.encrypt返回 ciphertext+tag）
            plaintext_bytes = plaintext.encode("utf-8")
            ciphertext_with_tag = self._aesgcm.encrypt(nonce, plaintext_bytes, None)

            # 分离密文和tag（GCM tag为最后16字节）
            ciphertext = ciphertext_with_tag[:-16]
            tag = ciphertext_with_tag[-16:]

            # Base64编码各部分
            iv_b64 = base64.urlsafe_b64encode(nonce).decode("ascii")
            ct_b64 = base64.urlsafe_b64encode(ciphertext).decode("ascii")
            tag_b64 = base64.urlsafe_b64encode(tag).decode("ascii")

            encrypted = f"{_ENCRYPTION_PREFIX}{iv_b64}:{ct_b64}:{tag_b64}"
            logger.debug("数据加密成功，长度: %d -> %d", len(plaintext), len(encrypted))
            return encrypted

        except Exception as exc:
            logger.error("加密失败: %s", exc)
            raise EncryptionError(f"加密失败: {exc}") from exc

    def decrypt(self, encrypted_str: str) -> str:
        """解密加密字符串。

        Args:
            encrypted_str: 加密字符串，格式为 ENC:AES256GCM:{iv}:{ciphertext}:{tag}

        Returns:
            解密后的明文字符串

        Raises:
            DecryptionError: 解密失败（格式错误、密钥不匹配等）
        """
        if not isinstance(encrypted_str, str):
            raise DecryptionError("解密输入必须是字符串类型")

        if not encrypted_str:
            return ""

        if not encrypted_str.startswith(_ENCRYPTION_PREFIX):
            # 未加密的数据直接返回（兼容性处理）
            return encrypted_str

        try:
            # 解析加密字符串
            parts = encrypted_str[len(_ENCRYPTION_PREFIX):].split(":")
            if len(parts) != 3:
                raise DecryptionError(f"加密数据格式错误，期望3部分，实际为{len(parts)}部分")

            iv_b64, ct_b64, tag_b64 = parts

            # Base64解码
            nonce = base64.urlsafe_b64decode(iv_b64)
            ciphertext = base64.urlsafe_b64decode(ct_b64)
            tag = base64.urlsafe_b64decode(tag_b64)

            # 重组 ciphertext+tag（AESGCM.decrypt需要合并格式）
            ciphertext_with_tag = ciphertext + tag

            # 解密
            plaintext_bytes = self._aesgcm.decrypt(nonce, ciphertext_with_tag, None)
            plaintext = plaintext_bytes.decode("utf-8")

            logger.debug("数据解密成功")
            return plaintext

        except DecryptionError:
            raise
        except Exception as exc:
            logger.error("解密失败: %s", exc)
            raise DecryptionError(f"解密失败: {exc}") from exc

    # -------------------------------------------------------------------
    # 字段级加密/解密
    # -------------------------------------------------------------------

    def encrypt_field(self, value: Any, field_name: str) -> Any:
        """对特定字段值进行加密。

        如果字段名匹配敏感字段模式，则加密该值。
        非字符串值和已加密的值不会被重复加密。

        Args:
            value: 字段值
            field_name: 字段名称

        Returns:
            加密后的值（如果需要加密）或原值
        """
        if not isinstance(value, str) or not value:
            return value

        # 已加密的数据不重复加密
        if self.is_encrypted(value):
            return value

        # 检查是否为敏感字段
        if self._is_sensitive_field(field_name):
            return self.encrypt(value)

        return value

    def decrypt_field(self, value: Any, field_name: str) -> Any:
        """对特定字段值进行解密。

        仅对已加密的字符串值进行解密。

        Args:
            value: 字段值
            field_name: 字段名称

        Returns:
            解密后的值（如果已加密）或原值
        """
        if not isinstance(value, str) or not value:
            return value

        if self.is_encrypted(value):
            return self.decrypt(value)

        return value

    # -------------------------------------------------------------------
    # 批量加密/解密
    # -------------------------------------------------------------------

    def encrypt_batch(self, items: list[dict[str, Any]], fields: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """批量加密数据项中的指定字段。

        如果未指定字段列表，则自动识别并加密所有敏感字段。

        Args:
            items: 待加密的数据项列表
            fields: 需要加密的字段名列表。如果为None，自动识别敏感字段。

        Returns:
            加密后的数据项列表（原地修改并返回）
        """
        if not items:
            return items

        for item in items:
            if not isinstance(item, dict):
                continue

            target_fields = fields if fields else self._find_sensitive_fields(item)

            for field_name in target_fields:
                if field_name in item:
                    item[field_name] = self.encrypt_field(item[field_name], field_name)

        logger.debug("批量加密完成，处理 %d 条数据", len(items))
        return items

    def decrypt_batch(self, items: list[dict[str, Any]], fields: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """批量解密数据项中的指定字段。

        如果未指定字段列表，则自动识别并解密所有已加密字段。

        Args:
            items: 待解密的数据项列表
            fields: 需要解密的字段名列表。如果为None，自动识别已加密字段。

        Returns:
            解密后的数据项列表（原地修改并返回）
        """
        if not items:
            return items

        for item in items:
            if not isinstance(item, dict):
                continue

            if fields:
                target_fields = fields
            else:
                # 自动查找已加密的字段
                target_fields = [
                    k for k, v in item.items()
                    if isinstance(v, str) and self.is_encrypted(v)
                ]

            for field_name in target_fields:
                if field_name in item:
                    item[field_name] = self.decrypt_field(item[field_name], field_name)

        logger.debug("批量解密完成，处理 %d 条数据", len(items))
        return items

    # -------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------

    @staticmethod
    def is_encrypted(value: str) -> bool:
        """检查字符串是否为加密数据。

        Args:
            value: 待检查的字符串

        Returns:
            True如果字符串是加密数据格式
        """
        if not isinstance(value, str):
            return False
        return value.startswith(_ENCRYPTION_PREFIX)

    @staticmethod
    def _is_sensitive_field(field_name: str) -> bool:
        """判断字段名是否匹配敏感字段模式。

        Args:
            field_name: 字段名称

        Returns:
            True如果字段名匹配敏感字段模式
        """
        for pattern in _SENSITIVE_FIELD_PATTERNS:
            if pattern.search(field_name):
                return True
        return False

    @staticmethod
    def _find_sensitive_fields(item: dict[str, Any]) -> list[str]:
        """在数据项中查找所有敏感字段。

        Args:
            item: 数据项字典

        Returns:
            匹配敏感模式的字段名列表
        """
        sensitive_fields: list[str] = []
        for field_name, value in item.items():
            if isinstance(value, str) and value and EncryptionManager._is_sensitive_field(field_name):
                sensitive_fields.append(field_name)
        return sensitive_fields


# ---------------------------------------------------------------------------
# 数据库集成辅助
# ---------------------------------------------------------------------------

class DatabaseEncryptionMiddleware:
    """数据库加密中间件，在SQLite写入前自动加密、读取后自动解密。

    使用方式：
        middleware = DatabaseEncryptionMiddleware(encryption_manager)

        # 写入前加密
        row = middleware.before_write(row_data)

        # 读取后解密
        row = middleware.after_read(row_data)
    """

    def __init__(self, encryption_manager: EncryptionManager) -> None:
        """初始化数据库加密中间件。

        Args:
            encryption_manager: 加密管理器实例
        """
        self._manager = encryption_manager

    def before_write(self, row: dict[str, Any]) -> dict[str, Any]:
        """在数据库写入前加密敏感字段。

        Args:
            row: 待写入的数据行

        Returns:
            加密后的数据行
        """
        return self._manager.encrypt_batch([row])[0] if row else row

    def after_read(self, row: dict[str, Any]) -> dict[str, Any]:
        """在数据库读取后解密敏感字段。

        Args:
            row: 从数据库读取的数据行

        Returns:
            解密后的数据行
        """
        return self._manager.decrypt_batch([row])[0] if row else row

    def before_write_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """批量写入前加密。

        Args:
            rows: 待写入的数据行列表

        Returns:
            加密后的数据行列表
        """
        return self._manager.encrypt_batch(rows)

    def after_read_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """批量读取后解密。

        Args:
            rows: 从数据库读取的数据行列表

        Returns:
            解密后的数据行列表
        """
        return self._manager.decrypt_batch(rows)


# ---------------------------------------------------------------------------
# 全局单例（懒加载）
# ---------------------------------------------------------------------------

_encryption_manager: Optional[EncryptionManager] = None


def get_encryption_manager() -> EncryptionManager:
    """获取全局加密管理器单例。

    Returns:
        EncryptionManager实例
    """
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager
