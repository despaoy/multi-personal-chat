"""
敏感数据加密存储模块 - AES-256-GCM加密

提供敏感数据的加密和解密功能，包括：
- AES-256-GCM加密算法
- 密钥管理（显式加载、历史路径迁移）
- 字段级加密/解密
- 敏感字段自动识别
- 由调用方显式执行加密和解密

使用方式：
    from infra.encryption import EncryptionManager

    manager = EncryptionManager()

    # 加密/解密
    encrypted = manager.encrypt("sensitive data")
    decrypted = manager.decrypt(encrypted)

    # 字段级加密
    encrypted_value = manager.encrypt_field("my_api_key", "api_key")

"""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from infra.deployment import decode_encryption_key

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

    使用AES-256-GCM加密算法，按配置优先级加载已有密钥。
    支持字段级加密、解密和批量加密。

    Attributes:
        _key: 当前使用的加密密钥（32字节）
        _aesgcm: AESGCM实例
    """

    def __init__(self, key: Optional[bytes] = None) -> None:
        """初始化加密管理器。

        Args:
            key: 可选的加密密钥（32字节）。如果未提供，
                 将按环境变量、backend/.env、历史路径加载已有密钥。

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
            self._key = self._load_key()

        self._aesgcm = AESGCM(self._key)
        logger.info("EncryptionManager 初始化完成")

    # -------------------------------------------------------------------
    # 密钥管理
    # -------------------------------------------------------------------

    @staticmethod
    def _decode_key(encoded_key: str, source: str) -> bytes:
        """Attach configuration-source information to the shared strict decoder."""
        try:
            return decode_encryption_key(encoded_key)
        except ValueError as exc:
            raise KeyManagementError(
                f"{source} 中的 {_ENV_KEY_NAME} 必须为有效的 {_KEY_LENGTH} 字节 Base64 编码"
            ) from exc

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

    def _load_key(self) -> bytes:
        """Load a configured key or migrate its legacy path; never create a replacement."""
        source = "环境变量"
        encoded_key = os.environ.get(_ENV_KEY_NAME)
        if not encoded_key:
            source = str(_ENV_FILE_PATH)
            encoded_key = self._read_key_from_file(_ENV_FILE_PATH)
        if encoded_key:
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

        raise KeyManagementError("必须显式配置 ENCRYPTION_KEY，禁止自动生成")

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
    # 字段级加密
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
