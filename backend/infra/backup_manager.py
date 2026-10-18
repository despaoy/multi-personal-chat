"""数据备份与恢复机制模块。

实现SQLite数据库的自动备份管理，支持全量备份和增量备份，
提供定时备份、事件触发备份、备份轮转、压缩存储和完整性校验。
"""

import asyncio
import gzip
import hashlib
import logging
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from tempfile import TemporaryDirectory
from uuid import uuid4

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """备份类型枚举。"""
    FULL = "full"
    INCREMENTAL = "incremental"


@dataclass
class BackupInfo:
    """备份文件信息。"""
    filename: str
    path: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    sha256: str
    db_name: str

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "filename": self.filename,
            "path": self.path,
            "backup_type": self.backup_type.value,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "db_name": self.db_name,
        }


@dataclass
class BackupStats:
    """备份统计信息。"""
    total_backups: int = 0
    total_size_bytes: int = 0
    full_backups: int = 0
    incremental_backups: int = 0
    oldest_backup: Optional[datetime] = None
    newest_backup: Optional[datetime] = None
    last_backup_time: Optional[datetime] = None
    last_restore_time: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "total_backups": self.total_backups,
            "total_size_bytes": self.total_size_bytes,
            "full_backups": self.full_backups,
            "incremental_backups": self.incremental_backups,
            "oldest_backup": self.oldest_backup.isoformat() if self.oldest_backup else None,
            "newest_backup": self.newest_backup.isoformat() if self.newest_backup else None,
            "last_backup_time": self.last_backup_time.isoformat() if self.last_backup_time else None,
            "last_restore_time": self.last_restore_time.isoformat() if self.last_restore_time else None,
        }


class BackupManager:
    """SQLite数据库备份管理器。

    支持全量备份和增量备份，定时备份和事件触发备份，
    备份文件压缩存储和轮转管理，恢复前安全副本创建和完整性校验。

    Args:
        db_path: SQLite数据库文件路径。
        backup_dir: 备份文件存储目录。
        schedule_interval_hours: 定时备份间隔（小时）。
        change_threshold: 事件触发备份的数据变更条数阈值。
        daily_retention_days: 每日备份保留天数。
        weekly_retention_weeks: 每周备份保留周数。
    """

    def __init__(
        self,
        db_path: str,
        backup_dir: str = "backups",
        schedule_interval_hours: float = 6.0,
        change_threshold: int = 100,
        daily_retention_days: int = 7,
        weekly_retention_weeks: int = 4,
    ) -> None:
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.schedule_interval_hours = schedule_interval_hours
        self.change_threshold = change_threshold
        self.daily_retention_days = daily_retention_days
        self.weekly_retention_weeks = weekly_retention_weeks

        self._stats = BackupStats()
        self._change_count: int = 0
        self._last_full_backup_time: Optional[datetime] = None
        self._scheduled_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._backups_cache: list[BackupInfo] = []

    def _generate_backup_filename(self, backup_type: BackupType) -> str:
        """生成备份文件名。

        格式: {db_name}_{timestamp}-{unique_id}_{type}.bak.gz
        """
        db_name = self.db_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{db_name}_{timestamp}-{uuid4().hex}_{backup_type.value}.bak.gz"

    def _compute_sha256(self, file_path: Path) -> str:
        """计算文件的SHA256哈希值。"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _backup_sync(self, backup_type: BackupType) -> BackupInfo:
        """Create a complete snapshot; publish it only after compression succeeds.

        The historical incremental type is also a complete SQLite snapshot.
        """
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Backup source database does not exist: {self.db_path}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        filename = self._generate_backup_filename(backup_type)
        final_path = self.backup_dir / filename
        with TemporaryDirectory(prefix=".backup-", dir=self.backup_dir) as workspace:
            snapshot = Path(workspace) / "snapshot.sqlite"
            compressed = Path(workspace) / filename
            with closing(sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(str(snapshot))) as target:
                    source.backup(target)
            sha256 = self._compute_sha256(snapshot)
            with snapshot.open("rb") as source, gzip.open(compressed, "wb") as target:
                shutil.copyfileobj(source, target)
            size_bytes = compressed.stat().st_size
            compressed.replace(final_path)

        backup_info = BackupInfo(
            filename=filename, path=str(final_path), backup_type=backup_type,
            timestamp=datetime.now(), size_bytes=size_bytes,
            sha256=sha256, db_name=self.db_path.stem,
        )
        if backup_type is BackupType.FULL:
            self._last_full_backup_time = backup_info.timestamp
            self._stats.full_backups += 1
        else:
            self._stats.incremental_backups += 1
        self._stats.total_backups += 1
        self._stats.total_size_bytes += size_bytes
        self._stats.last_backup_time = backup_info.timestamp
        self._update_time_range(backup_info.timestamp)
        self._change_count = 0
        self._backups_cache.append(backup_info)
        logger.info("备份完成 type=%s file=%s bytes=%d", backup_type.value, filename, size_bytes)
        return backup_info

    def _update_time_range(self, timestamp: datetime) -> None:
        """更新备份时间范围统计。"""
        if self._stats.oldest_backup is None or timestamp < self._stats.oldest_backup:
            self._stats.oldest_backup = timestamp
        if self._stats.newest_backup is None or timestamp > self._stats.newest_backup:
            self._stats.newest_backup = timestamp

    async def backup(self, backup_type: BackupType = BackupType.FULL) -> BackupInfo:
        """Return a completed backup or propagate its original failure."""
        return await asyncio.to_thread(self._backup_sync, BackupType(backup_type))

    def restore(self, backup_path: str) -> bool:
        """从指定备份恢复数据库。

        恢复前自动创建当前数据库的安全副本。

        Args:
            backup_path: 备份文件路径。

        Returns:
            是否恢复成功。
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            logger.error("备份文件不存在: %s", backup_path)
            return False

        if not self.db_path.exists():
            logger.error("当前数据库文件不存在: %s", self.db_path)
            return False

        try:
            # 创建安全副本
            safety_filename = (
                f"{self.db_path.stem}_safety_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            )
            safety_path = self.backup_dir / safety_filename
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(self.db_path), str(safety_path))
            logger.info("已创建安全副本: %s", safety_path)

            # 解压备份文件
            temp_db_path = self.backup_dir / f"_restore_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            with gzip.open(backup_file, "rb") as f_in:
                with open(temp_db_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 完整性校验
            sha256 = self._compute_sha256(temp_db_path)
            # 查找对应的备份信息
            backup_info = self._find_backup_info(backup_file.name)
            if backup_info and backup_info.sha256 != sha256:
                logger.error(
                    "备份完整性校验失败: 期望 %s, 实际 %s",
                    backup_info.sha256[:16], sha256[:16],
                )
                temp_db_path.unlink()
                return False

            # 验证备份文件是否为有效SQLite数据库
            try:
                test_conn = sqlite3.connect(str(temp_db_path))
                test_conn.execute("SELECT 1")
                test_conn.close()
            except sqlite3.Error as e:
                logger.error("备份文件不是有效的SQLite数据库: %s", e)
                temp_db_path.unlink()
                return False

            # 替换当前数据库
            shutil.move(str(temp_db_path), str(self.db_path))
            self._stats.last_restore_time = datetime.now()

            logger.info("数据库恢复成功: %s", backup_path)
            return True

        except Exception as e:
            logger.error("数据库恢复失败: %s", e)
            # 尝试从安全副本恢复
            safety_files = sorted(
                self.backup_dir.glob(f"{self.db_path.stem}_safety_*.db"),
                reverse=True,
            )
            if safety_files:
                try:
                    shutil.copy2(str(safety_files[0]), str(self.db_path))
                    logger.info("已从安全副本恢复: %s", safety_files[0])
                except Exception as restore_err:
                    logger.error("安全副本恢复也失败: %s", restore_err)
            return False

    def _find_backup_info(self, filename: str) -> Optional[BackupInfo]:
        """在缓存中查找备份信息。"""
        for info in self._backups_cache:
            if info.filename == filename:
                return info
        return None

    def list_backups(self) -> list[BackupInfo]:
        """列出所有备份文件信息。

        扫描备份目录，解析文件名并校验文件存在性。

        Returns:
            备份信息列表，按时间降序排列。
        """
        if not self.backup_dir.exists():
            return []

        backups: list[BackupInfo] = []
        for path in self.backup_dir.glob("*.bak.gz"):
            info = self._parse_backup_filename(path)
            if info is not None:
                backups.append(info)

        backups.sort(key=lambda b: b.timestamp, reverse=True)
        self._backups_cache = backups
        return backups

    def _parse_backup_filename(self, path: Path) -> Optional[BackupInfo]:
        """从文件名解析备份信息。

        文件名格式: {db_name}_{YYYYMMDD_HHMMSS}[-unique_id]_{type}.bak.gz
        """
        name = path.stem.replace(".bak", "")  # 去掉 .bak (stem 已去掉 .gz)
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            return None

        backup_type_str = parts[1]
        prefix = parts[0]

        # 从prefix中提取 db_name 和 timestamp
        # 格式: db_name_YYYYMMDD_HHMMSS
        prefix_parts = prefix.rsplit("_", 2)
        if len(prefix_parts) < 3:
            return None

        db_name = "_".join(prefix_parts[:-2])
        date_str = prefix_parts[-2]
        time_str = prefix_parts[-1]

        try:
            # Existing second-resolution names and new unique names share the same timestamp.
            time_str = time_str.partition("-")[0]
            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None

        try:
            backup_type = BackupType(backup_type_str)
        except ValueError:
            return None

        return BackupInfo(
            filename=path.name,
            path=str(path),
            backup_type=backup_type,
            timestamp=timestamp,
            size_bytes=path.stat().st_size,
            sha256="",  # 需要时再计算
            db_name=db_name,
        )

    def rotate_backups(self) -> int:
        """执行备份轮转策略。

        保留最近 daily_retention_days 天的每日备份，
        以及最近 weekly_retention_weeks 周的每周备份（每周保留最早的一个）。

        Returns:
            删除的备份数量。
        """
        backups = self.list_backups()
        if not backups:
            return 0

        now = datetime.now()
        daily_cutoff = now - timedelta(days=self.daily_retention_days)
        weekly_cutoff = now - timedelta(weeks=self.weekly_retention_weeks)

        # 按周分组，每周保留最早的一个备份
        weekly_keep: set[str] = set()
        weekly_backups: dict[str, list[BackupInfo]] = {}
        for b in backups:
            if b.timestamp < weekly_cutoff:
                continue
            week_key = b.timestamp.strftime("%Y_W%W")
            if week_key not in weekly_backups:
                weekly_backups[week_key] = []
            weekly_backups[week_key].append(b)

        for week_key, week_list in weekly_backups.items():
            # 超出每日保留期的周备份，只保留最早的一个
            earliest = min(week_list, key=lambda b: b.timestamp)
            if earliest.timestamp < daily_cutoff:
                weekly_keep.add(earliest.filename)

        # 确定要删除的备份
        to_delete: list[BackupInfo] = []
        for b in backups:
            if b.timestamp >= daily_cutoff:
                # 在每日保留期内，保留
                continue
            if b.filename in weekly_keep:
                # 每周保留的备份
                continue
            to_delete.append(b)

        # 执行删除
        deleted_count = 0
        for b in to_delete:
            try:
                Path(b.path).unlink()
                deleted_count += 1
                logger.info("轮转删除备份: %s", b.filename)
            except Exception as e:
                logger.error("删除备份失败 %s: %s", b.filename, e)

        # 更新统计
        self._stats.total_backups -= deleted_count
        if deleted_count > 0:
            self._stats.total_size_bytes = sum(
                b.size_bytes for b in self.list_backups()
            )

        return deleted_count

    async def record_change(self, count: int = 1) -> Optional[BackupInfo]:
        """记录数据变更，达到阈值时触发增量备份。

        Args:
            count: 变更条数。

        Returns:
            如果触发了备份则返回备份信息，否则返回None。
        """
        self._change_count += count
        if self._change_count >= self.change_threshold:
            logger.info(
                "数据变更达到阈值 (%d/%d)，触发增量备份",
                self._change_count, self.change_threshold,
            )
            return await self.backup(BackupType.INCREMENTAL)
        return None

    async def start_scheduled_backup(self) -> None:
        """启动定时备份任务。"""
        if self._running:
            logger.warning("定时备份任务已在运行")
            return

        self._running = True
        self._scheduled_task = asyncio.create_task(self._scheduled_backup_loop())
        logger.info(
            "定时备份任务已启动，间隔: %.1f 小时",
            self.schedule_interval_hours,
        )

    async def _scheduled_backup_loop(self) -> None:
        """定时备份循环。"""
        while self._running:
            try:
                await asyncio.sleep(self.schedule_interval_hours * 3600)
                if not self._running:
                    break

                logger.info("执行定时备份")
                # 交替全量和增量备份
                if self._last_full_backup_time is None or (
                    datetime.now() - self._last_full_backup_time
                ) > timedelta(hours=self.schedule_interval_hours * 4):
                    await self.backup(BackupType.FULL)
                else:
                    await self.backup(BackupType.INCREMENTAL)

                # 执行轮转
                self.rotate_backups()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("定时备份任务异常: %s", e)
                await asyncio.sleep(60)  # 出错后等待1分钟再重试

    async def stop_scheduled_backup(self) -> None:
        """停止定时备份任务。"""
        self._running = False
        if self._scheduled_task is not None:
            self._scheduled_task.cancel()
            try:
                await self._scheduled_task
            except asyncio.CancelledError:
                pass
            self._scheduled_task = None
        logger.info("定时备份任务已停止")

    def get_backup_stats(self) -> dict[str, Any]:
        """获取备份统计信息。

        Returns:
            包含备份数量、大小、时间范围等信息的字典。
        """
        return self._stats.to_dict()

    def verify_backup(self, backup_path: str) -> bool:
        """验证备份文件的完整性。

        Args:
            backup_path: 备份文件路径。

        Returns:
            是否通过完整性校验。
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            logger.error("备份文件不存在: %s", backup_path)
            return False

        try:
            # 尝试解压并验证
            temp_path = self.backup_dir / f"_verify_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            with gzip.open(backup_file, "rb") as f_in:
                with open(temp_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 计算哈希
            actual_sha256 = self._compute_sha256(temp_path)

            # 验证是否为有效SQLite数据库
            try:
                conn = sqlite3.connect(str(temp_path))
                conn.execute("SELECT 1")
                conn.close()
            except sqlite3.Error:
                temp_path.unlink()
                logger.error("备份文件不是有效的SQLite数据库: %s", backup_path)
                return False

            # 与备份信息中的哈希对比
            backup_info = self._find_backup_info(backup_file.name)
            if backup_info and backup_info.sha256:
                if actual_sha256 != backup_info.sha256:
                    temp_path.unlink()
                    logger.error(
                        "SHA256校验失败: 期望 %s, 实际 %s",
                        backup_info.sha256[:16], actual_sha256[:16],
                    )
                    return False

            temp_path.unlink()
            logger.info("备份文件完整性校验通过: %s", backup_path)
            return True

        except Exception as e:
            logger.error("备份文件完整性校验失败: %s", e)
            return False
