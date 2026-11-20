"""Read-only document source storage addressed by relative object keys."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from stat import S_ISDIR, S_ISREG
from typing import Protocol


class SourceBoundaryError(ValueError):
    """A requested source is outside the configured storage root."""


def source_key(value: str | PurePosixPath) -> PurePosixPath:
    key = PurePosixPath(value)
    if key.is_absolute() or ".." in key.parts or PureWindowsPath(str(key)).drive:
        raise SourceBoundaryError("来源路径超出知识库目录范围")
    return key


@dataclass(frozen=True)
class SourceEntry:
    directory: bool
    size: int
    identity: str


class KnowledgeSourceStorage(Protocol):
    """Provide metadata, immediate child keys and document bytes within one source root."""

    def info(self, key: PurePosixPath) -> SourceEntry | None: ...

    def children(self, key: PurePosixPath) -> list[PurePosixPath]: ...

    def read_bytes(self, key: PurePosixPath) -> bytes: ...


class LocalKnowledgeSourceStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _resolve(self, key: PurePosixPath) -> Path:
        try:
            path = self.root.joinpath(*source_key(key).parts).resolve()
        except RuntimeError as exc:
            raise SourceBoundaryError("知识库来源目录存在循环链接") from exc
        if not path.is_relative_to(self.root):
            raise SourceBoundaryError("来源路径超出知识库目录范围")
        return path

    def info(self, key: PurePosixPath) -> SourceEntry | None:
        path = self._resolve(key)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return None
        if S_ISDIR(metadata.st_mode):
            return SourceEntry(True, 0, str(path))
        if S_ISREG(metadata.st_mode):
            return SourceEntry(False, metadata.st_size, str(path))
        return None

    def children(self, key: PurePosixPath) -> list[PurePosixPath]:
        return [key / item.name for item in self._resolve(key).iterdir()]

    def read_bytes(self, key: PurePosixPath) -> bytes:
        return self._resolve(key).read_bytes()
