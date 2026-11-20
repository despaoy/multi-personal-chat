"""Windows file operations for readers that outlive snapshot publication."""

import ctypes
import os
from contextlib import contextmanager
from ctypes import wintypes
from msvcrt import open_osfhandle

_GENERIC_READ = 0x80000000
_DELETE = 0x00010000
_SHARE_READ_WRITE_DELETE = 0x00000007
_OPEN_EXISTING = 3
_FILE_RENAME_INFO_EX = 22
_REPLACE_WITH_POSIX_SEMANTICS = 0x00000003
_INVALID_HANDLE = ctypes.c_void_p(-1).value

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_create_file = _kernel32.CreateFileW
_create_file.argtypes = (
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
)
_create_file.restype = wintypes.HANDLE
_close_handle = _kernel32.CloseHandle
_close_handle.argtypes = (wintypes.HANDLE,)
_close_handle.restype = wintypes.BOOL
_set_file_information = _kernel32.SetFileInformationByHandle
_set_file_information.argtypes = (
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
)
_set_file_information.restype = wintypes.BOOL


class _RenameInfo(ctypes.Structure):
    _fields_ = [
        ("flags", wintypes.DWORD),
        ("root_directory", wintypes.HANDLE),
        ("name_length", wintypes.DWORD),
        ("name", wintypes.WCHAR * 1),
    ]


def _open_handle(path, access):
    handle = _create_file(os.fspath(path), access, _SHARE_READ_WRITE_DELETE, None, _OPEN_EXISTING, 0, None)
    if handle == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


@contextmanager
def open_snapshot(path):
    handle = _open_handle(path, _GENERIC_READ)
    try:
        descriptor = open_osfhandle(handle, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
    except BaseException:
        _close_handle(handle)
        raise
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            yield stream
    finally:
        os.close(descriptor)


def replace_snapshot(candidate, target):
    try:
        os.replace(candidate, target)
        return
    except PermissionError as error:
        if error.winerror not in (5, 32):
            raise

    # MoveFileEx cannot replace an open destination even with delete sharing.
    # POSIX rename keeps existing readers attached to the previous file.
    name = os.path.abspath(target).encode("utf-16-le", errors="surrogatepass")
    size = max(ctypes.sizeof(_RenameInfo), _RenameInfo.name.offset + len(name) + 2)
    buffer = ctypes.create_string_buffer(size)
    rename = _RenameInfo.from_buffer(buffer)
    rename.flags = _REPLACE_WITH_POSIX_SEMANTICS
    rename.name_length = len(name)
    ctypes.memmove(ctypes.addressof(buffer) + _RenameInfo.name.offset, name, len(name))

    handle = _open_handle(candidate, _DELETE)
    try:
        if not _set_file_information(handle, _FILE_RENAME_INFO_EX, buffer, size):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _close_handle(handle)
