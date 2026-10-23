"""存储抽象（全栈层架构 · 六）。

开发用本地 FS；生产切 MinIO/S3 时实现同一个 Storage 接口即可，
执行器与路由层零改动。
"""
from __future__ import annotations

import shutil
from pathlib import Path


class Storage:
    def save(self, src_path: str, key: str) -> str:  # noqa: D102
        raise NotImplementedError

    def path_of(self, key: str) -> Path:  # noqa: D102
        raise NotImplementedError


class LocalStorage(Storage):
    """本地目录存储：save 把产物搬进 root/<key>，返回可下载的 key。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, src_path: str, key: str) -> str:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_path, dest)          # 同盘 move，零拷贝
        return key

    def path_of(self, key: str) -> Path:
        return self.root / key
