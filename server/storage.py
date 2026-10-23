"""存储抽象（全栈层架构 · 六；第四步新增 MinIO 实现）。

开发用本地 FS；生产切 MinIO/S3 时实现同一个 Storage 接口即可，
执行器与路由层零改动。装配点唯一：storage_from_env()（第四步起
app.create_app 与 tasks.get_runtime 均经它取实例）——
设 MINIO_ENDPOINT 环境变量即整体切到对象存储，否则本地 FS。
"""
from __future__ import annotations

import os
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


class MinIOStorage(Storage):
    """MinIO / S3 对象存储（第四步）。

    - save：fput_object 上传，返回 key（result_url 语义与本地版一致）
    - path_of：惰性 fget_object 下载到本地缓存目录后返回 Path ——
      与 download 路由的 FileResponse / .exists() 判断天然兼容，
      上层零改动
    - 成片 TTL 清理随对象存储生命周期规则统一管理（本类不删对象）
    - minio SDK 延迟导入：纯本地部署环境无需安装
    """

    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str, cache_dir: str | Path, secure: bool = False):
        from minio import Minio              # 延迟导入（见类 docstring）

        self.client = Minio(endpoint, access_key=access_key,
                            secret_key=secret_key, secure=secure)
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, src_path: str, key: str) -> str:
        self.client.fput_object(self.bucket, key, str(src_path))
        return key

    def path_of(self, key: str) -> Path:
        dest = self.cache_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            self.client.fget_object(self.bucket, key, str(dest))
        return dest


def storage_from_env(local_root: str | Path,
                     cache_dir: str | Path | None = None) -> Storage:
    """唯一装配点：MINIO_ENDPOINT 存在 → MinIOStorage，否则 LocalStorage。

    LocalStorage → MinIO 切换只改环境变量，上层（路由 / 任务体）零改动。
    """
    endpoint = os.environ.get("MINIO_ENDPOINT")
    if not endpoint:
        return LocalStorage(local_root)
    return MinIOStorage(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        bucket=os.environ.get("MINIO_BUCKET", "renders"),
        cache_dir=cache_dir or Path(local_root).parent / "minio_cache",
        secure=os.environ.get("MINIO_SECURE", "0") == "1",
    )
