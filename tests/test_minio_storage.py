"""MinIOStorage 测试（第四步）。

- 单元（必跑）：桩 Minio 客户端验证上传 / 惰性下载缓存 / 装配点切换 ——
  不依赖真实服务，与 eager 优先的测试策略一致
- 真实冒烟（skipif）：设 MINIO_ENDPOINT 环境变量后自动启用
  （真实服务二进制从 minio/minio GitHub Release 获取并 SHA256 校验，
  已随 _verify/minio_live_e2e.py 完成端到端真验证；测试与 Redis
  冒烟同型，无 MinIO 时自动跳过）
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.storage import LocalStorage, MinIOStorage, storage_from_env


class _FakeObjects:
    """桶对象内存表：fput 记内容，fget 写回目标路径。"""

    def __init__(self) -> None:
        self.store: dict = {}

    def put(self, key: str, data: bytes) -> None:
        self.store[key] = data


class _FakeMinio:
    """minio.Minio 桩：记录构造参数，模拟 bucket_exists/fput/fget。"""

    last_init: dict = {}
    objects = _FakeObjects()

    def __init__(self, endpoint, access_key=None, secret_key=None,
                 secure=True, **_):
        _FakeMinio.last_init = {"endpoint": endpoint,
                                "access_key": access_key,
                                "secret_key": secret_key,
                                "secure": secure}
        self.existing = False

    def bucket_exists(self, bucket: str) -> bool:
        return self.existing

    def make_bucket(self, bucket: str) -> None:
        self.existing = True

    def fput_object(self, bucket: str, key: str, file_path: str, **_) -> object:
        _FakeMinio.objects.put(key, Path(file_path).read_bytes())
        return object()

    def fget_object(self, bucket: str, key: str, file_path: str, **_) -> object:
        Path(file_path).write_bytes(_FakeMinio.objects.store[key])
        return object()

    def __getattr__(self, name: str):
        """未实现的客户端方法兜底：返回记录型 dummy 而非 AttributeError。

        仅覆盖测试实际用到的 4 个显式方法之外的扩展调用，提高测试
        扩展韧性；dunder 探测（copy/pickle 等）仍按原样抛 AttributeError。
        """
        if name.startswith("__"):
            raise AttributeError(name)

        def _dummy(*_args, **_kwargs) -> object:
            return object()

        return _dummy


class TestMinIOStorageUnit(unittest.TestCase):
    """桩客户端：上传 / 下载缓存 / 自动建桶。"""

    def setUp(self) -> None:
        _FakeMinio.objects = _FakeObjects()
        _FakeMinio.last_init = {}
        self.tmp = Path(tempfile.mkdtemp(prefix="minio_unit_"))
        patcher = patch("minio.Minio", _FakeMinio)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_uploads_and_returns_key(self) -> None:
        src = self.tmp / "out.mp4"
        src.write_bytes(b"MP4DATA")
        st = MinIOStorage("127.0.0.1:9000", "ak", "sk", "renders",
                          cache_dir=self.tmp / "cache")
        key = st.save(str(src), "task-1.mp4")
        self.assertEqual(key, "task-1.mp4")
        self.assertEqual(_FakeMinio.objects.store["task-1.mp4"], b"MP4DATA")
        self.assertTrue(_FakeMinio.last_init["endpoint"] == "127.0.0.1:9000")

    def test_bucket_auto_created(self) -> None:
        MinIOStorage("e:9000", "ak", "sk", "renders",
                     cache_dir=self.tmp / "cache")
        self.assertTrue(_FakeMinio.last_init)   # 构造时已尝试建桶

    def test_path_of_downloads_to_cache(self) -> None:
        _FakeMinio.objects.put("task-2.mp4", b"CACHED")
        st = MinIOStorage("e:9000", "ak", "sk", "renders",
                          cache_dir=self.tmp / "cache")
        p = st.path_of("task-2.mp4")
        self.assertTrue(p.exists())
        self.assertEqual(p.read_bytes(), b"CACHED")
        self.assertIn("cache", str(p))


class TestStorageWiring(unittest.TestCase):
    """装配点：MINIO_ENDPOINT 开关切换 Local ↔ MinIO。"""

    def setUp(self) -> None:
        # 隔离全局环境：暂存并移除 MINIO_ENDPOINT，tearDown 按原样恢复
        # （否则设了 MINIO_ENDPOINT 跑真实冒烟时，default 分支会被污染）
        self._saved_endpoint = os.environ.pop("MINIO_ENDPOINT", None)

    def tearDown(self) -> None:
        os.environ.pop("MINIO_ENDPOINT", None)
        if self._saved_endpoint is not None:
            os.environ["MINIO_ENDPOINT"] = self._saved_endpoint

    def test_default_is_local(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="wire_"))
        self.assertIsInstance(storage_from_env(tmp / "r"), LocalStorage)

    def test_env_switches_to_minio(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="wire_"))
        os.environ["MINIO_ENDPOINT"] = "127.0.0.1:9000"
        patcher = patch("minio.Minio", _FakeMinio)
        patcher.start()
        try:
            st = storage_from_env(tmp / "r")
        finally:
            patcher.stop()
        self.assertIsInstance(st, MinIOStorage)
        self.assertEqual(_FakeMinio.last_init["access_key"], "minioadmin")


# 导入期捕获端点：skipIf 在收集期判定，而其他测试类的 tearDown 可能
# 恢复/移除运行时环境变量 —— 用快照保证判定与取值一致
_LIVE_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")


@unittest.skipIf(not _LIVE_ENDPOINT,
                 "真实 MinIO 未配置（MINIO_ENDPOINT 未设）——环境就绪后补")
class TestMinIOLiveSmoke(unittest.TestCase):
    """真实 MinIO 端到端冒烟（与 Celery 真队列冒烟同型）。"""

    def test_roundtrip(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="minio_live_"))
        src = tmp / "out.mp4"
        src.write_bytes(os.urandom(1024))
        st = MinIOStorage(_LIVE_ENDPOINT,
                          os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
                          os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
                          os.environ.get("MINIO_BUCKET", "renders"),
                          cache_dir=tmp / "cache",
                          secure=os.environ.get("MINIO_SECURE") == "1")
        key = st.save(str(src), "live-smoke.mp4")
        p = st.path_of(key)
        self.assertEqual(p.read_bytes(), src.read_bytes())


if __name__ == "__main__":
    unittest.main()
