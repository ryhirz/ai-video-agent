"""GET /ready 深度就绪探针测试（可观测性：readiness）.

覆盖 4 条：
1. 全依赖就绪 → 200 + status=ready（需本地 Redis，skipif 保护）
2. Redis 故障（mock 连接异常）→ 503 + status=not_ready + redis error
3. API_KEY 开启时 /ready 仍公开（探针无法携带凭证）
4. MinIO 配置了不可达端点 → 503 + minio error（可选依赖配置后即必检）
"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from minio import Minio

from server.app import create_app
from server.storage import MinIOStorage
from server.tasks import celery_app, get_runtime


def _redis_alive() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6379), timeout=0.5):
            return True
    except OSError:
        return False


def _make_app(tmp_prefix: str):
    tmp = Path(tempfile.mkdtemp(prefix=tmp_prefix))
    return create_app(db_path=tmp / "tasks.db",
                      storage_root=tmp / "renders",
                      tmp_root=tmp / "tmp")


class _DeadMinioStorage(MinIOStorage):
    """绕过父类构造（bucket_exists 会真实连网）的不可达 MinIO 桩."""

    def __init__(self) -> None:
        self.client = Minio("127.0.0.1:59999",   # 不可达端口
                            access_key="a", secret_key="b", secure=False)
        self.bucket = "renders"


class TestReadyApi(unittest.TestCase):
    """/ready：DB/Redis/MinIO 深度检查 + 公开性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(_make_app("ready_"))

    @unittest.skipIf(not _redis_alive(), "本地 Redis 未启动")
    def test_ready_all_ok(self) -> None:
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["dependencies"]["db"], "ok")
        self.assertEqual(body["dependencies"]["redis"], "ok")
        self.assertEqual(body["dependencies"]["minio"], "skipped")

    def test_ready_redis_down_returns_503(self) -> None:
        with mock.patch.object(celery_app, "connection",
                               side_effect=RuntimeError("broker boom")):
            resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "not_ready")
        self.assertIn("boom", body["dependencies"]["redis"])
        self.assertEqual(body["dependencies"]["db"], "ok")   # DB 不受影响

    @unittest.skipIf(not _redis_alive(), "本地 Redis 未启动")
    def test_ready_public_with_auth(self) -> None:
        with mock.patch.dict(os.environ, {"API_KEY": "secret-key"}):
            client = TestClient(_make_app("ready_auth_"))
            resp = client.get("/ready")
            self.assertEqual(resp.status_code, 200)          # 非 401
            # 对照组：业务路由被拦截
            self.assertEqual(
                client.get("/render/nonexistent").status_code, 401)

    def test_ready_minio_unreachable_returns_503(self) -> None:
        rt = get_runtime()
        original = rt.storage
        rt.storage = _DeadMinioStorage()
        try:
            resp = self.client.get("/ready")
        finally:
            rt.storage = original
        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "not_ready")
        self.assertIn("error:", body["dependencies"]["minio"])


if __name__ == "__main__":
    unittest.main()
