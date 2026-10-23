"""Prometheus /metrics 端点测试（可观测性：指标暴露）.

覆盖 4 条：
1. GET /metrics → 200 + Prometheus 文本格式
2. 响应体含 4 个指标类型声明（counter/histogram/gauge/info）
3. app_info 携带 version="0.2.0"
4. API_KEY 开启时 /metrics 仍公开可达（抓取器无法携带凭证）
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from server.app import create_app
from server.tasks import celery_app


def _make_app(tmp_prefix: str):
    tmp = Path(tempfile.mkdtemp(prefix=tmp_prefix))
    return create_app(db_path=tmp / "tasks.db",
                      storage_root=tmp / "renders",
                      tmp_root=tmp / "tmp")


class TestMetricsApi(unittest.TestCase):
    """/metrics：Prometheus 文本暴露 + 指标定义 + 公开性。"""

    @classmethod
    def setUpClass(cls) -> None:
        celery_app.conf.task_always_eager = True
        cls.client = TestClient(_make_app("metrics_"))
        # 先打几条请求，让计数器产生真实样本
        cls.client.get("/health")
        cls.client.get("/manifest")

    def test_metrics_returns_200(self) -> None:
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/plain", resp.headers["content-type"])

    def test_metrics_contains_expected_types(self) -> None:
        body = self.client.get("/metrics").text
        # 4 个指标家族的样本必须在位（计数器在 setUpClass 已打点）
        self.assertIn("http_requests_total", body)
        self.assertIn("http_request_duration_seconds", body)
        self.assertIn("http_requests_in_progress", body)
        self.assertIn("app_info", body)
        # /health 已打点 → counter 样本非零
        self.assertRegex(body,
                         r'http_requests_total\{[^}]*path="/health"[^}]*\}\s+[1-9]')

    def test_metrics_contains_app_version(self) -> None:
        body = self.client.get("/metrics").text
        self.assertIn('version="0.2.0"', body)

    def test_metrics_not_intercepted_by_auth(self) -> None:
        with mock.patch.dict(os.environ, {"API_KEY": "secret-key"}):
            client = TestClient(_make_app("metrics_auth_"))
            self.assertEqual(client.get("/metrics").status_code, 200)
            # 对照组：业务路由被拦截
            self.assertEqual(
                client.get("/render/nonexistent").status_code, 401)


if __name__ == "__main__":
    unittest.main()
