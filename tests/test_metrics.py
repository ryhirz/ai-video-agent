"""Prometheus /metrics 端点测试（可观测性：指标暴露）.

覆盖 4 条：
1. GET /metrics → 200 + Prometheus 文本格式
2. 响应体含 4 个指标类型声明（counter/histogram/gauge/info）
3. app_info 携带 version="0.2.0"
4. API_KEY 开启时 /metrics 仍公开可达（抓取器无法携带凭证）
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from ai_video_agent.render import resolve_ffmpeg
from server.app import create_app
from server.tasks import celery_app

FF = resolve_ffmpeg()
W, H = 320, 240


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


def _gen_source(path: Path, dur: float = 1.0) -> None:
    """生成 1s 真实源素材（视频 + 音频），与 test_server_api 同法。"""
    subprocess.run(
        [FF, "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={dur}",
         "-f", "lavfi", "-i", f"sine=f=440:d={dur}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)


def _timeline(src: str, dur: float = 1.0) -> dict:
    return {
        "schemaVersion": "1.0.0", "fps": 30,
        "resolution": {"w": W, "h": H}, "duration": dur,
        "tracks": [{"id": "v_main", "type": "video"},
                   {"id": "a_main", "type": "audio"}],
        "clips": [
            {"id": "src_video", "track": "v_main", "sourceRef": src,
             "in": 0.0, "out": dur, "sourceIn": 0.0, "sourceOut": dur},
            {"id": "src_audio", "track": "a_main", "sourceRef": src,
             "in": 0.0, "out": dur, "sourceIn": 0.0, "sourceOut": dur},
        ],
        "subtitles": [], "transitions": [], "texts": [],
    }


class TestWorkerMetrics(unittest.TestCase):
    """worker 侧渲染指标：eager 真渲染 → 终态计数/耗时/队列深度快照。"""

    @classmethod
    def setUpClass(cls) -> None:
        celery_app.conf.task_always_eager = True
        cls.tmp = Path(tempfile.mkdtemp(prefix="wmetrics_"))
        cls.src = cls.tmp / "src.mp4"
        _gen_source(cls.src)
        cls.client = TestClient(_make_app("wmetrics_"))

    def test_render_outcome_and_duration_counted(self) -> None:
        resp = self.client.post(
            "/render", json={"user_id": "u-wm",
                             "timeline": _timeline(str(self.src))})
        self.assertEqual(resp.status_code, 202)
        task_id = resp.json()["task_id"]
        # eager：POST 返回即终态
        state = self.client.get(f"/render/{task_id}").json()["state"]
        self.assertEqual(state, "success")

        body = self.client.get("/metrics").text
        self.assertIn('render_tasks_total{outcome="success"}', body)
        # 成功任务计入耗时直方图（count/sum 样本在位）
        self.assertIn("render_duration_seconds_count", body)
        self.assertIn("render_duration_seconds_sum", body)
        # 活跃数已回落（任务结束后 dec）
        self.assertIn("render_active 0.0", body)

    def test_queue_depth_snapshot_from_db(self) -> None:
        body = self.client.get("/metrics").text
        # DB 权威队列深度 Gauge 在位（本 app 无 pending 任务 → 0）
        self.assertIn("render_queue_depth 0.0", body)
        self.assertIn("render_running_depth 0.0", body)


class TestMultiprocessMetrics(unittest.TestCase):
    """multiprocess 聚合模式：子进程写共享目录 → 聚合端可见全部进程.

    prometheus_client 的 multiprocess 行为在指标创建（模块导入）时
    由 PROMETHEUS_MULTIPROC_DIR 决定 → 用子进程真实触发（env 先于
    import 设置），是最接近生产双进程的验证方式。
    """

    def test_aggregated_exposition_from_shared_dir(self) -> None:
        import subprocess
        import sys
        mp_dir = Path(tempfile.mkdtemp(prefix="mp_metrics_"))
        root = Path(__file__).resolve().parents[1]
        code = (
            "import server.metrics as m; "
            "m.RENDER_TOTAL.labels(outcome='success').inc(); "
            "m.RENDER_ACTIVE.inc(); "
            "body, _ = m.metrics_response(); "
            "print(body.decode())"
        )
        env = os.environ.copy()
        env["PROMETHEUS_MULTIPROC_DIR"] = str(mp_dir)
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(root),
                              env=env, capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = proc.stdout
        # worker 侧计数经共享目录聚合可见
        self.assertIn('render_tasks_total{outcome="success"}', body)
        self.assertIn("render_active", body)
        # Info（私有 registry 拼接）在 multiprocess 模式下不缺失
        self.assertIn('version="0.2.0"', body)
        # 单进程分支回归：聚合路径未破坏默认模式（app_info 恒在位）
        self.assertIn("app_info", body)


if __name__ == "__main__":
    unittest.main()
