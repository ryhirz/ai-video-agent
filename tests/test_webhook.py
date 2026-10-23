"""Webhook 终态回调测试（第四步，契约 docs/全栈层/04-API §7）。

- 单元：本地 HTTP 接收器验证载荷形状 / 重试耗尽 / 非 2xx 后成功
- 集成：eager 模式跑通真实渲染任务 → 回调体与 DB 终态一致
  （重试间隔 patch 为 0，生产语义 10s/60s/300s 不受影响）
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import server.webhooks as wh
from ai_video_agent.render import resolve_ffmpeg
from server.models import TaskDB
from server.storage import LocalStorage
from server.tasks import celery_app, configure_runtime, render_video

FF = resolve_ffmpeg()


def _gen_source(path: Path, dur: float = 1.0) -> None:
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
        "resolution": {"w": 320, "h": 240}, "duration": dur,
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


class _Receiver:
    """可编程状态码序列的本地 HTTP 接收器（记录全部请求体）。"""

    def __init__(self, statuses: tuple = (200,)):
        self.statuses = list(statuses)
        self.bodies: list = []
        recv = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", 0))
                recv.bodies.append(json.loads(self.rfile.read(n)))
                code = recv.statuses.pop(0) if recv.statuses else 200
                self.send_response(code)
                self.end_headers()

            def log_message(self, *args) -> None:   # 静默测试输出
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/callback"
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class TestWebhookUnit(unittest.TestCase):
    """fire_webhook 单元：载荷形状 / 重试 / 尽力投递语义。"""

    def tearDown(self) -> None:
        wh.RETRIES = (10.0, 60.0, 300.0)      # 恢复生产语义

    def test_retry_until_2xx(self) -> None:
        """前两次 5xx、第三次 2xx → 重试后成功，共 3 次请求。"""
        r = _Receiver(statuses=(500, 503, 200))
        try:
            wh.RETRIES = (0.0, 0.0, 0.0)      # 测试加速，语义不变
            ok = wh.fire_webhook(r.url, {"event": "task.finished"})
        finally:
            r.close()
        self.assertTrue(ok)
        self.assertEqual(len(r.bodies), 3)

    def test_retry_exhausted_returns_false(self) -> None:
        """全部尝试非 2xx → False，不抛异常（尽力投递）。"""
        r = _Receiver(statuses=(500, 500, 500, 500))
        try:
            wh.RETRIES = (0.0, 0.0, 0.0)
            ok = wh.fire_webhook(r.url, {"event": "task.finished"})
        finally:
            r.close()
        self.assertFalse(ok)
        self.assertEqual(len(r.bodies), 4)    # 1 次首发 + 3 次重试

    def test_payload_contract_shape(self) -> None:
        """载荷形状与冻结契约逐字段一致。"""
        r = _Receiver()
        try:
            wh.fire_webhook(r.url, wh.build_payload({
                "id": "tid-1", "state": "success", "progress": 1.0,
                "result_url": "renders/tid-1.mp4", "error_msg": None,
                "finished_at": "2026-10-23T06:00:00+00:00",
                "webhook_url": r.url}))
        finally:
            r.close()
        self.assertEqual(r.bodies, [{
            "event": "task.finished", "task_id": "tid-1",
            "state": "success", "progress": 1.0,
            "result_url": "renders/tid-1.mp4", "error_msg": None,
            "finished_at": "2026-10-23T06:00:00+00:00",
        }])


class TestWebhookIntegration(unittest.TestCase):
    """eager 端到端：真实渲染任务终态 → 回调体与 DB 终态一致。"""

    @classmethod
    def setUpClass(cls) -> None:
        celery_app.conf.task_always_eager = True
        wh.RETRIES = (0.0, 0.0, 0.0)
        cls.tmp = Path(tempfile.mkdtemp(prefix="whk_"))
        cls.src = cls.tmp / "src.mp4"
        _gen_source(cls.src)
        cls.db = TaskDB(str(cls.tmp / "tasks.db"))
        configure_runtime(cls.db, LocalStorage(cls.tmp / "renders"),
                          cls.tmp / "tmp")

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        wh.RETRIES = (10.0, 60.0, 300.0)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_success_callback_fires_with_result(self) -> None:
        r = _Receiver()
        try:
            tid = "webhook-success-0001"
            self.db.create(tid, "local", _timeline(str(self.src)), {}, r.url)
            render_video(tid, _timeline(str(self.src)))

            row = self.db.get(tid)
            self.assertEqual(row["state"], "success")
            time.sleep(0.2)                    # 回调在返回前同步完成，兜底
            self.assertEqual(len(r.bodies), 1)
            body = r.bodies[0]
            self.assertEqual(body["event"], "task.finished")
            self.assertEqual(body["task_id"], tid)
            self.assertEqual(body["state"], "success")
            self.assertEqual(body["progress"], 1.0)
            self.assertEqual(body["result_url"], f"{tid}.mp4")
            self.assertIsNone(body["error_msg"])
            self.assertIsNotNone(body["finished_at"])
        finally:
            r.close()

    def test_cancelled_callback_fires(self) -> None:
        """出队检查点停止（已取消任务）→ 回调 state=cancelled。"""
        r = _Receiver()
        try:
            tid = "webhook-cancel-0001"
            self.db.create(tid, "local", _timeline(str(self.src)), {}, r.url)
            self.db.update(tid, state="cancelled",
                           finished_at="2026-10-23T06:00:00+00:00")
            render_video(tid, _timeline(str(self.src)))

            time.sleep(0.2)
            self.assertEqual(len(r.bodies), 1)
            self.assertEqual(r.bodies[0]["state"], "cancelled")
            self.assertEqual(r.bodies[0]["task_id"], tid)
        finally:
            r.close()

    def test_no_webhook_no_request(self) -> None:
        """未提供 webhook_url → 不发任何请求，任务照常 SUCCESS。"""
        tid = "webhook-none-0001"
        self.db.create(tid, "local", _timeline(str(self.src)), {}, None)
        render_video(tid, _timeline(str(self.src)))
        self.assertEqual(self.db.get(tid)["state"], "success")


if __name__ == "__main__":
    unittest.main()
