"""全栈层第一步端到端：POST /render → 轮询 → download。

直接验证架构第一步的验收目标——「远程调用 compile() 能跑通」：
compile/render 纯函数被包进 HTTP + 状态机 + 本地存储，编译器零改动。
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from ai_video_agent.render import resolve_ffmpeg
from server.app import create_app

FF = resolve_ffmpeg()
W, H = 320, 240
TERMINAL = {"success", "failed", "cancelled", "timeout"}


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


class TestServerApi(unittest.TestCase):
    """单类共享一个 app 实例 + 临时目录；各测试独立提交任务。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="srv_"))
        cls.src = cls.tmp / "src.mp4"
        _gen_source(cls.src)
        cls.client = TestClient(create_app(
            db_path=cls.tmp / "tasks.db",
            storage_root=cls.tmp / "renders",
            tmp_root=cls.tmp / "tmp"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        shutil = __import__("shutil")
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _wait(self, task_id: str, timeout: float = 30.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = self.client.get(f"/render/{task_id}").json()
            if body["state"] in TERMINAL:
                return body
            time.sleep(0.2)
        self.fail(f"任务 {task_id} 未在 {timeout}s 内到达终态")

    def test_manifest_endpoint(self) -> None:
        body = self.client.get("/manifest").json()
        self.assertIn("operations", body)
        self.assertIn("speed", body["operations"])

    def test_full_render_flow(self) -> None:
        """提交 → 轮询至 SUCCESS → 下载成片非空。"""
        r = self.client.post("/render", json={"timeline": _timeline(str(self.src))})
        self.assertEqual(r.status_code, 202)
        task_id = r.json()["task_id"]
        self.assertEqual(r.json()["state"], "pending")

        body = self._wait(task_id)
        self.assertEqual(body["state"], "success", body)
        self.assertEqual(body["progress"], 1.0)

        dl = self.client.get(f"/render/{task_id}/download")
        self.assertEqual(dl.status_code, 200)
        self.assertGreater(len(dl.content), 0)

    def test_compile_error_becomes_failed(self) -> None:
        """非法 transform → 编译期失败，error_msg 可读。"""
        tl = _timeline(str(self.src))
        tl["clips"][0]["transform"] = {"x": -5}      # 负偏移：编译期拒绝
        r = self.client.post("/render", json={"timeline": tl})
        task_id = r.json()["task_id"]
        body = self._wait(task_id)
        self.assertEqual(body["state"], "failed")
        self.assertIn("Compile error", body["error_msg"])

    def test_cancel_pending_task(self) -> None:
        # 用 5s 素材拉长渲染窗口，避免「cancel 到达前已 SUCCESS」的竞态
        long_src = self.tmp / "long.mp4"
        _gen_source(long_src, dur=5.0)
        r = self.client.post("/render", json={"timeline": _timeline(str(long_src), 5.0)})
        task_id = r.json()["task_id"]
        c = self.client.post(f"/render/{task_id}/cancel").json()
        self.assertEqual(c["state"], "cancelled")
        body = self._wait(task_id)
        self.assertEqual(body["state"], "cancelled")
        # 已取消的任务不可下载
        self.assertEqual(self.client.get(f"/render/{task_id}/download").status_code, 400)

    def test_404_for_unknown_task(self) -> None:
        self.assertEqual(self.client.get("/render/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
