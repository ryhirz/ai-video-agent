"""真队列冒烟测试（ADR-1 的兜底面，本机无 Redis 自动跳过）。

与 eager 模式的分工：eager 覆盖任务体逻辑；本文件用**真 Redis broker +
真 Celery worker 子进程**验证 eager 盲区 —— 消息序列化、队列投递、
acks_late 消费路径、独立进程的资源惰性初始化（环境变量注入）。

设计：不起 uvicorn，直接 TaskDB 建任务 → render_video.delay（eager 关闭）
→ 轮询 DB 至终态。worker 以 --pool=solo 启动（Windows 兼容），任务体经
RENDER_DB_PATH / RENDER_STORAGE_ROOT / RENDER_TMP_ROOT 环境变量指向本
测试的临时目录。
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from server.models import TaskDB, new_task_id
from server.tasks import celery_app, render_video

REPO_ROOT = Path(__file__).resolve().parents[1]
REDIS_HOST, REDIS_PORT = "127.0.0.1", 6379
TERMINAL = {"success", "failed", "cancelled", "timeout"}
FF = str(REPO_ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe")


def _redis_available() -> bool:
    try:
        with socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=0.5):
            return True
    except OSError:
        return False


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


@unittest.skipUnless(_redis_available(),
                     f"Redis {REDIS_HOST}:{REDIS_PORT} 不可达，跳过真队列冒烟")
class TestCeleryRealQueueSmoke(unittest.TestCase):
    """真 Redis broker + worker 子进程的端到端冒烟。"""

    worker: subprocess.Popen | None = None

    @classmethod
    def setUpClass(cls) -> None:
        celery_app.conf.task_always_eager = False   # 真队列：强制关闭 eager
        cls.tmp = Path(tempfile.mkdtemp(prefix="celery_smoke_"))
        cls.src = cls.tmp / "src.mp4"
        _gen_source(cls.src)
        # worker 子进程资源注入：DB / 存储 / 临时目录全部指向本测试 tmp
        cls.env = dict(os.environ,
                       RENDER_DB_PATH=str(cls.tmp / "tasks.db"),
                       RENDER_STORAGE_ROOT=str(cls.tmp / "renders"),
                       RENDER_TMP_ROOT=str(cls.tmp / "tmp"),
                       CELERY_BROKER_URL=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
                       PYTHONPATH=str(REPO_ROOT))
        cls.worker = subprocess.Popen(
            [sys.executable, "-m", "celery", "-A", "server.tasks",
             "worker", "-Q", "render", "--pool=solo", "--concurrency=1",
             "--without-gossip", "--without-mingle", "--without-heartbeat",
             "-l", "warning"],
            cwd=str(REPO_ROOT), env=cls.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.worker:
            cls.worker.terminate()
            try:
                cls.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.worker.kill()
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_render_via_real_broker(self) -> None:
        """建任务 → delay → 真 worker 消费 → SUCCESS + 成片落库。"""
        db = TaskDB(str(self.tmp / "tasks.db"))
        tid = new_task_id()
        db.create(tid, "local", _timeline(str(self.src)), {}, None)

        render_video.delay(tid, _timeline(str(self.src)))

        deadline, state = time.time() + 90, None
        while time.time() < deadline:
            row = db.get(tid)
            state = row["state"]
            if state in TERMINAL:
                break
            time.sleep(0.5)
        self.assertEqual(state, "success", f"task {tid} 终态: {state}")
        self.assertEqual(db.get(tid)["progress"], 1.0)
        out = self.tmp / "renders" / f"{tid}.mp4"
        self.assertTrue(out.exists() and out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
