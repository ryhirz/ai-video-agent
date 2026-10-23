"""全栈层 API 端到端（eager 模式，ADR-1）。

第二步起执行链路为 Celery：本文件设 task_always_eager=True，任务在
API 进程内同步执行，覆盖「任务体状态推进 + 路由 + 存储全链路」；
broker/ack 真实路径由 tests/test_celery_smoke.py 兜底（无 Redis 跳过）。

eager 模式下 POST /render 同步完成渲染，无法构造「排队中即取消」的竞态，
取消语义改为三条用例覆盖：
  - 终态幂等：SUCCESS 后 cancel 不改库、原状态返回
  - DB 查询态检查点：任务体启动前 state=CANCELLED → 检查点 1 停止
  - 超时入口检查：created_at 距今超过 time_limit → 直接落 TIMEOUT
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from ai_video_agent.render import resolve_ffmpeg
from server.models import RenderTask
from server.state import TaskState
from server.tasks import celery_app, render_video

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
        celery_app.conf.task_always_eager = True   # ADR-1：eager 为主
        cls.tmp = Path(tempfile.mkdtemp(prefix="srv_"))
        cls.src = cls.tmp / "src.mp4"
        _gen_source(cls.src)
        cls.client = TestClient(create_app_fixture())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _wait(self, task_id: str, timeout: float = 30.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = self.client.get(f"/render/{task_id}").json()
            if body["state"] in TERMINAL:
                return body
            time.sleep(0.2)
        self.fail(f"任务 {task_id} 未在 {timeout}s 内到达终态")

    # ---- 路由层 ----

    def test_manifest_endpoint(self) -> None:
        body = self.client.get("/manifest").json()
        self.assertIn("operations", body)
        self.assertIn("speed", body["operations"])

    def test_full_render_flow(self) -> None:
        """提交 → 轮询至 SUCCESS → 下载成片非空（eager 下同步完成）。"""
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

    def test_404_for_unknown_task(self) -> None:
        self.assertEqual(self.client.get("/render/nope").status_code, 404)

    # ---- 取消语义（ADR-2：DB 查询态） ----

    def test_cancel_terminal_idempotent(self) -> None:
        """终态任务 cancel 幂等：返回原状态，不改库，不影响下载。"""
        r = self.client.post("/render", json={"timeline": _timeline(str(self.src))})
        task_id = r.json()["task_id"]
        body = self._wait(task_id)
        self.assertEqual(body["state"], "success")

        c = self.client.post(f"/render/{task_id}/cancel")
        self.assertEqual(c.status_code, 200)
        self.assertEqual(c.json()["state"], "success")
        # 下载不受影响
        self.assertEqual(
            self.client.get(f"/render/{task_id}/download").status_code, 200)

    def test_cancel_checkpoint_before_render(self) -> None:
        """DB 查询态检查点：任务体启动时 state=CANCELLED → 检查点 1 停止。

        直接操作 DB 构造「已取消但未执行」的任务，绕过 eager 同步竞态；
        任务体应保持 CANCELLED 不渲染、临时目录不产生。
        """
        db = self.client.app.state.db
        tid = "cancel-checkpoint-0001"
        db.create(tid, "local", _timeline(str(self.src)), {}, None)
        db.update(tid, state=TaskState.CANCELLED,
                  finished_at="2026-01-01T00:00:00+00:00")

        render_video(tid, _timeline(str(self.src)))   # 同步执行任务体

        row = db.get(tid)
        self.assertEqual(row["state"], "cancelled")   # 检查点 1 停止，未渲染
        self.assertEqual(row["started_at"], None)     # 未进入 RUNNING 推进
        self.assertFalse((Path(self.tmp) / "tmp" / tid).exists())

    def test_midrender_cancel(self) -> None:
        """渲染中段取消：RUNNING 后置 CANCELLED → ffmpeg 被两步杀 →
        终态保持 CANCELLED（不被 FAILED 覆盖）。任务体在后台线程执行。"""
        import threading
        db = self.client.app.state.db
        long_src = self.tmp / "long60.mp4"
        _gen_source(long_src, 60.0)
        tid = "midrender-cancel-0001"
        db.create(tid, "local", _timeline(str(long_src), 60.0), {}, None)

        th = threading.Thread(target=render_video,
                              args=(tid, _timeline(str(long_src), 60.0)))
        th.start()
        deadline = time.time() + 30
        while time.time() < deadline and db.get(tid)["state"] != "running":
            time.sleep(0.05)
        self.assertEqual(db.get(tid)["state"], "running")
        time.sleep(1.5)                       # 让 ffmpeg 真正跑起来

        c = self.client.post(f"/render/{tid}/cancel").json()
        self.assertEqual(c["state"], "cancelled")

        th.join(timeout=60)
        self.assertFalse(th.is_alive(), "任务体未在 60s 内退出")
        row = db.get(tid)
        self.assertEqual(row["state"], "cancelled")   # 中段取消 → 保持 CANCELLED
        self.assertIsNotNone(row["started_at"])
        self.assertIsNotNone(row["finished_at"])
        self.assertEqual(row["error_msg"], None)
        # 已取消任务不可下载
        self.assertEqual(
            self.client.get(f"/render/{tid}/download").status_code, 400)

    def test_timeout_entry_check(self) -> None:
        """超时衔接（ADR-3）：created_at 距今超过 time_limit → 直接落 TIMEOUT。"""
        db = self.client.app.state.db
        tid = "timeout-entry-check-001"
        db.create(tid, "local", _timeline(str(self.src)), {}, None)
        stale = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()
        with db._session.begin() as session:          # 测试直改 created_at
            session.query(RenderTask).filter_by(id=tid).update(
                {"created_at": stale})

        render_video(tid, _timeline(str(self.src)))

        row = db.get(tid)
        self.assertEqual(row["state"], "timeout")
        self.assertIn("entry check", row["error_msg"])
        self.assertEqual(row["started_at"], None)


def create_app_fixture():
    from server.app import create_app
    return create_app(db_path=TestServerApi.tmp / "tasks.db",
                      storage_root=TestServerApi.tmp / "renders",
                      tmp_root=TestServerApi.tmp / "tmp")


class TestAuthApi(unittest.TestCase):
    """可选认证（API_KEY 环境变量）：开 → 401/放行双态；关 → 行为不变。"""

    @classmethod
    def setUpClass(cls) -> None:
        from unittest import mock
        celery_app.conf.task_always_eager = True
        cls.tmp = Path(tempfile.mkdtemp(prefix="auth_"))
        cls.src = cls.tmp / "src.mp4"
        _gen_source(cls.src)
        from server.app import create_app

        def make(key):
            env = {"API_KEY": key} if key else {}
            with mock.patch.dict(os.environ, env, clear=False):
                # Key 在 create_app 内读取，上下文退出后配置已固化
                return TestClient(create_app(
                    db_path=cls.tmp / f"t_{key or 'off'}.db",
                    storage_root=cls.tmp / f"r_{key or 'off'}",
                    tmp_root=cls.tmp / f"tmp_{key or 'off'}"))

        cls.client_off = make(None)       # 未启用认证
        cls.client_on = make("sekrit")    # 启用认证

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_off.close()
        cls.client_on.close()
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_auth_disabled_by_default(self) -> None:
        """未设 API_KEY：不带任何凭证 → 202（历史行为零变化）。"""
        r = self.client_off.post("/render",
                                 json={"timeline": _timeline(str(self.src))})
        self.assertEqual(r.status_code, 202)

    def test_auth_enforced_when_set(self) -> None:
        """设 API_KEY：无头/错 Key → 401；对头放行；/manifest 保持公开。"""
        tl = {"timeline": _timeline(str(self.src))}
        self.assertEqual(
            self.client_on.post("/render", json=tl).status_code, 401)
        self.assertEqual(self.client_on.post(
            "/render", json=tl,
            headers={"X-API-Key": "wrong"}).status_code, 401)
        ok = self.client_on.post("/render", json=tl,
                                 headers={"X-API-Key": "sekrit"})
        self.assertEqual(ok.status_code, 202)

        # 查询参数等价放行（video.src / <a download> 无法携带自定义头）
        tid = ok.json()["task_id"]
        q = self.client_on.get(f"/render/{tid}?api_key=sekrit")
        self.assertEqual(q.status_code, 200)
        # 能力清单与静态页保持公开
        self.assertEqual(self.client_on.get("/manifest").status_code, 200)
        self.assertEqual(
            self.client_on.get("/render/nope?api_key=sekrit").status_code, 404)

    def test_download_with_query_key(self) -> None:
        """下载链路：头放行提交，查询参数放行下载（?api_key=）。"""
        ok = self.client_on.post("/render", json={"timeline": _timeline(str(self.src))},
                                 headers={"X-API-Key": "sekrit"})
        tid = ok.json()["task_id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            body = self.client_on.get(
                f"/render/{tid}?api_key=sekrit").json()
            if body["state"] in TERMINAL:
                break
            time.sleep(0.2)
        self.assertEqual(body["state"], "success", body)
        dl = self.client_on.get(f"/render/{tid}/download?api_key=sekrit")
        self.assertEqual(dl.status_code, 200)
        self.assertGreater(len(dl.content), 0)
        # 无凭证下载 → 401
        self.assertEqual(
            self.client_on.get(f"/render/{tid}/download").status_code, 401)


class TestHealthApi(unittest.TestCase):
    """GET /health：公开探针端点（容器编排 liveness/readiness 前置）。"""

    @classmethod
    def setUpClass(cls) -> None:
        from unittest import mock
        celery_app.conf.task_always_eager = True
        cls.tmp = Path(tempfile.mkdtemp(prefix="health_"))
        from server.app import create_app
        # 认证开启的实例：验证 /health 不受 API_KEY 影响（探针无凭证）
        with mock.patch.dict(os.environ, {"API_KEY": "hkey"}, clear=False):
            cls.client = TestClient(create_app(
                db_path=cls.tmp / "tasks.db",
                storage_root=cls.tmp / "renders",
                tmp_root=cls.tmp / "tmp"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_health_ok(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertIn("version", r.json())

    def test_health_public_even_with_api_key(self) -> None:
        """API_KEY 开启时 /health 仍公开；业务路由照常 401（对照）。"""
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(
            self.client.post("/render", json={"timeline": {}}).status_code,
            401)


class TestRequestIdApi(unittest.TestCase):
    """X-Request-ID 贯穿：透传 / 自动生成 + 响应头回写（结构化日志闭环）。"""

    @classmethod
    def setUpClass(cls) -> None:
        celery_app.conf.task_always_eager = True
        cls.tmp = Path(tempfile.mkdtemp(prefix="reqid_"))
        from server.app import create_app
        cls.client = TestClient(create_app(
            db_path=cls.tmp / "tasks.db",
            storage_root=cls.tmp / "renders",
            tmp_root=cls.tmp / "tmp"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_header_passthrough(self) -> None:
        """客户端/网关自带 X-Request-ID → 原样回写响应头（链路追踪）。"""
        r = self.client.get("/health", headers={"X-Request-ID": "test-123"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Request-ID"], "test-123")

    def test_auto_generated_uuid(self) -> None:
        """不带 ID → 自动生成 UUID v4 并回写响应头。"""
        r = self.client.get("/health")
        parsed = uuid.UUID(r.headers["X-Request-ID"])   # 非 UUID 会抛 ValueError
        self.assertEqual(parsed.version, 4)


if __name__ == "__main__":
    unittest.main()
