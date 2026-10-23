"""MinIO 真实实例端到端冒烟（前置：MinIO 服务运行于 127.0.0.1:9000）。

链路：POST /render（eager 真渲染）→ MinIOStorage 上传成片 →
GET /render/{id}/download 经 path_of 惰性拉回 → 校验字节与 MP4 魔数 →
本地 renders 目录应为空（证明走的是对象存储而非本地 FS）→
minio SDK stat_object 确认对象真实在桶 → 清理对象。

运行：.venv/Scripts/python.exe _verify/minio_live_e2e.py
退出码 0 = 全部通过；任何一步失败即非零退出。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENDPOINT = os.environ.get("MINIO_ENDPOINT", "127.0.0.1:9000")
BUCKET = os.environ.get("MINIO_BUCKET", "renders")
os.environ["MINIO_ENDPOINT"] = ENDPOINT
os.environ.setdefault("MINIO_BUCKET", BUCKET)

from fastapi.testclient import TestClient          # noqa: E402
from minio import Minio                            # noqa: E402
from ai_video_agent.render import resolve_ffmpeg   # noqa: E402
from server.app import create_app                  # noqa: E402
from server.tasks import celery_app                # noqa: E402

FF = resolve_ffmpeg()
tmp = Path(tempfile.mkdtemp(prefix="minio_e2e_"))

# ---- 1. 生成 1s 真实源素材（视频 + 音频） ----
src = tmp / "src.mp4"
subprocess.run(
    [FF, "-v", "error", "-y",
     "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1.0",
     "-f", "lavfi", "-i", "sine=f=440:d=1.0",
     "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-shortest", str(src)],
    check=True, capture_output=True)
print(f"[1] 源素材生成: {src.stat().st_size} bytes")

# ---- 2. eager 模式装配 app（存储 = MinIOStorage） ----
celery_app.conf.task_always_eager = True
app = create_app(db_path=tmp / "tasks.db",
                 storage_root=tmp / "renders",
                 tmp_root=tmp / "tmp")
client = TestClient(app)

timeline = {
    "schemaVersion": "1.0.0", "fps": 30,
    "resolution": {"w": 320, "h": 240}, "duration": 1.0,
    "tracks": [{"id": "v_main", "type": "video"},
               {"id": "a_main", "type": "audio"}],
    "clips": [
        {"id": "cv", "track": "v_main", "sourceRef": str(src),
         "in": 0.0, "out": 1.0, "sourceIn": 0.0, "sourceOut": 1.0},
        {"id": "ca", "track": "a_main", "sourceRef": str(src),
         "in": 0.0, "out": 1.0, "sourceIn": 0.0, "sourceOut": 1.0},
    ],
    "subtitles": [], "transitions": [], "texts": [],
}

# ---- 3. 提交渲染并等待终态 ----
r = client.post("/render", json={"timeline": timeline, "options": {}})
assert r.status_code == 202, f"POST /render -> {r.status_code}: {r.text}"
tid = r.json()["task_id"]
print(f"[2] 任务提交: {tid}")

deadline = time.time() + 60
state = None
while time.time() < deadline:
    st = client.get(f"/render/{tid}").json()
    state = st["state"]
    if state in ("success", "failed", "cancelled", "timeout"):
        break
    time.sleep(0.5)
assert state == "success", f"终态 {state}，期望 success: {st}"
key = st["result_url"]
print(f"[3] 渲染 success，progress={st['progress']}，result_url={key}")

# ---- 4. 本地 renders 目录应为空（证明走 MinIO 而非本地 FS） ----
leftovers = [p.name for p in (tmp / "renders").rglob("*") if p.is_file()]
assert not leftovers, f"本地 renders 目录有残留文件: {leftovers}（未走对象存储）"
print("[4] 本地 renders 目录为空 —— 产物确实进了 MinIO")

# ---- 5. minio SDK 直查桶：对象真实存在且大小一致 ----
mc = Minio(ENDPOINT, access_key="minioadmin", secret_key="minioadmin",
           secure=False)
assert mc.bucket_exists(BUCKET), f"桶 {BUCKET} 不存在"
obj = mc.stat_object(BUCKET, key)
print(f"[5] 桶内对象确认: {BUCKET}/{key} size={obj.size}")

# ---- 6. /download 路由经惰性缓存拉回，字节与 MP4 魔数校验 ----
r = client.get(f"/render/{tid}/download")
assert r.status_code == 200, f"download -> {r.status_code}: {r.text}"
data = r.content
assert len(data) == obj.size, f"下载 {len(data)} B != 桶内 {obj.size} B"
assert data[4:8] == b"ftyp", f"非 MP4 魔数: {data[:16]!r}"
print(f"[6] /download 拉回 {len(data)} bytes，MP4 魔数正确")

# ---- 7. 清理桶内对象 ----
mc.remove_object(BUCKET, key)
print("[7] 桶内冒烟对象已清理")

import shutil
shutil.rmtree(tmp, ignore_errors=True)
print("\nMINIO LIVE E2E PASSED —— 渲染→上传→拉回 全链路真实通过")
