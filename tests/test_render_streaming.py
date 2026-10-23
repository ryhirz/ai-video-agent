"""第三步核心层测试：render() 的流式进度 + 进程级两步杀取消。

进度：解析 ffmpeg stderr `time=` 折算 0.0~1.0，on_progress 单调上报；
取消：cancel_check 命中后 terminate→宽限→kill，运行目录清空、立即返回。
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ai_video_agent.render import render, resolve_ffmpeg
from ai_video_agent.timeline import Timeline

FF = resolve_ffmpeg()
W, H = 320, 240


def _gen_source(path: Path, dur: float) -> None:
    subprocess.run(
        [FF, "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=blue:s={W}x{H}:d={dur}",
         "-f", "lavfi", "-i", f"sine=f=440:d={dur}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)


def _timeline(src: str, dur: float) -> dict:
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


class TestRenderStreaming(unittest.TestCase):

    def test_progress_callback_monotonic(self) -> None:
        """10s 时间线：进度回调单调不减，末值 ≥0.9。"""
        tmp = Path(tempfile.mkdtemp(prefix="stream_prog_"))
        src = tmp / "src.mp4"
        _gen_source(src, 10.0)
        tl = Timeline.from_dict(_timeline(str(src), 10.0))

        marks: list = []
        path, msg = render(tl, base_dir=str(tmp / "out"),
                           on_progress=marks.append, poll_interval=0.2)
        self.assertIsNotNone(path, msg)
        self.assertTrue(marks, "未收到任何进度回调")
        self.assertEqual(marks, sorted(marks), "进度必须单调不减")
        self.assertGreaterEqual(marks[-1], 0.9)
        self.assertLessEqual(marks[-1], 1.0)

    def test_cancel_two_step_kill(self) -> None:
        """1s 后置位取消：ffmpeg 被杀、立即返回取消说明、半成品清空。"""
        tmp = Path(tempfile.mkdtemp(prefix="stream_cancel_"))
        src = tmp / "src.mp4"
        _gen_source(src, 60.0)
        tl = Timeline.from_dict(_timeline(str(src), 60.0))

        flag = threading.Event()
        threading.Timer(1.0, flag.set).start()

        t0 = time.monotonic()
        path, msg = render(tl, base_dir=str(tmp / "out"),
                           cancel_check=flag.is_set, poll_interval=0.2)
        elapsed = time.monotonic() - t0

        self.assertIsNone(path, "取消后不应产出成片")
        self.assertIn("取消", msg)
        # 完整渲染约需数秒~十秒级；两步杀应在置位后 ~1s 内返回
        self.assertLess(elapsed, 5.0, f"取消耗时异常：{elapsed:.1f}s")
        # 半成品必须清理干净
        leftovers = list(tmp.glob("**/output.mp4"))
        self.assertEqual(leftovers, [], f"残留半成品：{leftovers}")


if __name__ == "__main__":
    unittest.main()
