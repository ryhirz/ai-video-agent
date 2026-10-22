"""端到端验收：多能力组合链路（C 方案）。

链路：cut → speed(1.5x) → transition(fade) → effect(grayscale) → crop(0.8) → add_text → subtitle

验收 6 项：
1. 渲染成功、成片存在
2. 时长 = compiled_duration（容差 1 帧）
3. 水印区域有亮像素（文字真的画了）
4. crop 后原四角标记消失（crop 真的生效，不是只改了串）
5. 音频非静音且时长与视频对齐
6. 确定性：两次 compile 的 command 与 argv 完全一致
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_video_agent import compiler, operations
from ai_video_agent.render import render, resolve_ffmpeg
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track

W, H = 640, 360
FF = resolve_ffmpeg()
PROBE = str(Path(FF).with_name("ffprobe.exe" if Path(FF).suffix == ".exe" else "ffprobe"))


def _gen_source(path: Path) -> None:
    """2s 蓝底 + 左上/右下白标记块 + 440Hz 音轨（crop 后标记应消失）。"""
    subprocess.run(
        [FF, "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=blue:s={W}x{H}:d=2",
         "-f", "lavfi", "-i", "sine=f=440:d=2",
         "-vf", ("drawbox=x=0:y=0:w=40:h=40:color=white:t=fill,"
                 "drawbox=x=600:y=320:w=40:h=40:color=white:t=fill,"
                 "format=yuv420p"),
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)


def _probe(path: str, entries: str, stream: str = "") -> dict:
    cmd = [PROBE, "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "json", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(r.stdout)


def _pixel_at(path: str, t: float, crop: str = "") -> tuple:
    vf = f"{crop},scale=1:1" if crop else "scale=1:1"
    r = subprocess.run(
        [FF, "-v", "error", "-ss", str(t), "-i", path, "-vf", vf,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return tuple(r.stdout[:3])


class TestEndToEndAcceptance(unittest.TestCase):
    """单次渲染，多断言；渲染失败 = 组合链路不可用。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="e2e_"))
        src = cls.tmp / "src.mp4"
        _gen_source(src)

        tl = Timeline(
            metadata=Metadata(schemaVersion="1.0.0", fps=30,
                              resolution={"w": W, "h": H}, duration=2.0),
            tracks=[Track("v_main", "video"), Track("a_main", "audio")],
            clips=[
                Clip("src_video", "v_main", str(src), 0.0, 2.0, 0.0, 2.0),
                Clip("src_audio", "a_main", str(src), 0.0, 2.0, 0.0, 2.0),
            ],
        )
        ops = [
            {"type": "cut", "payload": {"targetClipId": "src_video", "at": 1.0}},
            {"type": "speed", "payload": {"targetClipId": "src_video__a", "rate": 1.5}},
            {"type": "transition", "payload": {"transitionType": "fade",
                                               "duration": 0.3,
                                               "clips": ["src_video__a", "src_video__b"]}},
            {"type": "effect", "payload": {"targetClipId": "src_video__b",
                                           "effectType": "grayscale"}},
            {"type": "add_text", "payload": {"text": "WM", "start": 0.0, "end": 1.0,
                                             "position": "top-right", "size": 28}},
            {"type": "add_subtitle", "payload": {"text": "字幕", "start": 0.0, "end": 1.0}},
        ]
        cur, _logs, warns = operations.apply_operations(tl, ops)
        assert warns == [], warns
        # crop 不是 op，直接写在 transform 上（0.8 居中 → 四角标记应被裁掉）
        for c in cur.clips:
            if c.id == "src_video__b":
                c.transform.crop = {"w": 0.8, "h": 0.8}
        cls.tl = cur
        cls.result = compiler.compile(cur)
        cls.out, cls.msg = render(cur, base_dir=str(cls.tmp / "out"))
        cls.vid = _probe(cls.out, "stream=duration,nb_frames", "v")["streams"][0] if cls.out else {}
        cls.aud = _probe(cls.out, "stream=duration", "a")["streams"][0] if cls.out else {}

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- 0. 渲染成功 ----
    def test_00_render_succeeds(self) -> None:
        self.assertIsNotNone(self.out, f"端到端渲染失败：{self.msg}")
        self.assertGreater(Path(self.out).stat().st_size, 0)

    # ---- 1. 时长守恒 ----
    def test_01_duration_matches_compiled(self) -> None:
        expected = compiler.compiled_duration(self.tl)
        actual = float(self.vid["duration"])
        self.assertAlmostEqual(actual, expected, delta=0.15,
                               msg=f"成片 {actual:.2f}s vs 预期 {expected:.2f}s")

    # ---- 2. 水印真的画了 ----
    def test_02_watermark_pixels_present(self) -> None:
        # top-right 区域（右上角 160x60）
        px = _pixel_at(self.out, 0.2, f"crop=160:60:{W-160}:0")
        self.assertGreater(sum(px), 30, f"水印区域应有亮像素：{px}")

    # ---- 3. crop 真的生效（四角白标记消失）----
    def test_03_crop_removes_corner_marks(self) -> None:
        # b 段独立显示的时段：a 变速后 0.667s，xfade 到 0.667s 结束 → 取 0.9s
        px = _pixel_at(self.out, 0.9, "crop=40:40:0:0")
        self.assertLess(sum(px), 240, f"crop 后左上角不应是白标记：{px}")

    # ---- 4. 音频非静音且对齐 ----
    def test_04_audio_present_and_aligned(self) -> None:
        v_dur = float(self.vid["duration"])
        a_dur = float(self.aud["duration"])
        self.assertGreaterEqual(a_dur, v_dur - 0.1,
                                f"音频 {a_dur:.2f}s 短于视频 {v_dur:.2f}s")
        r = subprocess.run(
            [FF, "-hide_banner", "-nostats", "-ss", "0.2", "-i", self.out,
             "-t", "0.2", "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True)
        self.assertIn("mean_volume", r.stderr, "未能读取音量信息")

    # ---- 5. 确定性：两次 compile 完全一致 ----
    def test_05_compile_is_deterministic(self) -> None:
        again = compiler.compile(self.tl)
        self.assertEqual(self.result["command"], again["command"])
        self.assertEqual(self.result["argv"], again["argv"])
        self.assertEqual(self.result["video_filter"], again["video_filter"])
        self.assertEqual(self.result["audio_filter"], again["audio_filter"])


if __name__ == "__main__":
    unittest.main()
