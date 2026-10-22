"""T5.2 缺音轨补齐：anullsrc 填充 + 与 T6 gap 不叠加。

职责分离：补齐的静音 clip 只覆盖「片段本身时长」；组间空隙仍由 T6 的
gap 机制统一插静音 —— 两条路径各司其职，anullsrc 计数可精确断言。
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_video_agent import compiler
from ai_video_agent.render import render, resolve_ffmpeg
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track

W, H = 640, 360
FF = resolve_ffmpeg()
PROBE = str(Path(FF).with_name("ffprobe.exe" if Path(FF).suffix == ".exe" else "ffprobe"))


def _tl(clips) -> Timeline:
    dur = max((c.out for c in clips), default=0.0)
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H}, duration=dur),
        tracks=[Track("v_main", "video"), Track("a_main", "audio")],
        clips=list(clips),
    )


class TestFillMissingAudio(unittest.TestCase):
    def test_one_of_two_has_audio(self) -> None:
        """2 视频 + 1 音频 → 缺的第二条补静音，不再走「跳过」降级。"""
        tl = _tl([
            Clip("v1", "v_main", "src", 0.0, 1.0, 0.0, 1.0),
            Clip("v2", "v_main", "src", 1.0, 2.0, 1.0, 2.0),
            Clip("a1", "a_main", "src", 0.0, 1.0, 0.0, 1.0),
        ])
        res = compiler.compile(tl)
        # 补齐是自动修正，不记 unsupported（行为声明在 manifest 的 partial_audio_policy）
        self.assertFalse(any("不一致" in u for u in res["unsupported"]),
                         f"不应出现「不一致跳过」降级：{res['unsupported']}")
        self.assertEqual(res["audio_filter"].count("anullsrc"), 1)
        self.assertIn("concat=n=2:v=0:a=1", res["audio_filter"])

    def test_no_audio_at_all(self) -> None:
        """2 视频 + 0 音频 → 每段各补一条静音。"""
        tl = _tl([
            Clip("v1", "v_main", "src", 0.0, 1.0, 0.0, 1.0),
            Clip("v2", "v_main", "src", 1.0, 2.0, 1.0, 2.0),
        ])
        res = compiler.compile(tl)
        self.assertEqual(res["audio_filter"].count("anullsrc"), 2)
        self.assertIn("concat=n=2:v=0:a=1", res["audio_filter"])

    def test_fill_and_gap_do_not_double_insert(self) -> None:
        """缺音轨 + 组间 gap：anullsrc = 1(补齐) + 1(gap)，不叠加。"""
        tl = _tl([
            Clip("v1", "v_main", "src", 0.0, 1.0, 0.0, 1.0),
            Clip("v2", "v_main", "src", 2.5, 3.5, 1.0, 2.0),   # gap 1.5s，且缺音频
            Clip("a1", "a_main", "src", 0.0, 1.0, 0.0, 1.0),
        ])
        res = compiler.compile(tl)
        af = res["audio_filter"]
        self.assertEqual(af.count("anullsrc"), 2,
                         f"补齐 1 条 + gap 1 条 = 2，实际：{af}")
        # 补齐静音对齐片段时长（1s），gap 静音对齐空隙（1.5s）
        self.assertIn("atrim=start=0:end=1,", af)
        self.assertIn("atrim=start=0:end=1.5", af)

    def test_fill_respects_speed(self) -> None:
        """补齐静音的时长按「真实流时长」算（视频 2x → 静音也减半）。"""
        tl = _tl([
            Clip("v1", "v_main", "src", 0.0, 2.0, 0.0, 2.0,
                 ),   # 无变速
            Clip("v2", "v_main", "src", 2.0, 4.0, 0.0, 2.0),
            Clip("a1", "a_main", "src", 0.0, 2.0, 0.0, 2.0),
        ])
        tl.clips[0].effects.append(
            {"effectType": "speed", "params": {"rate": 2.0}})
        tl.clips[1].effects.append(
            {"effectType": "speed", "params": {"rate": 2.0}})
        res = compiler.compile(tl)
        # v1/v2 都变速 2x（音频 a1 配对 v1 也变速）；v2 的配对音频缺失，
        # 补齐静音时长 = 2 / 2 = 1s
        self.assertIn("atrim=start=0:end=1,", res["audio_filter"])


class TestFillRealRender(unittest.TestCase):
    def test_filled_audio_stream_is_stereo_and_aligned(self) -> None:
        """真实渲染：补齐后成片有 stereo 音轨，时长与视频对齐。"""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            subprocess.run(
                [FF, "-v", "error", "-y",
                 "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
                 "-f", "lavfi", "-i", "sine=f=440:d=2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-shortest", str(src)],
                check=True, capture_output=True)

            tl = _tl([
                Clip("v1", "v_main", str(src), 0.0, 1.0, 0.0, 1.0),
                Clip("v2", "v_main", str(src), 2.5, 3.5, 1.0, 2.0),
                Clip("a1", "a_main", str(src), 0.0, 1.0, 0.0, 1.0),  # v2 缺音频
            ])
            p, msg = render(tl, base_dir=str(Path(td) / "out"))
            self.assertIsNotNone(p, msg)

            def stream(stream: str) -> dict:
                r = subprocess.run(
                    [PROBE, "-v", "error", "-select_streams", stream,
                     "-show_entries", "stream=duration,channels",
                     "-of", "json", p], capture_output=True, text=True)
                return json.loads(r.stdout)["streams"][0]

            vid = stream("v")
            aud = stream("a")
            self.assertEqual(aud["channels"], 2,
                             f"补齐场景应统一 stereo：{aud}")
            self.assertGreaterEqual(float(aud["duration"]),
                                    float(vid["duration"]) - 0.1,
                                    f"音频 {aud} 短于视频 {vid}")


if __name__ == "__main__":
    unittest.main()
