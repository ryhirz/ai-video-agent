"""T7 speed 变速：契约级约束。

三个关键回归（附件方案会踩的坑）：
1. speed 必须随 Timeline 序列化（不能存动态属性）
2. 视频 setpts 必须在 trim 之后
3. 变速要做 A/V 同步（否则 -shortest 截断音频）
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_video_agent import compiler, operations
from ai_video_agent.render import render, resolve_ffmpeg
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track

W, H = 640, 360


def _tl() -> Timeline:
    """一个视频 clip + 一条配对音频（同 sourceRef / 同源区间）。"""
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H}, duration=8.0),
        tracks=[Track("v_main", "video"), Track("a_main", "audio")],
        clips=[
            Clip("v1", "v_main", "src", 0.0, 8.0, 0.0, 8.0),
            Clip("a1", "a_main", "src", 0.0, 8.0, 0.0, 8.0),
        ],
    )


def _with_speed(rate: float) -> Timeline:
    cur, _logs, warns = operations.apply_operations(
        _tl(), [{"type": "speed", "payload": {"targetClipId": "v1", "rate": rate}}])
    assert warns == [], warns
    return cur


class TestAtempoFilters(unittest.TestCase):
    def test_rate_one_is_noop(self) -> None:
        self.assertEqual(compiler._atempo_filters(1.0), [])

    def test_single_stage(self) -> None:
        self.assertEqual(compiler._atempo_filters(0.5), ["atempo=0.5"])
        self.assertEqual(compiler._atempo_filters(2.0), ["atempo=2"])
        self.assertEqual(compiler._atempo_filters(100.0), ["atempo=100"])

    def test_fast_chain(self) -> None:
        self.assertEqual(compiler._atempo_filters(200.0), ["atempo=100", "atempo=2"])

    def test_slow_chain(self) -> None:
        self.assertEqual(compiler._atempo_filters(0.25), ["atempo=0.5", "atempo=0.5"])

    def test_extreme_chain(self) -> None:
        self.assertEqual(compiler._atempo_filters(1500.0), ["atempo=100", "atempo=15"])


class TestSpeedSerialization(unittest.TestCase):
    def test_survives_timeline_roundtrip(self) -> None:
        """speed 必须写进可序列化字段，dump/load 后仍在。"""
        tl = _with_speed(2.0)
        reloaded = Timeline.loads(tl.dumps())
        self.assertEqual(compiler._speed_rate(reloaded.clips[0]), 2.0)

    def test_survives_followup_operation(self) -> None:
        """[speed, effect] 两条操作后 speed 仍在（动态属性方案会在此丢失）。"""
        tl = _with_speed(2.0)
        cur, _logs, warns = operations.apply_operations(tl, [
            {"type": "effect", "payload": {"targetClipId": "v1",
                                           "effectType": "grayscale"}}])
        self.assertEqual(warns, [])
        self.assertEqual(compiler._speed_rate(cur.clips[0]), 2.0)


class TestSpeedCompile(unittest.TestCase):
    def test_video_setpts_after_trim(self) -> None:
        """setpts 变速必须在 trim 之后，否则切点按变速后 PTS 解释。"""
        vf = compiler.compile(_with_speed(2.0))["video_filter"]
        self.assertIn("setpts=PTS/2", vf)
        self.assertLess(vf.index("trim=start="), vf.index("setpts=PTS/2"))

    def test_audio_atempo_synced(self) -> None:
        """A/V 同步：配对音频也必须带 atempo。"""
        af = compiler.compile(_with_speed(2.0))["audio_filter"]
        self.assertIn("atempo=2", af)

    def test_rate_one_no_filter(self) -> None:
        res = compiler.compile(_with_speed(1.0))
        self.assertNotIn("PTS/", res["video_filter"])
        self.assertNotIn("atempo", res["audio_filter"])

    def test_speed_not_reported_unsupported(self) -> None:
        res = compiler.compile(_with_speed(2.0))
        self.assertFalse(any("speed" in u for u in res["unsupported"]),
                         res["unsupported"])

    def test_only_target_clip(self) -> None:
        tl = Timeline(
            metadata=Metadata(schemaVersion="1.0.0", fps=30,
                              resolution={"w": W, "h": H}, duration=8.0),
            tracks=[Track("v_main", "video")],
            clips=[Clip("a", "v_main", "s1", 0.0, 4.0, 0.0, 4.0),
                   Clip("b", "v_main", "s2", 4.0, 8.0, 0.0, 4.0)],
        )
        cur, _l, _w = operations.apply_operations(tl, [
            {"type": "speed", "payload": {"targetClipId": "a", "rate": 2.0}}])
        vf = compiler.compile(cur)["video_filter"]
        self.assertEqual(vf.count("setpts=PTS/2"), 1)


class TestSpeedValidation(unittest.TestCase):
    def test_zero_rate_rejected(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "speed", "payload": {"targetClipId": "v1", "rate": 0}}])
        self.assertTrue(any(e["field"] == "payload.rate" for e in errs), errs)

    def test_too_large_rate_rejected(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "speed", "payload": {"targetClipId": "v1", "rate": 1001}}])
        self.assertTrue(any(e["field"] == "payload.rate" for e in errs), errs)

    def test_valid_rate_passes(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "speed", "payload": {"targetClipId": "v1", "rate": 2}}])
        self.assertEqual(errs, [])


class TestSpeedRealRender(unittest.TestCase):
    def test_two_x_halves_duration(self) -> None:
        """真实渲染冒烟：2x 后成片时长约为源的一半。"""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            ff = resolve_ffmpeg()
            subprocess.run(
                [ff, "-v", "error", "-y",
                 "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d=2",
                 "-f", "lavfi", "-i", "sine=f=440:d=2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-shortest", str(src)],
                check=True, capture_output=True)

            tl = Timeline(
                metadata=Metadata(schemaVersion="1.0.0", fps=30,
                                  resolution={"w": 320, "h": 240}, duration=2.0),
                tracks=[Track("v_main", "video"), Track("a_main", "audio")],
                clips=[
                    Clip("v1", "v_main", str(src), 0.0, 2.0, 0.0, 2.0),
                    Clip("a1", "a_main", str(src), 0.0, 2.0, 0.0, 2.0),
                ],
            )
            cur, _logs, warns = operations.apply_operations(
                tl, [{"type": "speed", "payload": {"targetClipId": "v1", "rate": 2.0}}])
            self.assertEqual(warns, [])

            p, msg = render(cur, base_dir=str(Path(td) / "out"))
            self.assertIsNotNone(p, msg)

            probe = str(Path(ff).with_name("ffprobe.exe"))
            r = subprocess.run(
                [probe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", p], capture_output=True, text=True)
            dur = float(r.stdout.strip())
            self.assertAlmostEqual(dur, 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
