"""T6 契约级约束：gap/tpad（方案 A）+ 字幕样式。

方案 A：gap 只插在「无转场的相邻片段之间」；转场组内部不插 gap。
守恒式：成片时长 = Σ片段流时长 − Σ转场时长 + Σgap。
"""
from __future__ import annotations

import unittest

from ai_video_agent import compiler
from ai_video_agent.timeline import Clip, Metadata, Subtitle, Timeline, Track, Transition

W, H = 640, 360


def _v(cid: str, s_in: float, s_out: float, t_in: float | None = None) -> Clip:
    """视频片段：s_in/s_out 是源区间，t_in 是时间轴入点（默认等于 s_in）。"""
    t_in = s_in if t_in is None else t_in
    return Clip(cid, "v_main", "src_" + cid, t_in, t_in + (s_out - s_in), s_in, s_out)


def _tl(clips, transitions=(), subtitles=()) -> Timeline:
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H},
                          duration=max((c.out for c in clips), default=0.0)),
        tracks=[Track("v_main", "video"), Track("s_main", "subtitle")],
        clips=list(clips), transitions=list(transitions), subtitles=list(subtitles),
    )


class TestGapTpad(unittest.TestCase):
    def test_gap_between_clips_inserts_tpad(self) -> None:
        # A 时间轴 0-4；B 时间轴 6-10 → gap 2s
        vf = compiler.compile(_tl([_v("a", 0, 4), _v("b", 4, 8, t_in=6)]))["video_filter"]
        self.assertIn("tpad=", vf)
        self.assertIn("stop_duration=2", vf)

    def test_no_gap_inserts_no_tpad(self) -> None:
        vf = compiler.compile(_tl([_v("a", 0, 4), _v("b", 4, 8)]))["video_filter"]
        self.assertNotIn("tpad=", vf)

    def test_transition_group_has_no_tpad(self) -> None:
        """有转场的相邻片段：转场本身即衔接，不插 gap。"""
        tl = _tl([_v("a", 0, 4), _v("b", 4, 8)],
                 [Transition("t1", "fade", 1.0, ["a", "b"])])
        vf = compiler.compile(tl)["video_filter"]
        self.assertIn("xfade=", vf)
        self.assertNotIn("tpad=", vf)

    def test_gap_after_transition_group(self) -> None:
        """A-B 转场成组，C 与 B 之间有 2s gap → tpad 加在组流之后。"""
        tl = _tl([_v("a", 0, 4), _v("b", 4, 8), _v("c", 8, 12, t_in=10)],
                 [Transition("t1", "fade", 1.0, ["a", "b"])])
        vf = compiler.compile(tl)["video_filter"]
        self.assertIn("xfade=", vf)
        self.assertIn("tpad=", vf)
        self.assertIn("stop_duration=2", vf)


class TestDurationConservationWithGap(unittest.TestCase):
    def test_total_includes_gap(self) -> None:
        # 4 + gap 2 + 4 = 10
        tl = _tl([_v("a", 0, 4), _v("b", 4, 8, t_in=6)])
        self.assertAlmostEqual(compiler.compiled_duration(tl), 10.0)

    def test_total_with_transition_and_gap(self) -> None:
        # A4 + B4 - trans1 + C4 + gap2 = 13
        tl = _tl([_v("a", 0, 4), _v("b", 4, 8), _v("c", 8, 12, t_in=10)],
                 [Transition("t1", "fade", 1.0, ["a", "b"])])
        self.assertAlmostEqual(compiler.compiled_duration(tl), 13.0)

    def test_no_gap_no_transition_keeps_sum(self) -> None:
        tl = _tl([_v("a", 0, 4), _v("b", 4, 8)])
        self.assertAlmostEqual(compiler.compiled_duration(tl), 8.0)


class TestSubtitleStyle(unittest.TestCase):
    def test_font_size_color_position_applied(self) -> None:
        tl = _tl([_v("a", 0, 4)], subtitles=[
            Subtitle("s1", "s_main", "你好", 0.5, 2.0, {
                "font": "Arial", "size": 48, "color": "#FF0000",
                "position": "top-center"})])
        ass = compiler.build_ass_content(tl)
        self.assertIn("\\fnArial", ass)
        self.assertIn("\\fs48", ass)
        self.assertIn("&H000000FF", ass)      # ASS 是 BBGGRR：红 = 0000FF
        self.assertIn("\\an8", ass)            # top-center

    def test_default_style_keeps_ass_defaults(self) -> None:
        tl = _tl([_v("a", 0, 4)], subtitles=[
            Subtitle("s1", "s_main", "默认", 0.5, 2.0)])
        ass = compiler.build_ass_content(tl)
        self.assertIn("Noto Sans CJK SC", ass)
        self.assertNotIn("\\fn", ass)

    def test_bg_applied(self) -> None:
        tl = _tl([_v("a", 0, 4)], subtitles=[
            Subtitle("s1", "s_main", "带底", 0.5, 2.0,
                     {"bg": "rgba(0,0,0,0.5)"})])
        ass = compiler.build_ass_content(tl)
        self.assertIn("\\4c", ass)             # 背景色走边框色覆盖


if __name__ == "__main__":
    unittest.main()
