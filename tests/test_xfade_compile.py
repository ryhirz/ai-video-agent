"""T5 真转场：xfade / acrossfade 的契约级约束（先红后绿）。

三条定义（已评审通过）：
1. offset = clipA 流时长(sourceOut-sourceIn) - duration；零时长素材直接报错
2. acrossfade 的 d 必须等于 xfade 的 duration；曲线固定 fade
3. 分组：组内相邻必有 transition、组间相邻必无；xfade 链 + 组间 concat
"""
from __future__ import annotations

import unittest

from ai_video_agent import compiler, registry
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track, Transition

W, H = 640, 360


def _v(cid: str, s_in: float, s_out: float) -> Clip:
    return Clip(cid, "v_main", "src_" + cid, s_in, s_out, s_in, s_out)


def _a(cid: str, s_in: float, s_out: float) -> Clip:
    return Clip(cid, "a_main", "src_" + cid, s_in, s_out, s_in, s_out)


def _tl(clips, transitions=()) -> Timeline:
    dur = max((c.out for c in clips), default=0.0)
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H}, duration=dur),
        tracks=[Track("v_main", "video"), Track("a_main", "audio")],
        clips=list(clips),
        transitions=list(transitions),
    )


class TestOffset(unittest.TestCase):
    def test_offset_uses_stream_duration(self) -> None:
        # A: sourceIn=10, sourceOut=18 -> 流时长 8；D=1 -> offset = 7
        self.assertAlmostEqual(compiler.xfade_offset(_v("a", 10, 18), 1.0), 7.0)

    def test_rejected_when_second_clip_shorter(self) -> None:
        """duration 必须小于两段流时长的较小值（不能只看第一段）。"""
        tl = _tl([_v("a", 0, 4), _v("b", 4, 6)],
                 [Transition("t1", "fade", 3.0, ["a", "b"])])
        res = compiler.compile(tl)
        self.assertNotIn("xfade=", res["video_filter"])
        self.assertTrue(any("不合法" in u for u in res["unsupported"]),
                        res["unsupported"])

    def test_zero_duration_clip_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compiler.xfade_offset(_v("z", 5, 5), 1.0)

    def test_duration_not_shorter_than_clip_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compiler.xfade_offset(_v("a", 0, 2), 2.0)

    def test_offset_lands_in_filtergraph(self) -> None:
        result = compiler.compile(_tl(
            [_v("a", 10, 18), _v("b", 0, 8)],
            [Transition("t1", "fade", 1.0, ["a", "b"])]))
        self.assertIn("xfade=transition=fade:duration=1:offset=7",
                      result["video_filter"])


class TestAcrossfadeParity(unittest.TestCase):
    def test_acrossfade_duration_equals_xfade(self) -> None:
        result = compiler.compile(_tl(
            [_v("a", 0, 4), _v("b", 4, 8), _a("aa", 0, 4), _a("ab", 4, 8)],
            [Transition("t1", "fade", 1.0, ["a", "b"])]))
        self.assertIn("xfade=transition=fade:duration=1:offset=3",
                      result["video_filter"])
        self.assertIn("acrossfade=d=1:c1=tri:c2=tri", result["audio_filter"])


class TestGrouping(unittest.TestCase):
    def test_break_at_missing_transition(self) -> None:
        clips = [_v("a", 0, 4), _v("b", 4, 8), _v("c", 8, 12), _v("d", 12, 16)]
        trans = [Transition("t1", "fade", 1, ["a", "b"]),
                 Transition("t2", "fade", 1, ["b", "c"])]
        groups = compiler.group_clips_by_transition(clips, trans)
        self.assertEqual([[c.id for c in g] for g in groups],
                         [["a", "b", "c"], ["d"]])

    def test_singletons_without_transitions(self) -> None:
        clips = [_v("a", 0, 4), _v("b", 4, 8)]
        groups = compiler.group_clips_by_transition(clips, [])
        self.assertEqual([len(g) for g in groups], [1, 1])


class TestDurationConservation(unittest.TestCase):
    def test_total_is_sum_minus_transitions(self) -> None:
        clips = [_v("a", 0, 4), _v("b", 4, 8), _v("c", 8, 12)]
        trans = [Transition("t1", "fade", 1, ["a", "b"]),
                 Transition("t2", "fade", 1, ["b", "c"])]
        self.assertAlmostEqual(compiler.xfade_total_duration(clips, trans), 10.0)

    def test_no_transition_keeps_sum(self) -> None:
        clips = [_v("a", 0, 4), _v("b", 4, 8)]
        self.assertAlmostEqual(compiler.xfade_total_duration(clips, []), 8.0)


class TestNoAudio(unittest.TestCase):
    def test_video_only_still_xfades(self) -> None:
        result = compiler.compile(_tl(
            [_v("a", 0, 4), _v("b", 4, 8)],
            [Transition("t1", "fade", 1.0, ["a", "b"])]))
        self.assertIn("xfade=", result["video_filter"])
        self.assertNotIn("acrossfade", result["audio_filter"])


class TestLegacyPathUnchanged(unittest.TestCase):
    def test_no_transition_still_uses_concat(self) -> None:
        result = compiler.compile(_tl([_v("a", 0, 4), _v("b", 4, 8)]))
        self.assertNotIn("xfade", result["video_filter"])
        self.assertIn("concat=n=2", result["video_filter"])


class TestChainOffsetAccumulates(unittest.TestCase):
    def test_three_clip_chain_offsets(self) -> None:
        """链式 xfade：第二对的 offset 基准是「累积流时长」，不是单段时长。"""
        result = compiler.compile(_tl(
            [_v("a", 0, 4), _v("b", 4, 8), _v("c", 8, 12)],
            [Transition("t1", "fade", 1, ["a", "b"]),
             Transition("t2", "fade", 1, ["b", "c"])]))
        vf = result["video_filter"]
        # 第一对: 4-1=3；累积 4+4-1=7 -> 第二对: 7-1=6
        self.assertIn("offset=3", vf)
        self.assertIn("offset=6", vf)


class TestManifestTransitionExported(unittest.TestCase):
    """清单诚实性（T5）：transition 段必须导出真实使用的 xfade/acrossfade 映射。"""

    def test_manifest_has_transition_specs(self) -> None:
        tr = registry.capability_manifest()["transition"]
        self.assertEqual(set(tr), {"fade", "cross", "wipe"})
        for name, spec in tr.items():
            self.assertTrue(spec["supported"], name)
            self.assertIn("xfade=transition=", spec["video"])
            self.assertIn("acrossfade=c1=", spec["audio"])

    def test_transition_types_still_compatible(self) -> None:
        self.assertEqual(list(registry.TRANSITION_TYPES), ["fade", "cross", "wipe"])

    def test_manifest_matches_compiled_filters(self) -> None:
        """manifest 声明的 video/audio 映射，必须与编译器实际产出对得上。"""
        tr = registry.capability_manifest()["transition"]
        for name, spec in tr.items():
            result = compiler.compile(_tl(
                [_v("a", 0, 4), _v("b", 4, 8), _a("aa", 0, 4), _a("ab", 4, 8)],
                [Transition("t1", name, 1.0, ["a", "b"])]))
            self.assertIn(spec["video"], result["video_filter"], name)
            self.assertIn("c1=tri:c2=tri", result["audio_filter"], name)


if __name__ == "__main__":
    unittest.main()
