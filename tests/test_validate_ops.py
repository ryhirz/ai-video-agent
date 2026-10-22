"""T3 前置校验：越界 / 未知 / 非法时长的操作，必须在动 Timeline 之前被拦下。

验收要点：校验是纯函数 —— 失败时 Timeline 必须逐字节不变。
"""
from __future__ import annotations

import unittest

from ai_video_agent import operations
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track


def _tl() -> Timeline:
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": 640, "h": 360}, duration=8.0),
        tracks=[Track("v_main", "video"), Track("a_main", "audio")],
        clips=[Clip("c1", "v_main", "media_001", 0.0, 8.0, 0.0, 8.0)],
    )


class TestEffectParamRange(unittest.TestCase):
    def test_out_of_range_saturation_rejected_and_timeline_untouched(self) -> None:
        tl = _tl()
        before = tl.dumps()
        errs = operations.validate_ops(tl, [{
            "id": "op_1", "type": "effect",
            "payload": {"targetClipId": "c1", "effectType": "saturation",
                        "params": {"value": 5}},
        }])
        self.assertTrue(errs)
        self.assertEqual(errs[0]["stage"], "validate")
        self.assertEqual(errs[0]["opId"], "op_1")
        self.assertEqual(errs[0]["field"], "payload.params.value")
        self.assertIn("超过上限", errs[0]["reason"])
        self.assertEqual(tl.dumps(), before, "validate_ops 不得修改 Timeline")

    def test_in_range_saturation_passes(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "effect",
            "payload": {"targetClipId": "c1", "effectType": "saturation",
                        "params": {"value": 1.2}},
        }])
        self.assertEqual(errs, [])

    def test_unknown_effect_rejected(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "effect",
            "payload": {"targetClipId": "c1", "effectType": "cinematic"},
        }])
        self.assertTrue(errs)
        self.assertIn("不支持的效果", errs[0]["reason"])

    def test_non_numeric_param_rejected(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "effect",
            "payload": {"targetClipId": "c1", "effectType": "brightness",
                        "params": {"value": "很亮"}},
        }])
        self.assertTrue(any("必须是数字" in e["reason"] for e in errs))


class TestOperationAndTiming(unittest.TestCase):
    def test_unknown_operation_type(self) -> None:
        errs = operations.validate_ops(_tl(), [{"type": "teleport", "payload": {}}])
        self.assertEqual(errs[0]["field"], "type")
        self.assertIn("不支持的操作类型", errs[0]["reason"])

    def test_missing_type(self) -> None:
        errs = operations.validate_ops(_tl(), [{"payload": {}}])
        self.assertEqual(errs[0]["field"], "type")

    def test_cut_at_outside_clip(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "cut", "payload": {"targetClipId": "c1", "at": 99.0},
        }])
        self.assertTrue(any("不在片段" in e["reason"] for e in errs))

    def test_transition_bad_duration(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "transition",
            "payload": {"transitionType": "fade", "duration": 0, "clips": ["c1", "c2"]},
        }])
        self.assertTrue(any(e["field"] == "payload.duration" for e in errs))

    def test_transition_unknown_type(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "transition",
            "payload": {"transitionType": "explode", "duration": 1, "clips": ["c1", "c2"]},
        }])
        self.assertTrue(any(e["field"] == "payload.transitionType" for e in errs))

    def test_bgm_volume_out_of_range(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "add_bgm",
            "payload": {"sourceRef": "bgm", "in": 0, "out": 8, "volume": 3},
        }])
        self.assertTrue(any(e["field"] == "payload.volume" for e in errs))

    def test_bgm_fade_longer_than_clip(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "add_bgm",
            "payload": {"sourceRef": "bgm", "in": 0, "out": 4,
                        "fadeIn": 3, "fadeOut": 3},
        }])
        self.assertTrue(any(e["field"] == "payload.fadeIn" for e in errs))

    def test_subtitle_bad_time(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "add_subtitle", "payload": {"text": "x", "start": 5, "end": 3},
        }])
        self.assertTrue(errs)

    def test_valid_batch_passes(self) -> None:
        ops = [
            {"type": "effect", "payload": {"targetClipId": "c1",
                                           "effectType": "brightness",
                                           "params": {"value": 0.2}}},
            {"type": "add_subtitle", "payload": {"text": "hi", "start": 1, "end": 2}},
            {"type": "trim", "payload": {"targetClipId": "c1", "in": 1, "out": 6}},
        ]
        self.assertEqual(operations.validate_ops(_tl(), ops), [])


class TestApplyOperationsStopsInvalid(unittest.TestCase):
    def test_invalid_op_skipped_and_timeline_stays_clean(self) -> None:
        tl = _tl()
        before = tl.dumps()
        cur, logs, warns = operations.apply_operations(tl, [{
            "type": "effect",
            "payload": {"targetClipId": "c1", "effectType": "saturation",
                        "params": {"value": 5}},
        }])
        self.assertEqual(cur.dumps(), before, "越界操作不得改动 Timeline")
        self.assertEqual(logs, [])
        self.assertTrue(any("校验失败" in w for w in warns), warns)

    def test_strict_mode_raises(self) -> None:
        with self.assertRaises(operations.OperationError):
            operations.apply_operations(_tl(), [{
                "type": "effect",
                "payload": {"targetClipId": "c1", "effectType": "saturation",
                            "params": {"value": 5}},
            }], strict=True)
