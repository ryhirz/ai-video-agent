"""T4 transform 全字段：pad / rotate / colorchannelmixer 的编译落地。

覆盖：单字段、组合顺序、默认值省略、表达式 rotation、越界/类型非法拦截、
以及"rotate 后尺寸回调保证 concat 安全"这条工程约束。
"""
from __future__ import annotations

import unittest

from ai_video_agent import compiler, registry
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track, Transform

W, H = 640, 360


def _tl(**tf) -> Timeline:
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H}, duration=8.0),
        tracks=[Track("v_main", "video")],
        clips=[Clip("c1", "v_main", "media_001", 0.0, 8.0, 0.0, 8.0,
                    Transform(**tf))],
    )


def _vf(**tf) -> str:
    return compiler.compile(_tl(**tf))["video_filter"]


class TestPadOnly(unittest.TestCase):
    def test_xy_produces_pad_only(self) -> None:
        vf = _vf(x=12, y=8)
        self.assertIn("pad=iw:ih:12:8", vf)
        self.assertNotIn("rotate=", vf)
        self.assertNotIn("colorchannelmixer", vf)

    def test_pad_absent_at_default(self) -> None:
        self.assertNotIn("pad=", _vf())

    def test_negative_offset_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _vf(x=-5)

    def test_non_numeric_offset_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _vf(y="top")


class TestRotateOnly(unittest.TestCase):
    def test_rotation_produces_rotate(self) -> None:
        vf = _vf(rotation=0.5)
        self.assertIn("rotate=a=0.5", vf)
        self.assertIn("ow=rotw(0.5)", vf)
        self.assertIn("oh=roth(0.5)", vf)
        self.assertNotIn("pad=", vf)
        self.assertNotIn("colorchannelmixer", vf)

    def test_rotate_keeps_concat_safe_size(self) -> None:
        """rotate 扩画布后必须回调到目标分辨率（concat 要求片段同尺寸）。"""
        vf = _vf(rotation=0.5)
        after_rotate = vf[vf.index("rotate="):]
        self.assertIn(f"scale={W}:{H}", after_rotate)

    def test_rotation_expression_string(self) -> None:
        vf = _vf(rotation="0.5*t")
        self.assertIn("rotate=a=0.5*t", vf)
        self.assertIn("ow=iw:oh=ih", vf)   # 动态表达式无法静态求 rotw/roth

    def test_rotation_zero_omits_rotate(self) -> None:
        self.assertNotIn("rotate=", _vf())


class TestOpacityOnly(unittest.TestCase):
    def test_opacity_produces_rgba_and_mixer(self) -> None:
        vf = _vf(opacity=0.5)
        self.assertIn("format=rgba", vf)
        self.assertIn("colorchannelmixer=aa=0.5", vf)
        self.assertNotIn("pad=", vf)
        self.assertNotIn("rotate=", vf)

    def test_opacity_one_omits_both(self) -> None:
        vf = _vf(opacity=1.0)
        self.assertNotIn("format=rgba", vf)
        self.assertNotIn("colorchannelmixer", vf)


class TestOrdering(unittest.TestCase):
    def test_full_transform_order(self) -> None:
        vf = _vf(x=10, y=20, rotation=0.5, opacity=0.5)
        self.assertLess(vf.index("pad=iw:ih:10:20"), vf.index("rotate=a=0.5"))
        self.assertLess(vf.index("rotate=a=0.5"), vf.index("format=rgba"))
        self.assertLess(vf.index("format=rgba"),
                        vf.index("colorchannelmixer=aa=0.5"))

    def test_legacy_normalization_scale_kept(self) -> None:
        """回归：默认情况仍保留原有 scale=W:H 归一化。"""
        self.assertIn(f"scale={W}:{H}", _vf())


class TestValidation(unittest.TestCase):
    def test_opacity_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            _vf(opacity=1.5)
        with self.assertRaises(ValueError):
            _vf(opacity=-0.1)

    def test_rotation_bad_type(self) -> None:
        with self.assertRaises(ValueError):
            _vf(rotation=[1, 2])

    def test_rotation_bad_expression(self) -> None:
        with self.assertRaises(ValueError):
            _vf(rotation="abc")

    def test_registry_transform_manifest_complete(self) -> None:
        m = registry.capability_manifest()["transform"]
        for name in ("x", "y", "rotation", "opacity", "scale"):
            self.assertIn(name, m)
            self.assertIn("description", m[name])
        self.assertEqual(m["opacity"]["min"], 0.0)
        self.assertEqual(m["opacity"]["max"], 1.0)
        self.assertEqual(m["opacity"]["default"], 1.0)
        self.assertEqual(m["rotation"]["default"], 0)


if __name__ == "__main__":
    unittest.main()
