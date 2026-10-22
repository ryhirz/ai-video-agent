"""T7 transform.crop 契约级约束（A'：源坐标 + 比例）。

- `w`/`h` 必填；`0 < v <= 1` 视为占比（`iw*0.8`），否则按源像素。
- `x`/`y` 可省略（ffmpeg 默认居中裁剪）。
- crop 必须出现在 `scale` 之前（作用于源画面）。
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_video_agent import compiler
from ai_video_agent.registry import validate_transform
from ai_video_agent.render import render, resolve_ffmpeg
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track, Transform

W, H = 640, 360


def _tl(**crop) -> Timeline:
    tf = Transform(crop=crop or None)
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H}, duration=8.0),
        tracks=[Track("v_main", "video")],
        clips=[Clip("v1", "v_main", "src", 0.0, 8.0, 0.0, 8.0, tf)],
    )


def _vf(**crop) -> str:
    return compiler.compile(_tl(**crop))["video_filter"]


class TestCropFilter(unittest.TestCase):
    def test_no_crop_no_filter(self) -> None:
        self.assertNotIn("crop=", _vf())

    def test_ratio(self) -> None:
        self.assertIn("crop=iw*0.8:ih*0.6", _vf(w=0.8, h=0.6))

    def test_pixels(self) -> None:
        self.assertIn("crop=200:100", _vf(w=200, h=100))

    def test_xy_explicit(self) -> None:
        self.assertIn("crop=200:100:10:20", _vf(w=200, h=100, x=10, y=20))

    def test_ratio_xy(self) -> None:
        self.assertIn("crop=iw*0.5:ih*0.5:iw*0.25:ih*0.25",
                      _vf(w=0.5, h=0.5, x=0.25, y=0.25))

    def test_full_ratio_means_no_crop(self) -> None:
        """v == 1.0 按 100% 占比处理（等价不裁），不是 1 像素。"""
        self.assertIn("crop=iw*1:ih*1", _vf(w=1.0, h=1.0))

    def test_omitted_xy_uses_ffmpeg_centering(self) -> None:
        """省略 x/y 时不写坐标 → ffmpeg 默认居中裁剪。"""
        vf = _vf(w=0.5, h=0.5)
        self.assertIn("crop=iw*0.5:ih*0.5", vf)
        self.assertNotIn("crop=iw*0.5:ih*0.5:", vf)

    def test_crop_before_scale(self) -> None:
        vf = _vf(w=0.5, h=0.5)
        self.assertLess(vf.index("crop="), vf.index(f"scale={W}:{H}"))


class TestCropValidation(unittest.TestCase):
    def test_missing_w(self) -> None:
        errs = validate_transform({"crop": {"h": 100}})
        self.assertTrue(any("缺少必填字段 w" in e for e in errs), errs)

    def test_zero_w(self) -> None:
        errs = validate_transform({"crop": {"w": 0, "h": 100}})
        self.assertTrue(any("必须 > 0" in e for e in errs), errs)

    def test_negative_x(self) -> None:
        errs = validate_transform({"crop": {"w": 100, "h": 100, "x": -1}})
        self.assertTrue(any("必须 >= 0" in e for e in errs), errs)

    def test_valid_passes(self) -> None:
        self.assertEqual(validate_transform({"crop": {"w": 0.8, "h": 0.6}}), [])

    def test_compile_raises_on_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _vf(w=0, h=100)


class TestCropRealRender(unittest.TestCase):
    def test_crop_left_half_is_blue(self) -> None:
        """真实渲染：左蓝右红的素材裁左半 -> 画面应偏蓝。"""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            ff = resolve_ffmpeg()
            subprocess.run(
                [ff, "-v", "error", "-y",
                 "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1",
                 "-f", "lavfi", "-i", "color=c=red:s=320x240:d=1",
                 "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                 "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
                check=True, capture_output=True)

            tl = Timeline(
                metadata=Metadata(schemaVersion="1.0.0", fps=30,
                                  resolution={"w": 320, "h": 240}, duration=1.0),
                tracks=[Track("v_main", "video")],
                clips=[Clip("v1", "v_main", str(src), 0.0, 1.0, 0.0, 1.0,
                            Transform(crop={"w": 0.5, "h": 1.0, "x": 0}))],
            )
            p, msg = render(tl, base_dir=str(Path(td) / "out"))
            self.assertIsNotNone(p, msg)

            r = subprocess.run(
                [ff, "-v", "error", "-ss", "0.5", "-i", p,
                 "-vf", "scale=1:1", "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                capture_output=True)
            px = tuple(r.stdout[:3])
            self.assertGreater(px[2], px[0], f"裁左半后应偏蓝：{px}")


if __name__ == "__main__":
    unittest.main()
