"""T7 add_text 契约级约束（先红后绿）。

坐标方案 C：`position`（九宫格）+ `margin` 默认 20；`x` / `y` 非空时覆盖前者。
字体：本机 fontconfig 实际不可用（Cannot load default config file），
      `font` 必须解析为 `fontfile=` 字体文件路径。
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
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H}, duration=8.0),
        tracks=[Track("v_main", "video")],
        clips=[Clip("v1", "v_main", "src", 0.0, 8.0, 0.0, 8.0)],
    )


def _vf(**payload) -> str:
    base = {"text": "Hello", "start": 0.0, "end": 3.0}
    base.update(payload)
    tl, _logs, warns = operations.apply_operations(
        _tl(), [{"type": "add_text", "payload": base}])
    if warns:
        raise AssertionError(f"add_text 未实现或校验失败：{warns}")
    return compiler.compile(tl)["video_filter"]


class TestBasicFilter(unittest.TestCase):
    def test_emits_drawtext_with_text(self) -> None:
        vf = _vf()
        self.assertIn("drawtext=", vf)
        self.assertIn("text='Hello'", vf)

    def test_size_and_color(self) -> None:
        vf = _vf(size=32, color="#FF0000")
        self.assertIn("fontsize=32", vf)
        self.assertIn("fontcolor=#FF0000", vf)

    def test_font_resolved_to_fontfile(self) -> None:
        """fontconfig 不可用，font 必须落成 fontfile 路径。"""
        vf = _vf(font="Arial")
        self.assertIn("fontfile=", vf)
        self.assertIn("arial", vf.lower())


class TestPosition(unittest.TestCase):
    def test_default_bottom_center_margin_20(self) -> None:
        vf = _vf()
        self.assertIn("x=(w-text_w)/2", vf)
        self.assertIn("y=h-text_h-20", vf)

    def test_top_left(self) -> None:
        vf = _vf(position="top-left")
        self.assertIn("x=20", vf)
        self.assertIn("y=20", vf)

    def test_center(self) -> None:
        vf = _vf(position="center")
        self.assertIn("x=(w-text_w)/2", vf)
        self.assertIn("y=(h-text_h)/2", vf)

    def test_custom_margin(self) -> None:
        vf = _vf(position="bottom-right", margin=48)
        self.assertIn("x=w-text_w-48", vf)
        self.assertIn("y=h-text_h-48", vf)

    def test_xy_overrides_position(self) -> None:
        vf = _vf(position="top-left", x=100, y=50)
        self.assertIn("x=100", vf)
        self.assertIn("y=50", vf)


class TestTiming(unittest.TestCase):
    def test_enable_between(self) -> None:
        vf = _vf(start=1.5, end=4.0)
        self.assertIn("between(t,1.5,4)", vf)

    def test_zero_start(self) -> None:
        vf = _vf(start=0, end=2)
        self.assertIn("between(t,0,2)", vf)


class TestBackground(unittest.TestCase):
    def test_bg_enables_box(self) -> None:
        vf = _vf(bg="#00000080")
        self.assertIn("box=1", vf)
        self.assertIn("boxcolor=#00000080", vf)

    def test_no_bg_no_box(self) -> None:
        self.assertNotIn("box=1", _vf())


class TestEscaping(unittest.TestCase):
    def test_colon_is_escaped(self) -> None:
        """drawtext 的 text 里 `:` 必须转义，否则会被当作参数分隔符。"""
        vf = _vf(text="a:b")
        self.assertIn("a\\:b", vf)


class TestValidation(unittest.TestCase):
    def test_missing_text_rejected(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "add_text", "payload": {"start": 0, "end": 1}}])
        self.assertTrue(any(e["field"] == "payload.text" for e in errs), errs)

    def test_bad_position_rejected(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "add_text",
            "payload": {"text": "x", "start": 0, "end": 1, "position": "middle"}}])
        self.assertTrue(any(e["field"] == "payload.position" for e in errs), errs)

    def test_bad_time_rejected(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "add_text",
            "payload": {"text": "x", "start": 3, "end": 1}}])
        self.assertTrue(any(e["field"] == "payload.start" for e in errs), errs)

    def test_valid_passes(self) -> None:
        errs = operations.validate_ops(_tl(), [{
            "type": "add_text",
            "payload": {"text": "x", "start": 0, "end": 1}}])
        self.assertEqual(errs, [])


class TestAddTextRealRender(unittest.TestCase):
    def test_text_is_actually_drawn(self) -> None:
        """真实渲染：文字必须真的出现在画面上（不只看 filter 串）。"""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            ff = resolve_ffmpeg()
            subprocess.run(
                [ff, "-v", "error", "-y",
                 "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
                check=True, capture_output=True)

            tl = Timeline(
                metadata=Metadata(schemaVersion="1.0.0", fps=30,
                                  resolution={"w": 320, "h": 240}, duration=1.0),
                tracks=[Track("v_main", "video")],
                clips=[Clip("v1", "v_main", str(src), 0.0, 1.0, 0.0, 1.0)],
            )
            cur, _logs, warns = operations.apply_operations(tl, [{
                "type": "add_text",
                "payload": {"text": "TEST", "start": 0, "end": 1,
                            "position": "center", "size": 48,
                            "color": "#FFFFFF", "font": "Arial"}}])
            self.assertEqual(warns, [])

            p, msg = render(cur, base_dir=str(Path(td) / "out"))
            self.assertIsNotNone(p, msg)

            # 抽 0.5s 的中心区域并缩成 1 像素：黑底 + 白字 → 平均亮度应 > 0
            r = subprocess.run(
                [ff, "-v", "error", "-ss", "0.5", "-i", p,
                 "-vf", "crop=200:80:60:80,scale=1:1",
                 "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                capture_output=True)
            px = tuple(r.stdout[:3])
            self.assertGreater(sum(px), 10,
                               f"文字未渲染到画面（中心区域平均像素 {px}）")


if __name__ == "__main__":
    unittest.main()
