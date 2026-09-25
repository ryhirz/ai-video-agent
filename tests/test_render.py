"""渲染层测试：验证 compiler 的 ffmpeg 指令真的能出 MP4（不只是生成命令字符串）。

用项目自带 ffmpeg 临时生成一段纯色测试片源，构造只在内存里的 Timeline，
直接调用 render() 出片并校验产物存在且时长 > 0。
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_video_agent.timeline import Clip, Metadata, Timeline, Track
from ai_video_agent.render import render, resolve_ffmpeg


def _gen_clip(path: Path, dur: float = 2.0) -> None:
    ffmpeg = resolve_ffmpeg()
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={dur}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, text=True,
    )


def _timeline(path: Path, dur: float = 2.0) -> Timeline:
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                         resolution={"w": 320, "h": 240}, duration=dur),
        tracks=[Track("v_main", "video"), Track("a_main", "audio")],
        clips=[Clip(id="src_video", track="v_main", sourceRef=str(path),
                    in_=0.0, out=dur, sourceIn=0.0, sourceOut=dur)],
    )


class TestRender(unittest.TestCase):
    def test_resolve_ffmpeg_found(self) -> None:
        p = resolve_ffmpeg()
        self.assertIsNotNone(p, "resolve_ffmpeg 应找到 ffmpeg")
        self.assertTrue(Path(p).exists(), f"{p} 应存在")

    def test_render_produces_playable_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "src.mp4"
            _gen_clip(src)

            out, msg = render(_timeline(src), base_dir=str(td))
            self.assertIsNotNone(out, f"应成功出片，实际：{msg}")
            self.assertTrue(Path(out).exists(), f"成片应存在：{out}")

            # 用 ffprobe 校验时长 > 0
            ffprobe = str(Path(resolve_ffmpeg()).with_name(
                "ffprobe.exe" if os.name == "nt" else "ffprobe"))
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", out],
                capture_output=True, text=True,
            )
            dur = float(probe.stdout.strip())
            self.assertGreater(dur, 0.0, "成片时长应 > 0")

    def test_render_bad_input_fails_gracefully(self) -> None:
        """源文件不存在时优雅返回 (None, 原因)，不抛异常、不在项目 outputs 留垃圾。"""
        with tempfile.TemporaryDirectory() as td:
            out, msg = render(_timeline(Path(td) / "nope.mp4"), base_dir=td)
            self.assertIsNone(out)
            self.assertTrue(msg, "失败时应给出原因说明")


if __name__ == "__main__":
    unittest.main()
