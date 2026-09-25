"""M1 摄取层单测：用 ffmpeg 生成含一次硬切变的测试片，验证 probe / Footage Index / 初始 Timeline。"""
import os
import tempfile
import unittest
import subprocess

from ai_video_agent.footage_index import FootageIndex
from ai_video_agent.ingest import probe, build_footage_index, build_initial_timeline

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = os.path.join(HERE, "tools", "ffmpeg", "bin", "ffmpeg.exe")


def _make_test_clip(path: str, dur: float = 4.0) -> None:
    """生成前2s红、后2s蓝的测试片（含一次硬切变），用于验证场景检测。"""
    cmd = [
        FF, "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
        "-filter_complex", "[0][1]concat=n=2:v=1:a=0", path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


class TestIngest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.clip = os.path.join(self.tmp, "test.mp4")
        _make_test_clip(self.clip)

    def test_probe_metadata(self) -> None:
        m = probe(self.clip)
        self.assertGreater(m["duration"], 3.5)
        self.assertEqual(m["width"], 320)
        self.assertEqual(m["height"], 240)
        self.assertTrue(m["has_audio"] is False)  # lavfi color 无音轨

    def test_footage_index_structure(self) -> None:
        idx = build_footage_index(self.clip)
        self.assertGreater(len(idx.scenes), 0)
        self.assertGreater(len(idx.keyframes), 0)
        # 关键帧应在时间轴范围内
        for t in idx.keyframes:
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, idx.metadata["duration"])

    def test_initial_timeline_bridges_m3(self) -> None:
        idx = build_footage_index(self.clip)
        tl = build_initial_timeline(idx)
        # 无音轨素材 -> 仅 1 条视频 clip
        self.assertEqual(len(tl.clips), 1)
        self.assertEqual(tl.clips[0].id, "src_video")
        # 不变量 (out-in) === (sourceOut-sourceIn) 成立
        c = tl.clips[0]
        self.assertAlmostEqual((c.out - c.in_), (c.sourceOut - c.sourceIn))
        self.assertEqual(tl.metadata.duration, idx.metadata["duration"])


if __name__ == "__main__":
    unittest.main()
