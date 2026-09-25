"""端到端联调：M1 摄取 -> 初始 Timeline -> Agent 下 cut 指令 -> 确定性编译。

验证「Footage Index 是唯一真相、AI 只产结构化操作、编译器确定性翻译」整条链在真实
ffmpeg + cv2 下跑通（不依赖 ML 重依赖）。M2 冒烟见 test_understand.py。
"""
import os
import tempfile
import unittest
import subprocess

from ai_video_agent.ingest import build_footage_index, build_initial_timeline
from ai_video_agent.operations import apply_operation
from ai_video_agent.compiler import compile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = os.path.join(HERE, "tools", "ffmpeg", "bin", "ffmpeg.exe")


def _make_clip(path: str, dur: float = 4.0) -> None:
    """生成前2s红、后2s蓝的测试片（含一次硬切变）。"""
    subprocess.run([
        FF, "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
        "-filter_complex", "[0][1]concat=n=2:v=1:a=0", path,
    ], capture_output=True, text=True, check=True)


class TestPipelineM1M3(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.clip = os.path.join(self.tmp, "t.mp4")
        _make_clip(self.clip)

    def test_ingest_to_compile(self) -> None:
        idx = build_footage_index(self.clip)
        tl = build_initial_timeline(idx)

        # Agent 下一条自然语言指令："在 2.0s 处把视频剪开"
        op = {"id": "op1", "type": "cut", "actor": "planner",
              "payload": {"targetClipId": "src_video", "at": 2.0}}
        tl2 = apply_operation(tl, op)

        # 切成两段（不变量保持）
        self.assertEqual(len(tl2.clips), 2)
        a, b = tl2.clips[0], tl2.clips[1]
        self.assertAlmostEqual((a.out - a.in_), (a.sourceOut - a.sourceIn))
        self.assertAlmostEqual((b.out - b.in_), (b.sourceOut - b.sourceIn))

        # 确定性编译：filtergraph 含 trim + concat=n=2
        out = compile(tl2)
        self.assertIn("trim=start=", out["command"])
        self.assertIn("concat=n=2", out["command"])
        # 同输入重编译结果一致（确定性）
        self.assertEqual(out["command"], compile(tl2)["command"])


if __name__ == "__main__":
    unittest.main()
