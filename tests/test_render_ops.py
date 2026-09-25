"""「操作 -> 成片」层回归测试。

为什么需要这一层？
    原有 75 个用例里，operation 的测试只断言「Timeline 字段被改了」，compiler 的测试
    只断言「命令字符串里有某个片段」。两边都绿，却完全测不出**操作对成片毫无影响**——
    实测发现 add_bgm / move_clip / transition / effect 四类操作当时能成功改 Timeline，
    编译出的命令却一模一样，成片自然一点变化都没有（等于静默失效）。
    本文件把「操作必须改变成片」变成断言，堵住这个盲区。

两类断言：
    1) 编译层（快）：每个操作应用后，编译产物必须与操作前**不同**；
    2) 渲染层（真跑 ffmpeg）：抽帧比像素 / 比 PCM，证明画面和声音**真的变了**。
"""
import array
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_video_agent.agent import OPS_V1
from ai_video_agent.compiler import compile as compile_tl
from ai_video_agent.operations import apply_operations
from ai_video_agent.render import render, resolve_ffmpeg
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track

DUR = 2.0
W, H = 320, 240
# 素材：前 1s 蓝、后 1s 红，配 440Hz 正弦音轨（便于用像素/音频差分判定）
BLUE, RED = (0, 0, 255), (255, 0, 0)


def _ffprobe() -> str:
    return str(Path(resolve_ffmpeg()).with_name(
        "ffprobe.exe" if os.name == "nt" else "ffprobe"))


def _gen_clip(path: Path, dur: float = DUR) -> None:
    half = dur / 2
    subprocess.run(
        [resolve_ffmpeg(), "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=blue:s={W}x{H}:d={half}",
         "-f", "lavfi", "-i", f"color=c=red:s={W}x{H}:d={half}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        check=True, capture_output=True, text=True)


def _tl(path: Path, dur: float = DUR) -> Timeline:
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": W, "h": H}, duration=dur),
        tracks=[Track("v_main", "video"), Track("a_main", "audio"),
                Track("s_main", "subtitle")],
        clips=[Clip(id="src_video", track="v_main", sourceRef=str(path),
                    in_=0.0, out=dur, sourceIn=0.0, sourceOut=dur),
               Clip(id="src_audio", track="a_main", sourceRef=str(path),
                    in_=0.0, out=dur, sourceIn=0.0, sourceOut=dur)],
    )


def _pixel(path: str, t: float):
    r = subprocess.run(
        [resolve_ffmpeg(), "-v", "error", "-i", path, "-ss", f"{t:.3f}",
         "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True)
    b = r.stdout
    return (b[0], b[1], b[2]) if len(b) >= 3 else None


def _pcm(path: str) -> bytes:
    return subprocess.run(
        [resolve_ffmpeg(), "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout


def _pcm_diff(p1: bytes, p2: bytes) -> float:
    n = min(len(p1), len(p2)) // 2 * 2
    if n == 0:
        return -1.0
    a, b = array.array("h"), array.array("h")
    a.frombytes(p1[:n]); b.frombytes(p2[:n])
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


# 每个操作一条「能体现它」的最小操作序列（顺序即执行顺序）
OP_CASES = {
    "cut": [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 1.0}}],
    "trim": [{"type": "trim", "payload": {"targetClipId": "src_video",
                                          "in": 0.5, "out": 1.5}}],
    "add_subtitle": [{"type": "add_subtitle",
                      "payload": {"text": "开场", "start": 0.0, "end": 1.0}}],
    "add_bgm": [{"type": "add_bgm", "payload": {"sourceRef": "__SRC__",
                                                "in": 0.0, "out": DUR,
                                                "volume": 0.4}}],
    "delete_clip": [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 1.0}},
                    {"type": "delete_clip", "payload": {"targetClipId": "src_video__b"}}],
    "delete_range": [{"type": "delete_range", "payload": {"start": 0.5, "end": 1.5}}],
    "move_clip": [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 1.0}},
                  {"type": "move_clip", "payload": {"targetClipId": "src_video__b",
                                                    "toIn": 0.0}}],
    "transition": [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 1.0}},
                   {"type": "transition", "payload": {"transitionType": "fade",
                                                      "duration": 0.4,
                                                      "clips": ["src_video__a",
                                                                "src_video__b"]}}],
    "effect": [{"type": "effect", "payload": {"targetClipId": "src_video",
                                              "effectType": "grayscale",
                                              "params": {}}}],
}


class TestNoOpIsImpossible(unittest.TestCase):
    """契约里的 9 类操作，应用后必须让**编译产物**发生变化。

    这是防「空转操作」的结构性护栏：新加操作时如果编译器没消费它，这条会直接失败。
    """

    def test_every_op_changes_compiled_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            _gen_clip(src)
            base = compile_tl(_tl(src))

            for op_name in OPS_V1:
                with self.subTest(op=op_name):
                    ops = OP_CASES.get(op_name)
                    self.assertIsNotNone(ops, f"OPS_V1 里的 `{op_name}` 没有对应测试用例")
                    ops = [{**o, "payload": {k: (str(src) if v == "__SRC__" else v)
                                             for k, v in o["payload"].items()}}
                           for o in ops]
                    tl, _logs, warns = apply_operations(_tl(src), ops, strict=True)
                    self.assertEqual(warns, [], f"{op_name}: 不应出现警告 {warns}")
                    after = compile_tl(tl)
                    self.assertNotEqual(
                        (base["video_filter"], base["audio_filter"], base["ass"]),
                        (after["video_filter"], after["audio_filter"], after["ass"]),
                        f"`{op_name}` 应用后编译产物完全没变 —— 说明它对成片毫无影响")


class TestRenderedEffectIsReal(unittest.TestCase):
    """真跑 ffmpeg 出片，用像素/音频差分证明变化真实存在。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._td = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._td.name)
        cls.src = cls.dir / "src.mp4"
        _gen_clip(cls.src)
        cls.base = cls.dir / "base.mp4"
        p, msg = render(_tl(cls.src), base_dir=str(cls.dir / "base"))
        assert p, msg
        cls.base = Path(p)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._td.cleanup()

    def _render(self, ops, tag: str) -> str:
        ops = [{**o, "payload": {k: (str(self.src) if v == "__SRC__" else v)
                                 for k, v in o["payload"].items()}}
               for o in ops]
        tl, _l, w = apply_operations(_tl(self.src), ops, strict=True)
        self.assertEqual(w, [])
        p, msg = render(tl, base_dir=str(self.dir / tag))
        self.assertIsNotNone(p, f"{tag} 渲染失败：{msg}")
        return p

    def test_grayscale_actually_grays_the_frame(self) -> None:
        p = self._render(OP_CASES["effect"], "gray")
        px = _pixel(p, 0.5)
        self.assertIsNotNone(px)
        self.assertLessEqual(abs(px[0] - px[1]), 12, f"应无彩色分量：{px}")
        self.assertLessEqual(abs(px[1] - px[2]), 12, f"应无彩色分量：{px}")
        self.assertNotEqual(px, _pixel(str(self.base), 0.5), "像素应与原片不同")

    def test_transition_darkens_the_boundary(self) -> None:
        p = self._render(OP_CASES["transition"], "trans")
        at_cut = _pixel(p, 1.0)
        base_cut = _pixel(str(self.base), 1.0)
        self.assertIsNotNone(at_cut)
        # 淡入起点应接近黑（fade from black），原片该点是纯红
        self.assertLess(sum(at_cut), sum(base_cut),
                        f"转场后边界应更暗：{at_cut} vs 原片 {base_cut}")

    def test_bgm_is_really_mixed_into_the_audio(self) -> None:
        p = self._render(OP_CASES["add_bgm"], "bgm")
        d = _pcm_diff(_pcm(p), _pcm(str(self.base)))
        self.assertGreater(d, 50.0, f"BGM 未混入（PCM 平均绝对差 {d:.2f}）")

    def test_move_clip_reorder_is_visible(self) -> None:
        p = self._render(OP_CASES["move_clip"], "move")
        # 原片 0~1s 是蓝、1~2s 是红；重排后开头应是红
        px = _pixel(p, 0.5)
        self.assertIsNotNone(px)
        self.assertGreater(px[0], px[2], f"重排后开头应变成红色：{px}")

    def test_fade_in_effect_darkens_first_frame(self) -> None:
        """淡入是「画面效果」而不是「转场」——单个片段也要能淡入。"""
        ops = [{"type": "effect", "payload": {"targetClipId": "src_video",
                                              "effectType": "fade_in",
                                              "params": {"duration": 1.0}}}]
        p = self._render(ops, "fadein")
        px = _pixel(p, 0.05)
        base = _pixel(str(self.base), 0.05)
        self.assertIsNotNone(px)
        self.assertLess(sum(px), sum(base),
                        f"淡入起点应更暗：{px} vs 原片 {base}")


class TestUnsupportedIsNeverSilent(unittest.TestCase):
    """认不出的东西必须显式暴露，不能静默通过。"""

    def test_unknown_effect_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            _gen_clip(src)
            ops = [{"type": "effect", "payload": {"targetClipId": "src_video",
                                                  "effectType": "不存在的效果",
                                                  "params": {}}}]
            with self.assertRaises(Exception):
                apply_operations(_tl(src), ops, strict=True)
            tl, _logs, warns = apply_operations(_tl(src), ops, strict=False)
            self.assertTrue(warns, "非严格模式下应有可见警告")
            self.assertIn("不支持", " ".join(warns))

    def test_known_effects_all_compile(self) -> None:
        """词表里列出的每个效果都得真能编出滤镜，不能有「在册但不可用」。"""
        from ai_video_agent.compiler import AUDIO_EFFECTS, VIDEO_EFFECTS
        from ai_video_agent.compiler import audio_effect_filter, video_effect_filter
        for et in VIDEO_EFFECTS:
            self.assertIsNotNone(video_effect_filter({"effectType": et, "params": {}}),
                                 f"画面效果 `{et}` 号称支持却编不出滤镜")
        for et in AUDIO_EFFECTS:
            self.assertIsNotNone(
                audio_effect_filter({"effectType": et, "params": {}}, 2.0),
                f"音频效果 `{et}` 号称支持却编不出滤镜")

    def test_compile_reports_unsupported_list(self) -> None:
        """compile() 的返回值必须带 unsupported 字段（供上层回显）。"""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            _gen_clip(src)
            res = compile_tl(_tl(src))
            self.assertIn("unsupported", res)
            self.assertEqual(res["unsupported"], [], "干净的 timeline 不该有 unsupported")


if __name__ == "__main__":
    unittest.main()
