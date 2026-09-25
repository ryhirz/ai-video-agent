"""M2 视觉理解冒烟（方案 A 核心）：YOLOv8 真实检测 + FunASR 接线验证。

网络依赖（首次运行）：
- 演示图 bus.jpg（ultralytics 官方，~130KB，raw.githubusercontent）；
- yolov8n.pt 权重（~6MB，ultralytics GitHub releases）。
bus_clip 由 bus.jpg 循环成 2s 无声视频；YOLOv8 应检出 bus/person 等 → visual_segments 非空。
无音轨片 transcribe() 直接返回 []（不触发 FunASR 模型下载），用于验证管线接线。
"""
import os
import tempfile
import unittest
import subprocess

# 优雅降级守卫：YOLO 真实检测需要 ultralytics + torch（完整依赖）。
# 只装了基础依赖（gradio/numpy/opencv）的机器上，这两类测试应跳过而非 ERROR，
# 这样「一键测试」在轻量环境下也能干净跑过（其余 29 个纯逻辑测试不受影响）。
try:
    import ultralytics  # noqa: F401
    _HAS_ULTRALYTICS = True
except Exception:  # pragma: no cover - 仅在缺依赖时命中
    _HAS_ULTRALYTICS = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "tests", ".cache")
BUS_JPG = os.path.join(CACHE, "bus.jpg")
FF = os.path.join(HERE, "tools", "ffmpeg", "bin", "ffmpeg.exe")
BUS_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"


def _ensure_bus_jpg() -> str:
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(BUS_JPG):
        subprocess.run(["curl", "-L", "--max-time", 120, "-o", BUS_JPG, BUS_URL],
                       capture_output=True, text=True, check=True)
    return BUS_JPG


def _make_bus_clip(path: str, dur: float = 2.0) -> None:
    subprocess.run([FF, "-y", "-loop", "1", "-i", BUS_JPG, "-t", str(dur),
                    "-vf", "scale=320:240", "-pix_fmt", "yuv420p", path],
                   capture_output=True, text=True, check=True)


class TestUnderstand(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.clip = os.path.join(self.tmp, "bus.mp4")
        _ensure_bus_jpg()
        _make_bus_clip(self.clip)

    @unittest.skipUnless(_HAS_ULTRALYTICS,
                         "需要 ultralytics/torch（完整依赖）才能跑 YOLO 真实检测")
    def test_detect_objects_real(self) -> None:
        """YOLOv8 在真实帧上应检出目标（非空 visual_segments）。"""
        from ai_video_agent.ingest import build_footage_index
        from ai_video_agent.understand import detect_objects
        idx = build_footage_index(self.clip)
        segs = detect_objects(idx, weights="yolov8n.pt", conf=0.25, device="cpu")
        self.assertIsInstance(segs, list)
        self.assertGreater(len(segs), 0, "YOLOv8 应检出 bus/person 等目标")
        print("DETECTED_LABELS:", sorted({s.label for s in segs}))

    def test_transcribe_no_audio_returns_empty(self) -> None:
        """无音轨素材 transcribe() 直接返回 []，验证接线且不下权重。"""
        from ai_video_agent.ingest import build_footage_index
        from ai_video_agent.understand import transcribe
        idx = build_footage_index(self.clip)
        self.assertFalse(idx.metadata.get("has_audio"))
        self.assertEqual(transcribe(idx), [])

    @unittest.skipUnless(_HAS_ULTRALYTICS,
                         "需要 ultralytics/torch（完整依赖）才能跑 YOLO 真实检测")
    def test_build_visual_understanding(self) -> None:
        """M2 主编排：视觉段非空、无音轨转写为空。"""
        from ai_video_agent.ingest import build_footage_index
        from ai_video_agent.understand import build_visual_understanding
        idx = build_footage_index(self.clip)
        out = build_visual_understanding(idx, weights="yolov8n.pt", conf=0.25, device="cpu")
        self.assertGreater(len(out.visual_segments), 0)
        self.assertEqual(out.transcript_segments, [])


class TestAsrHallucinationGuard(unittest.TestCase):
    """FunASR 幻觉护栏（纯逻辑，不加载模型、不联网）。

    背景（实测 2026-09-23）：20s 纯音乐素材被 FunASR 转成 `'the'`，且因退化分支
    取整片时长，产出 `0.0-20.0s "the"` 这种段 → 会污染 LLM 事实块与界面。
    护栏刻意保守：只有「覆盖近全片 + 不含汉字 + 词数 ≤ 2」才丢。
    """

    @staticmethod
    def _seg(start: float, end: float, text: str):
        from ai_video_agent.footage_index import TranscriptSegment
        return TranscriptSegment(start=start, end=end, text=text)

    def test_music_hallucination_dropped(self) -> None:
        """实测场景：纯音乐 → 整片一条的 'the'，必须丢掉。"""
        from ai_video_agent.understand import _looks_like_hallucination
        seg = self._seg(0.0, 20.0, "the")
        self.assertTrue(_looks_like_hallucination(seg, 20.0, from_sentence_info=False))

    def test_two_word_hallucination_dropped(self) -> None:
        from ai_video_agent.understand import _looks_like_hallucination
        seg = self._seg(0.0, 18.0, "Thank you")
        self.assertTrue(_looks_like_hallucination(seg, 18.0, from_sentence_info=False))

    def test_chinese_narration_kept(self) -> None:
        """素材2 的真实旁白：整片一条但含汉字 → 不能被误杀。"""
        from ai_video_agent.understand import _looks_like_hallucination
        seg = self._seg(0.0, 24.0, "第一幕画面里有一辆公交车第二幕画面里是两个人")
        self.assertFalse(_looks_like_hallucination(seg, 24.0, from_sentence_info=False))

    def test_long_english_sentence_kept(self) -> None:
        """词数超阈值的长英文句不丢。"""
        from ai_video_agent.understand import _looks_like_hallucination
        seg = self._seg(0.0, 30.0, "hello everyone welcome to my channel")
        self.assertFalse(_looks_like_hallucination(seg, 30.0, from_sentence_info=False))

    def test_short_span_hallucination_like_text_kept(self) -> None:
        """不覆盖全片的短段即使在结构上像幻觉也保留 —— 它可能真是台词。"""
        from ai_video_agent.understand import _looks_like_hallucination
        seg = self._seg(3.0, 4.2, "the")
        self.assertFalse(_looks_like_hallucination(seg, 20.0, from_sentence_info=False))

    def test_sentence_info_short_segment_kept(self) -> None:
        """带真实断句时间戳的段一律保留，不看内容长短。"""
        from ai_video_agent.understand import _looks_like_hallucination
        seg = self._seg(1.0, 1.4, "好")
        self.assertFalse(_looks_like_hallucination(seg, 1.4, from_sentence_info=True))

    def test_empty_text_dropped(self) -> None:
        from ai_video_agent.understand import _looks_like_hallucination
        self.assertTrue(_looks_like_hallucination(
            self._seg(0.0, 10.0, "   "), 10.0, from_sentence_info=False))


class TestVadSpans(unittest.TestCase):
    """VAD 段解析（纯逻辑，mock 掉模型）。

    实测的返回结构：`res[0]["value"] == [[beg_ms, end_ms], ...]`。
    这里只测"把它翻译成秒区间"这一段纯逻辑，不加载任何模型、不联网。
    """

    @staticmethod
    def _patch_vad(value=None, raises=False):
        from unittest.mock import patch

        class _Fake:
            def generate(self, input=None, **kw):     # noqa: A002
                if raises:
                    raise RuntimeError("model load failed")
                return [{"key": "x", "value": value}]

        return patch("ai_video_agent.understand._get_vad", lambda device: _Fake())

    def test_parses_ms_to_seconds(self) -> None:
        """实测素材2 的真实返回：三段毫秒 → 秒。"""
        from ai_video_agent.understand import _vad_spans
        with self._patch_vad([[80, 6710], [8190, 14530], [16160, 23980]]):
            spans = _vad_spans("x.wav", "cpu", 24.0)
        self.assertEqual(len(spans), 3)
        self.assertAlmostEqual(spans[0][0], 0.08, places=3)
        self.assertAlmostEqual(spans[0][1], 6.71, places=3)
        self.assertAlmostEqual(spans[2][1], 23.98, places=3)

    def test_unclosed_segment_uses_duration(self) -> None:
        """end = -1 表示未闭合（模型没给出收尾），要用素材时长兜底，否则算出负长度。"""
        from ai_video_agent.understand import _vad_spans
        with self._patch_vad([[1000, -1]]):
            self.assertEqual(_vad_spans("x.wav", "cpu", 10.0), [(1.0, 10.0)])

    def test_end_clamped_to_duration(self) -> None:
        from ai_video_agent.understand import _vad_spans
        with self._patch_vad([[0, 99999]]):
            self.assertEqual(_vad_spans("x.wav", "cpu", 10.0), [(0.0, 10.0)])

    def test_malformed_items_skipped(self) -> None:
        """坏数据不该让整次转写崩掉：只跳过坏项。"""
        from ai_video_agent.understand import _vad_spans
        with self._patch_vad([None, [], [1000, 2000]]):
            self.assertEqual(_vad_spans("x.wav", "cpu", 10.0), [(1.0, 2.0)])

    def test_zero_length_dropped(self) -> None:
        """零长度段必须丢 —— 它会变成零长度删除区间，下游会报错/静默跳过。"""
        from ai_video_agent.understand import _vad_spans
        with self._patch_vad([[500, 500]]):
            self.assertEqual(_vad_spans("x.wav", "cpu", 10.0), [])

    def test_empty_value(self) -> None:
        """无语音素材：VAD 可能返回空 value。"""
        from ai_video_agent.understand import _vad_spans
        with self._patch_vad([]):
            self.assertEqual(_vad_spans("x.wav", "cpu", 10.0), [])

    def test_model_failure_falls_back_to_empty(self) -> None:
        """VAD 加载/推理失败必须退化为 []（调用方走整片兜底），不能把转写整体搞崩。"""
        from ai_video_agent.understand import _vad_spans
        with self._patch_vad(raises=True):
            self.assertEqual(_vad_spans("x.wav", "cpu", 10.0), [])


class TestSilentFootageGuard(unittest.TestCase):
    """整体判据（纯逻辑）：识别"无语音素材被 VAD 切细后逐段幻觉"。

    用例数据取自 2026-09-23 的真实实测输出（VAD 阈值 300ms）。
    """

    @staticmethod
    def _seg(start, end, text):
        from ai_video_agent.footage_index import TranscriptSegment
        return TranscriptSegment(start=start, end=end, text=text)

    def _cands(self, *rows):
        return [(self._seg(s, e, t), False) for (s, e, t) in rows]

    def test_music_6_segments_dropped(self) -> None:
        """素材3 的真实输出：纯音乐被切 6 段，逐段幻觉成拟声词，覆盖全片 → 整批丢。"""
        from ai_video_agent.understand import _looks_like_silent_footage
        cands = self._cands(
            (0.00, 3.41, "嗯嗯"), (3.41, 7.43, "呜啦啦"), (7.43, 11.55, "呜呜呜"),
            (11.55, 15.44, "嗯"), (15.44, 19.47, "嗯"), (19.47, 19.98, "嗯"))
        self.assertTrue(_looks_like_silent_footage(cands, 20.0))

    def test_chinese_narration_11_segments_kept(self) -> None:
        """素材2 的真实输出：11 段真旁白 —— 有长段（9 字），不能被整体误杀。"""
        from ai_video_agent.understand import _looks_like_silent_footage
        cands = self._cands(
            (0.08, 1.28, "第一幕"), (1.28, 3.35, "画面里有一辆公交车"),
            (4.15, 5.27, "第二幕"), (5.27, 6.71, "画面里是两个人"),
            (8.15, 9.34, "第三幕"), (9.34, 11.28, "又出现了一辆公交车"),
            (12.14, 13.34, "第四幕"), (13.34, 14.53, "还是这两个人"),
            (16.16, 17.36, "第五幕"), (17.36, 19.05, "公交车再次出现"),
            (20.16, 22.75, "第六幕画面回到人物"))
        self.assertFalse(_looks_like_silent_footage(cands, 24.0))

    def test_few_segments_not_triggered(self) -> None:
        """段数 < 3 不启用整体判据（单段素材走 covers_all 老路）。"""
        from ai_video_agent.understand import _looks_like_silent_footage
        cands = self._cands((0.0, 20.0, "嗯"), (0.0, 20.0, "嗯"))
        self.assertFalse(_looks_like_silent_footage(cands, 20.0))

    def test_short_phrases_without_full_coverage_kept(self) -> None:
        """短句但不覆盖全片 → 保留（可能真是零散短台词）。"""
        from ai_video_agent.understand import _looks_like_silent_footage
        cands = self._cands((0.0, 2.0, "嗯"), (5.0, 7.0, "好"), (10.0, 12.0, "嗯"))
        self.assertFalse(_looks_like_silent_footage(cands, 20.0))

    def test_one_long_segment_disqualifies(self) -> None:
        """只要有一段是长文本，整批就不能按"无语音"丢弃。"""
        from ai_video_agent.understand import _looks_like_silent_footage
        cands = self._cands(
            (0.0, 6.0, "嗯"), (6.0, 12.0, "好"), (12.0, 18.0, "这里有一句正常长度的台词"))
        self.assertFalse(_looks_like_silent_footage(cands, 20.0))

    def test_union_len_merges_overlaps(self) -> None:
        """并集长度要合并重叠区间，否则重叠段的覆盖率会被高估。"""
        from ai_video_agent.understand import _union_len
        self.assertAlmostEqual(_union_len([(0.0, 5.0), (3.0, 8.0)]), 8.0, places=3)
        self.assertAlmostEqual(_union_len([(0.0, 2.0), (4.0, 6.0)]), 4.0, places=3)
        self.assertAlmostEqual(_union_len([]), 0.0, places=3)

    def test_overlapping_segments_do_not_fake_coverage(self) -> None:
        """三段全挤在前 7s，即使相互重叠也不该被当成"覆盖全片"。"""
        from ai_video_agent.understand import _looks_like_silent_footage
        cands = self._cands((0.0, 6.0, "嗯"), (0.5, 6.5, "嗯"), (1.0, 7.0, "好"))
        self.assertFalse(_looks_like_silent_footage(cands, 20.0))


class TestAsrSegmentTexts(unittest.TestCase):
    """逐段转写的两级兜底（纯逻辑，mock 掉模型与 ffmpeg）。

    优化路径是"内存切片 + 批量转写"（实测 11 段 5.02s → 1.36s），但**安全**比快重要：
    任何"条数对不上"的情况都必须退回逐条，否则时间戳会错位到别的段。
    """

    SPANS = [(0.0, 1.0), (1.0, 2.0)]

    @staticmethod
    def _fake_am(batch_texts=None, batch_raises=False, single_raises=False):
        class _AM:
            def __init__(self):
                self.batch_calls = 0
                self.single_calls = 0

            def generate(self, input=None, **kw):       # noqa: A002
                if isinstance(input, list):             # 批量：输入是数组列表
                    self.batch_calls += 1
                    if batch_raises:
                        raise RuntimeError("batch boom")
                    return [{"text": t} for t in (batch_texts or [])]
                self.single_calls += 1                  # 逐条：输入是文件路径
                if single_raises:
                    raise RuntimeError("single boom")
                return [{"text": f"单:{input}"}]
        return _AM()

    def test_batch_path_used_when_consistent(self) -> None:
        """条数一致时走批量，且完全不碰 ffmpeg。"""
        import numpy as np
        from unittest.mock import patch
        from ai_video_agent.understand import _asr_segment_texts
        am = self._fake_am(batch_texts=["甲", "乙"])
        with patch("ai_video_agent.understand._read_wav_mono16k",
                   return_value=np.zeros(32000, dtype="float32")), \
             patch("ai_video_agent.understand._slice_wav") as sw:
            texts = _asr_segment_texts(am, "x.wav", "ff", self.SPANS)
        self.assertEqual(texts, ["甲", "乙"])
        self.assertEqual((am.batch_calls, am.single_calls), (1, 0))
        sw.assert_not_called()

    def test_falls_back_on_count_mismatch(self) -> None:
        """批量返回条数 != 输入条数 → 必须退回逐条（否则时间戳会错位）。"""
        import numpy as np
        from unittest.mock import patch
        from ai_video_agent.understand import _asr_segment_texts
        am = self._fake_am(batch_texts=["只有一条"])
        with patch("ai_video_agent.understand._read_wav_mono16k",
                   return_value=np.zeros(32000, dtype="float32")), \
             patch("ai_video_agent.understand._slice_wav", side_effect=lambda ff, s, a, b: "p"):
            texts = _asr_segment_texts(am, "x.wav", "ff", self.SPANS)
        self.assertEqual(am.batch_calls, 1)
        self.assertEqual(am.single_calls, 2, "条数不符时应逐条兜底")
        self.assertEqual(texts, ["单:p", "单:p"])

    def test_falls_back_when_soundfile_unavailable(self) -> None:
        """读不出数组（soundfile 缺失/采样率不符）→ 直接走 ffmpeg 逐条路径。"""
        from unittest.mock import patch
        from ai_video_agent.understand import _asr_segment_texts
        am = self._fake_am()
        with patch("ai_video_agent.understand._read_wav_mono16k", return_value=None), \
             patch("ai_video_agent.understand._slice_wav", side_effect=lambda ff, s, a, b: "p"):
            texts = _asr_segment_texts(am, "x.wav", "ff", self.SPANS)
        self.assertEqual(am.batch_calls, 0)
        self.assertEqual(am.single_calls, 2)
        self.assertEqual(texts, ["单:p", "单:p"])

    def test_falls_back_when_batch_raises(self) -> None:
        """批量调用抛异常 → 退回逐条，而不是整段转写失败。"""
        import numpy as np
        from unittest.mock import patch
        from ai_video_agent.understand import _asr_segment_texts
        am = self._fake_am(batch_raises=True)
        with patch("ai_video_agent.understand._read_wav_mono16k",
                   return_value=np.zeros(32000, dtype="float32")), \
             patch("ai_video_agent.understand._slice_wav", side_effect=lambda ff, s, a, b: "p"):
            texts = _asr_segment_texts(am, "x.wav", "ff", self.SPANS)
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts, ["单:p", "单:p"])

    def test_single_segment_failure_does_not_break_others(self) -> None:
        """逐条兜底路径里，单段失败只留空串，不能拖垮其余段。"""
        from unittest.mock import patch
        from ai_video_agent.understand import _asr_segment_texts
        am = self._fake_am()
        calls = {"n": 0}

        def flaky(ff, s, a, b):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("slice boom")
            return "p"

        with patch("ai_video_agent.understand._read_wav_mono16k", return_value=None), \
             patch("ai_video_agent.understand._slice_wav", side_effect=flaky):
            texts = _asr_segment_texts(am, "x.wav", "ff", self.SPANS)
        self.assertEqual(texts, ["", "单:p"])


if __name__ == "__main__":
    unittest.main()
