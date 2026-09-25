"""契约单测：验证冻结基线 v1.0.0 的确定性、不变量与校验。

运行：python -m unittest tests.test_contract -v
"""
import unittest

from ai_video_agent.timeline import Timeline, Clip, Track, Metadata, Subtitle
from ai_video_agent.operations import apply_operation
from ai_video_agent.compiler import compile, build_ass_content


def sample_timeline() -> Timeline:
    return Timeline(
        metadata=Metadata(duration=16.0),
        tracks=[
            Track("v_main", "video"),
            Track("a_main", "audio"),
            Track("s_main", "subtitle"),
        ],
        clips=[
            Clip("clip_001", "v_main", "media_001", 0.0, 8.0, 12.0, 20.0),
            Clip("clip_002", "v_main", "media_002", 8.0, 16.0, 0.0, 8.0),
        ],
    )


class TestDeterminism(unittest.TestCase):
    def test_compile_is_deterministic(self):
        tl = sample_timeline()
        self.assertEqual(compile(tl)["command"], compile(tl)["command"])
        self.assertEqual(compile(tl)["ass"], compile(tl)["ass"])

    def test_ass_is_deterministic(self):
        tl = sample_timeline()
        nt = apply_operation(tl, {"id": "op3", "type": "add_subtitle", "payload": {
            "track": "s_main", "text": "你好世界", "start": 1.0, "end": 4.5,
            "style": {"position": "bottom-center"}}})
        self.assertEqual(build_ass_content(nt), build_ass_content(nt))


class TestInvariants(unittest.TestCase):
    def test_cut_splits_with_invariant(self):
        tl = sample_timeline()
        nt = apply_operation(tl, {"id": "op1", "type": "cut",
                                  "payload": {"targetClipId": "clip_001", "at": 4.0}})
        a = next(c for c in nt.clips if c.id == "clip_001__a")
        b = next(c for c in nt.clips if c.id == "clip_001__b")
        # 不变量 (out-in) === (sourceOut-sourceIn)
        self.assertEqual((a.out - a.in_), (a.sourceOut - a.sourceIn))
        self.assertEqual((b.out - b.in_), (b.sourceOut - b.sourceIn))
        self.assertEqual(a.in_, 0.0)
        self.assertEqual(b.out, 8.0)
        self.assertEqual(len(nt.clips), 3)  # 原 1 个被拆成 2 个

    def test_trim_adjusts_source(self):
        tl = sample_timeline()
        nt = apply_operation(tl, {"id": "op2", "type": "trim",
                                  "payload": {"targetClipId": "clip_001", "in": 2.0, "out": 6.0}})
        c = next(x for x in nt.clips if x.id == "clip_001")
        self.assertEqual(c.in_, 2.0)
        self.assertEqual(c.out, 6.0)
        self.assertEqual(c.sourceIn, 14.0)    # 裁头 2s -> sourceIn +2
        self.assertEqual(c.sourceOut, 18.0)   # 裁尾 2s -> sourceOut -2
        self.assertEqual((c.out - c.in_), (c.sourceOut - c.sourceIn))

    def test_subtitle_appended(self):
        tl = sample_timeline()
        nt = apply_operation(tl, {"id": "op3", "type": "add_subtitle", "payload": {
            "track": "s_main", "text": "你好", "start": 1.0, "end": 4.5,
            "style": {"position": "bottom-center"}}})
        self.assertEqual(len(nt.subtitles), 1)
        ass = build_ass_content(nt)
        # 覆盖标签必须用 {} 包裹：{\an2}你好；裸 \an2 会被当作正文显示出来
        self.assertIn("{\\an2}", ass)
        self.assertIn("{\\an2}你好", ass)


class TestSubtitleFitsFrame(unittest.TestCase):
    """字幕必须落在画面内。

    回归的是一条实测缺陷：原 ASS 既没写 `PlayResX/PlayResY`、`WrapStyle` 又是 2（只按 `\\N` 断行），
    libass 便按 **384×288** 解释脚本坐标 —— 于是声明的 `Fontsize: 36` 实际是"画面高度的 12.5%"，
    横屏 1280×720 碰巧装得下，**竖屏 720×1280 横向溢出、两端各被裁掉约一个字**（已用抽帧眼见为实）。

    注意这个 bug 的隐蔽之处：**光看声明的字号是看不出来的**（36 看着很小），
    必须先把坐标系（PlayResX/Y）钉死，声明的字号才有确定含义。所以最关键的一条断言是
    `test_playres_matches_resolution` —— 它在旧实现上必然失败。
    """

    TEXT = "竖屏测试字幕在这里"

    def _tl(self, w: int, h: int) -> Timeline:
        return Timeline(
            metadata=Metadata(duration=10.0, resolution={"w": w, "h": h}),
            tracks=[Track("v_main", "video"), Track("a_main", "audio"),
                    Track("s_main", "subtitle")],
            subtitles=[Subtitle(id="sub1", track="s_main", text=self.TEXT,
                                start=1.0, end=4.0)],
        )

    @staticmethod
    def _style_field(ass: str, name: str) -> str:
        fmt = next(l for l in ass.splitlines() if l.startswith("Format: Name"))
        names = [x.strip() for x in fmt.split(":", 1)[1].split(",")]
        style = next(l for l in ass.splitlines() if l.startswith("Style: Default"))
        return style.split(":", 1)[1].strip().split(",")[names.index(name)]

    def test_playres_matches_resolution(self) -> None:
        """坐标系必须显式声明为 timeline 分辨率（旧实现完全没写 → 竖屏溢出）。"""
        for (w, h) in ((1280, 720), (720, 1280), (1920, 1080)):
            with self.subTest(res=f"{w}x{h}"):
                ass = build_ass_content(self._tl(w, h))
                self.assertIn(f"PlayResX: {w}", ass)
                self.assertIn(f"PlayResY: {h}", ass)

    def test_wrap_style_allows_wrapping(self) -> None:
        """WrapStyle 2 = 只按 `\\N` 断行；长字幕不折行就会直接溢出被裁。"""
        self.assertIn("WrapStyle: 0", build_ass_content(self._tl(720, 1280)))

    def test_text_fits_width(self) -> None:
        """在钉死的坐标系下，按字号估算的文本宽度必须小于可用宽度。"""
        for (w, h) in ((1280, 720), (720, 1280), (1920, 1080)):
            with self.subTest(res=f"{w}x{h}"):
                ass = build_ass_content(self._tl(w, h))
                font = float(self._style_field(ass, "Fontsize"))
                ml = float(self._style_field(ass, "MarginL"))
                mr = float(self._style_field(ass, "MarginR"))
                width = font * len(self.TEXT)      # 汉字按 1em 保守估算
                avail = w - ml - mr
                self.assertLessEqual(
                    width, avail,
                    f"{w}x{h}：字幕约 {width:.0f}px 超出可用宽 {avail:.0f}px，会被裁掉")

    def test_font_size_scales_with_resolution(self) -> None:
        """不同分辨率下字号必须跟着变（写死数值 = 换分辨率就崩）。"""
        small = float(self._style_field(build_ass_content(self._tl(720, 1280)), "Fontsize"))
        large = float(self._style_field(build_ass_content(self._tl(1920, 1080)), "Fontsize"))
        self.assertNotEqual(small, large)

    def test_style_format_is_standard_v4plus(self) -> None:
        """Style 的 Format 用标准 V4+ 23 字段。

        原实现少了 3 个颜色/编码字段，还把 `Outline`/`Shadow` 各声明了两次
        （第 5、6 个字段填的是颜色值），解析结果依赖 libass 对重复字段的处理细节，很脆。
        """
        ass = build_ass_content(self._tl(1280, 720))
        fmt = next(l for l in ass.splitlines() if l.startswith("Format: Name"))
        names = [x.strip() for x in fmt.split(":", 1)[1].split(",")]
        self.assertEqual(names[:4], ["Name", "Fontname", "Fontsize", "PrimaryColour"])
        self.assertIn("Encoding", names)
        self.assertEqual(names.count("Outline"), 1)
        self.assertEqual(names.count("Shadow"), 1)


class TestValidation(unittest.TestCase):
    def test_cut_out_of_range_raises(self):
        tl = sample_timeline()
        with self.assertRaises(ValueError):
            apply_operation(tl, {"id": "op4", "type": "cut",
                                 "payload": {"targetClipId": "clip_001", "at": 8.0}})

    def test_unsupported_op_raises(self):
        tl = sample_timeline()
        with self.assertRaises(ValueError):
            apply_operation(tl, {"id": "op5", "type": "frobnicate", "payload": {}})

    def test_subtitle_out_of_range_raises(self):
        tl = sample_timeline()
        with self.assertRaises(ValueError):
            apply_operation(tl, {"id": "op6", "type": "add_subtitle", "payload": {
                "track": "s_main", "text": "x", "start": -1.0, "end": 4.5}})

    def test_delete_drops_transition(self):
        tl = sample_timeline()
        tl.transitions.append(__import__("ai_video_agent.timeline", fromlist=["Transition"]).Transition(
            "trans_001", "fade", 1.0, ["clip_001", "clip_002"]))
        nt = apply_operation(tl, {"id": "op7", "type": "delete_clip",
                                  "payload": {"targetClipId": "clip_001"}})
        self.assertEqual(len(nt.clips), 1)
        self.assertEqual(len(nt.transitions), 0)  # Q1：引用被删 clip 的转场一并丢弃


if __name__ == "__main__":
    unittest.main()
