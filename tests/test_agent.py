"""M4 编辑智能体冒烟：NL → 结构化操作 → 确定性编译（与 M3 operations 联调）。

默认走 RuleBasedPlanner（离线确定性）；同时验证 LLMProvider 热插拔点（fake provider）。
"""
import unittest

from ai_video_agent.timeline import Timeline, Track, Clip, Metadata
from ai_video_agent.footage_index import FootageIndex, VisualSegment
from ai_video_agent.ingest import build_initial_timeline
from ai_video_agent.operations import apply_operation, apply_operations
from ai_video_agent.agent import EditingAgent, RuleBasedPlanner, LLMProvider


def _tl(dur: float = 10.0) -> Timeline:
    """构造单 src_video(0..dur) 的初版 Timeline，复用 M3 桥接函数。"""
    idx = FootageIndex(
        mediaId="x", filepath="dummy.mp4",
        metadata={"duration": dur, "fps": 30, "width": 320, "height": 240,
                  "has_audio": False},
    )
    return build_initial_timeline(idx)


class _FakeProvider:
    """模拟 LLMProvider：固定返回一条 cut 操作，验证热插拔路径。"""
    def complete(self, prompt: str, *, system=None) -> str:
        return '[{"type":"cut","payload":{"targetClipId":"src_video","at":1.0}}]'


class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = EditingAgent()          # 默认规则 Planner

    # --- 基础指令映射 ---
    def test_cut_at(self) -> None:
        ops = self.agent.plan("在第2秒切一刀", _tl(10))
        self.assertEqual(ops, [{"type": "cut",
                                "payload": {"targetClipId": "src_video", "at": 2.0}}])
        new = apply_operation(_tl(10), ops[0])
        self.assertEqual(len(new.clips), 2)   # 一刀两片

    def test_trim_head(self) -> None:
        ops = self.agent.plan("去掉前2秒", _tl(10))
        self.assertEqual(ops[0]["type"], "trim")
        self.assertEqual(ops[0]["payload"]["in"], 2.0)
        new = apply_operation(_tl(10), ops[0])
        self.assertEqual(new.clips[0].in_, 2.0)

    def test_add_subtitle(self) -> None:
        ops = self.agent.plan('加字幕"你好"', _tl(10))
        self.assertEqual(ops[0]["type"], "add_subtitle")
        self.assertEqual(ops[0]["payload"]["text"], "你好")
        self.assertEqual(ops[0]["payload"]["start"], 0.0)
        self.assertEqual(ops[0]["payload"]["end"], 5.0)   # 默认取前 5 秒
        new = apply_operation(_tl(10), ops[0])
        self.assertEqual(len(new.subtitles), 1)

    # --- 语义检索：NL → 通过 FootageIndex 反查时间区间 → 操作（M2+M4 价值闭环）---
    def test_remove_label_via_index(self) -> None:
        index = FootageIndex(
            mediaId="x", filepath="dummy.mp4",
            metadata={"duration": 10.0, "fps": 30, "width": 320, "height": 240,
                      "has_audio": False},
            visual_segments=[VisualSegment(label="bus", start=3.0, end=5.0,
                                           source="yolo")],
        )
        ops = self.agent.plan("删掉有bus的片段", _tl(10), index)
        # 期望：对检出的每个 bus 区间各一条 delete_range（按素材秒数删除，
        # 不依赖 cut 派生 id —— 旧的三步写法在多区间时会被静默跳过）
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["type"], "delete_range")
        self.assertEqual(ops[0]["payload"]["start"], 3.0)
        self.assertEqual(ops[0]["payload"]["end"], 5.0)

        # 逐步 apply，验证 bus 区间 (3,5) 被剔除
        tl = _tl(10)
        for op in ops:
            tl = apply_operation(tl, op)
        video = [c for c in tl.clips if c.track == "v_main"]
        self.assertEqual(len(video), 2)
        for c in video:
            self.assertFalse(3.0 < c.in_ and c.out <= 5.0, "bus 区间应被删除")
        spans = [(c.in_, c.out) for c in video]
        self.assertIn((0.0, 3.0), spans)
        self.assertIn((5.0, 10.0), spans)

    # --- LLMProvider 热插拔点 ---
    def test_provider_plugged_in(self) -> None:
        agent = EditingAgent(provider=_FakeProvider())   # type: ignore[arg-type]
        ops = agent.plan("随便说点什么", _tl(10))
        self.assertEqual(ops, [{"type": "cut",
                                "payload": {"targetClipId": "src_video", "at": 1.0}}])

    # --- 确定性：同指令同结果 ---
    def test_determinism(self) -> None:
        a = self.agent.plan("在第2秒切一刀", _tl(10))
        b = self.agent.plan("在第2秒切一刀", _tl(10))
        self.assertEqual(a, b)

    # --- 便捷 execute 整链 ---
    def test_execute_end_to_end(self) -> None:
        new = self.agent.execute("在第2秒切一刀", _tl(10))
        self.assertEqual(len(new.clips), 2)

    # --- 画面效果类指令（离线也能用，不需要云端 Key）---
    def test_effect_commands(self) -> None:
        cases = {"黑白": "grayscale", "灰度": "grayscale", "复古": "sepia",
                 "反色": "invert", "镜像": "mirror", "模糊": "blur",
                 "淡入": "fade_in", "淡出": "fade_out"}
        for nl, want in cases.items():
            with self.subTest(nl=nl):
                ops = self.agent.plan(nl, _tl(10))
                self.assertTrue(ops, f"{nl} 应产出操作")
                eff = [o for o in ops if o["type"] == "effect"]
                self.assertEqual(len(eff), 1, f"{nl} 应恰好一条 effect：{ops}")
                self.assertEqual(eff[0]["payload"]["effectType"], want)
                self.assertEqual(eff[0]["payload"]["targetClipId"], "src_video")

    def test_effect_commands_compile(self) -> None:
        """规则 Planner 产出的效果必须真能编出滤镜（否则就是空转）。"""
        from ai_video_agent.compiler import compile as compile_tl
        for nl in ("黑白", "模糊", "淡入", "淡出"):
            with self.subTest(nl=nl):
                new = self.agent.execute(nl, _tl(10))
                self.assertTrue(new.clips[0].effects, f"{nl} 应挂上效果")
                res = compile_tl(new)
                self.assertEqual(res["unsupported"], [], f"{nl} 不应有 unsupported")
                self.assertNotEqual(res["video_filter"], "", f"{nl} 应产出滤镜")


class TestSemanticRemoveCoversWholeShot(unittest.TestCase):
    """语义删除的区间必须以**镜头**为单位，不能只取关键帧跨度。

    回归的是一条实测出来、且用户直接可以看到的缺陷：
      源里 0–5s 的公交镜头只在 0.5/2/3/4s 命中了关键帧 → 旧实现给出区间 (0.5, 4.0)，
      删完成片里**仍留着 0–0.5s 与 4–5s 的公交画面**，用户看到的还是"bus 没删干净"；
      更糟的是"只被一个关键帧命中"的镜头会得到零长度区间 (t, t)，
      `delete_range` 要求 start < end → 严格模式报错、UI 的非严格模式则**静默跳过**。
    """

    SCENES = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0},
              {"start": 8.0, "end": 12.0}, {"start": 12.0, "end": 16.0},
              {"start": 16.0, "end": 20.0}, {"start": 20.0, "end": 24.0}]

    def test_single_keyframe_hit_expands_to_scene(self) -> None:
        """只命中一个关键帧 → 应外扩成整个镜头，绝不能是零长度区间。"""
        from ai_video_agent.understand import _label_ranges_from_times
        got = _label_ranges_from_times(self.SCENES, [2.0], gap=4.0)
        self.assertEqual(got, [(0.0, 4.0)])

    def test_keyframes_in_one_scene_give_one_shot_range(self) -> None:
        from ai_video_agent.understand import _label_ranges_from_times
        self.assertEqual(_label_ranges_from_times(self.SCENES, [8.0, 10.0], 4.0),
                         [(8.0, 12.0)])

    def test_adjacent_scenes_merge_into_one_range(self) -> None:
        """同一幕被切成两个镜头时（关键帧都在同一画面），命中的相邻镜头应并成一段。"""
        from ai_video_agent.understand import _label_ranges_from_times
        scenes = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 5.0},
                  {"start": 5.0, "end": 10.0}]
        self.assertEqual(_label_ranges_from_times(scenes, [0.5, 2.0, 3.0, 4.0], 4.0),
                         [(0.0, 5.0)])

    def test_non_adjacent_scenes_stay_separate(self) -> None:
        from ai_video_agent.understand import _label_ranges_from_times
        self.assertEqual(_label_ranges_from_times(self.SCENES, [2.0, 18.0], 4.0),
                         [(0.0, 4.0), (16.0, 20.0)])

    def test_boundary_time_belongs_to_next_scene(self) -> None:
        """t=4.0 取到的是下一镜的画面（与 ffmpeg 抽帧语义一致），应归下一镜。"""
        from ai_video_agent.understand import _scene_index_at
        self.assertEqual(_scene_index_at(self.SCENES, 4.0), 1)
        self.assertEqual(_scene_index_at(self.SCENES, 3.999), 0)

    def test_no_scene_info_still_nonzero(self) -> None:
        """没有镜头信息时退化为关键帧跨度，但必须保证非零长度。"""
        from ai_video_agent.understand import _label_ranges_from_times
        got = _label_ranges_from_times([], [2.0], 4.0)
        self.assertEqual(len(got), 1)
        self.assertLess(got[0][0], got[0][1], "不能产出零长度区间")

    def test_remove_label_leaves_no_residual_shot(self) -> None:
        """端到端：删掉有 bus 的镜头后，成片里不能再有任何 bus 镜头的时间区间。"""
        bus_shots = [(0.0, 4.0), (8.0, 12.0), (16.0, 20.0)]
        index = FootageIndex(
            mediaId="x", filepath="dummy.mp4",
            metadata={"duration": 24.0, "fps": 25, "width": 320, "height": 240,
                      "has_audio": True},
            scenes=self.SCENES,
            visual_segments=[VisualSegment(label="bus", start=s, end=e, source="yolo")
                             for (s, e) in bus_shots],
        )
        tl = build_initial_timeline(index)
        ops = RuleBasedPlanner().plan("删掉有bus的片段", tl, index)
        self.assertEqual(len(ops), 3)
        for op in ops:
            self.assertEqual(op["type"], "delete_range")
            self.assertLess(op["payload"]["start"], op["payload"]["end"],
                            "区间必须非零长度")
        self.assertEqual([(o["payload"]["start"], o["payload"]["end"]) for o in ops],
                         bus_shots)

        tl2, _logs, warns = apply_operations(tl, ops, strict=True)
        self.assertEqual(warns, [])
        kept = sorted((c.sourceIn, c.sourceOut)
                      for c in tl2.clips if c.track == "v_main")
        self.assertEqual(kept, [(4.0, 8.0), (12.0, 16.0), (20.0, 24.0)])

        # 关键断言：保留的区间与任何 bus 镜头都不得有交集
        for (ks, ke) in kept:
            for (bs, be) in bus_shots:
                self.assertFalse(ks < be and ke > bs,
                                 f"成片保留了 bus 画面：{(ks, ke)} 与 {(bs, be)} 重叠")

        # 音频必须跟着删（A/V 对齐铁律）
        audio = sorted((c.sourceIn, c.sourceOut)
                       for c in tl2.clips if c.track == "a_main")
        self.assertEqual(audio, kept)


class TestDeleteBySpeech(unittest.TestCase):
    """按**台词**删除：区间同样必须以镜头为单位（与视觉语义删除同一条铁律）。

    台词段的时间戳来自 M2 的 VAD 分段，实测粒度可达"半句"（素材2 六幕旁白被切成 11 段，
    段边界与幕边界对齐）。**但正因为细到半句，直接拿它去删就会留下半个镜头** ——
    所以 `_speech_ranges` 必须把台词区间外扩到它覆盖的镜头。
    """

    SCENES = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0},
              {"start": 8.0, "end": 12.0}, {"start": 12.0, "end": 16.0},
              {"start": 16.0, "end": 20.0}, {"start": 20.0, "end": 24.0}]

    # 素材2 的真实转写输出（VAD 阈值 300ms 时实测得到）
    SPEECH = [(0.08, 1.28, "第一幕"), (1.28, 3.35, "画面里有一辆公交车"),
              (4.15, 5.27, "第二幕"), (5.27, 6.71, "画面里是两个人"),
              (8.15, 9.34, "第三幕"), (9.34, 11.28, "又出现了一辆公交车"),
              (12.14, 13.34, "第四幕"), (13.34, 14.53, "还是这两个人"),
              (16.16, 17.36, "第五幕"), (17.36, 19.05, "公交车再次出现"),
              (20.16, 22.75, "第六幕画面回到人物")]

    def _index(self, speech=None, visuals=None):
        from ai_video_agent.footage_index import TranscriptSegment
        idx = FootageIndex(
            mediaId="x", filepath="dummy.mp4",
            metadata={"duration": 24.0, "fps": 25, "width": 1280, "height": 720,
                      "has_audio": True})
        idx.scenes = list(self.SCENES)
        idx.transcript_segments = [TranscriptSegment(start=s, end=e, text=t)
                                   for (s, e, t) in (speech if speech is not None
                                                     else self.SPEECH)]
        idx.visual_segments = list(visuals or [])
        return idx

    def test_speech_range_expands_to_shot(self) -> None:
        """半句台词 [8.15,9.34] 应外扩成整个镜头 (8,12)，而不是原样返回。"""
        from ai_video_agent.agent import _speech_ranges
        got = _speech_ranges(self._index(), "第三幕")
        self.assertEqual(got, [(8.0, 12.0)])

    def test_same_shot_deduped(self) -> None:
        """同一镜头内的两句台词（"第三幕" / "又出现了一辆公交车"）只应给一个区间。"""
        from ai_video_agent.agent import _speech_ranges
        got = _speech_ranges(self._index(), "第三幕")
        hit2 = _speech_ranges(self._index(), "公交车")
        self.assertEqual(got, [(8.0, 12.0)])
        # "公交车" 出现在幕1/幕3/幕5 的旁白里 → 三个镜头（与视觉上的公交镜头一致）
        self.assertEqual(hit2, [(0.0, 4.0), (8.0, 12.0), (16.0, 20.0)])

    def test_no_keyword_match_gives_nothing(self) -> None:
        from ai_video_agent.agent import _speech_ranges
        self.assertEqual(_speech_ranges(self._index(), "不存在的词"), [])

    def test_delete_by_speech_keyword(self) -> None:
        """端到端：'删掉说公交车的片段' → 3 条 delete_range，且删除后不留残余。"""
        idx = self._index()
        tl = build_initial_timeline(idx)
        ops = RuleBasedPlanner().plan("删掉说公交车的片段", tl, idx)
        self.assertEqual([(o["payload"]["start"], o["payload"]["end"]) for o in ops],
                         [(0.0, 4.0), (8.0, 12.0), (16.0, 20.0)])
        tl2, _logs, warns = apply_operations(tl, ops, strict=True)
        self.assertEqual(warns, [])
        kept = sorted((c.sourceIn, c.sourceOut)
                      for c in tl2.clips if c.track == "v_main")
        self.assertEqual(kept, [(4.0, 8.0), (12.0, 16.0), (20.0, 24.0)])
        audio = sorted((c.sourceIn, c.sourceOut)
                       for c in tl2.clips if c.track == "a_main")
        self.assertEqual(audio, kept, "音频必须跟着删（A/V 对齐铁律）")

    def test_visual_takes_priority_when_available(self) -> None:
        """视觉标签有匹配时走视觉（画面是主要编辑对象），不回退台词。"""
        visuals = [VisualSegment(label="bus", start=0.0, end=4.0, confidence=0.9,
                                 source="yolo")]
        idx = self._index(visuals=visuals)
        tl = build_initial_timeline(idx)
        ops = RuleBasedPlanner().plan("删掉有公交车的片段", tl, idx)
        self.assertEqual([(o["payload"]["start"], o["payload"]["end"]) for o in ops],
                         [(0.0, 4.0)], "只该删视觉命中的幕1，而不是台词命中的三幕")

    def test_falls_back_to_speech_when_no_visual(self) -> None:
        """视觉没匹配到（如标签是英文 bus 而用户说'公交车'的下位词）→ 回退台词检索。"""
        idx = self._index(visuals=[])
        tl = build_initial_timeline(idx)
        ops = RuleBasedPlanner().plan("删掉有公交车的片段", tl, idx)
        self.assertEqual([(o["payload"]["start"], o["payload"]["end"]) for o in ops],
                         [(0.0, 4.0), (8.0, 12.0), (16.0, 20.0)])

    def test_facts_block_groups_speech_by_shot(self) -> None:
        """事实块要把台词按镜头聚合 —— 模型拿到的就是可直接删除的区间。"""
        from ai_video_agent.agent import facts_block
        txt = facts_block(self._index())
        # 断言"聚合"这个标识本身，而不是"按镜头聚合"这个子串 ——
        # 退化分支里写着"未按镜头聚合"，它**包含**该子串，弱断言会在退化时照样通过。
        self.assertIn("台词（**按镜头聚合", txt)
        self.assertNotIn("未按镜头聚合", txt)
        self.assertIn("镜头 8.00-12.00s", txt)
        # 聚合后不应再出现半句台词的小时间戳
        self.assertNotIn("8.15-9.34s", txt)

    def test_facts_block_without_scenes_falls_back(self) -> None:
        """没有镜头信息时退化为逐条列台词，而不是丢掉台词 —— 且必须**自报家门**。"""
        from ai_video_agent.agent import facts_block
        idx = self._index()
        idx.scenes = []
        txt = facts_block(idx)
        self.assertIn("8.15-9.34s", txt)
        # 退化分支必须显式标注，否则提示词会与 schema 规则 3b（"事实里已按镜头聚合"）自相矛盾
        self.assertIn("台词（**未按镜头聚合", txt)
        self.assertNotIn("镜头 8.00-12.00s", txt)

    def test_facts_block_falls_back_when_speech_misses_all_shots(self) -> None:
        """有镜头、但台词与任何镜头都不重叠 → 同样走退化分支，且标注一致。"""
        from ai_video_agent.agent import facts_block
        from ai_video_agent.footage_index import TranscriptSegment
        idx = self._index()
        # 台词全落在镜头区间之外（如镜头被截断/时间基准不一致）
        idx.transcript_segments = [TranscriptSegment(start=30.0, end=31.0, text="界外台词")]
        txt = facts_block(idx)
        self.assertIn("未按镜头聚合", txt)
        self.assertIn("30.00-31.00s", txt)

    # --- 「降级」必须被显式识别出来（否则 UI 无法把静默降级回显给用户）---

    def test_degradation_detected_without_scenes(self) -> None:
        from ai_video_agent.agent import speech_delete_degrades_to_raw_segments
        idx = self._index()
        idx.scenes = []
        self.assertTrue(
            speech_delete_degrades_to_raw_segments("删掉说公交车的片段", idx),
            "没有镜头信息时，按台词删除只能给语音段落 → 属于降级")

    def test_no_degradation_when_shots_available(self) -> None:
        from ai_video_agent.agent import speech_delete_degrades_to_raw_segments
        self.assertFalse(
            speech_delete_degrades_to_raw_segments("删掉说公交车的片段", self._index()),
            "有镜头信息时会外扩到镜头 → 不是降级")

    def test_no_degradation_for_other_commands(self) -> None:
        from ai_video_agent.agent import speech_delete_degrades_to_raw_segments
        idx = self._index()
        idx.scenes = []
        for cmd in ("去掉前2秒", "删掉有公交车的片段", "加字幕\"开场\""):
            self.assertFalse(speech_delete_degrades_to_raw_segments(cmd, idx),
                             f"`{cmd}` 不是按台词删除，不该报降级")

    def test_no_degradation_when_keyword_never_spoken(self) -> None:
        """关键词压根没在台词里出现 → 那是"没匹配到操作"，不是降级。"""
        from ai_video_agent.agent import speech_delete_degrades_to_raw_segments
        idx = self._index()
        idx.scenes = []
        self.assertFalse(
            speech_delete_degrades_to_raw_segments("删掉说火锅的片段", idx))

    def test_degradation_helper_tolerates_missing_inputs(self) -> None:
        from ai_video_agent.agent import speech_delete_degrades_to_raw_segments
        self.assertFalse(speech_delete_degrades_to_raw_segments("", self._index()))
        self.assertFalse(speech_delete_degrades_to_raw_segments("删掉说公交车的片段", None))

    # --- 用 M2 事实核对模型的删除区间（真机实测暴露的缺口）---

    def _ops(self, ranges):
        return [{"type": "delete_range", "payload": {"start": a, "end": b}}
                for (a, b) in ranges]

    GOOD = [(0.0, 4.0), (8.0, 12.0), (16.0, 20.0)]

    def test_expected_ranges_from_speech(self) -> None:
        from ai_video_agent.agent import expected_semantic_ranges
        got = expected_semantic_ranges("删掉说公交车的片段", self._index())
        self.assertEqual(got, self.GOOD)

    def test_expected_ranges_from_visual(self) -> None:
        """「删掉有 X」视觉优先 —— 视觉命中就不看台词。"""
        from ai_video_agent.agent import expected_semantic_ranges
        visuals = [VisualSegment(label="bus", start=0.0, end=4.0, confidence=0.9,
                                 source="yolo")]
        # visuals 固定 → 只该给出视觉那一段，而不是台词命中的三段
        self.assertEqual(
            expected_semantic_ranges("删掉有公交车的片段", self._index(visuals=visuals)),
            [(0.0, 4.0)])

    def test_expected_ranges_none_for_non_semantic_commands(self) -> None:
        from ai_video_agent.agent import expected_semantic_ranges
        for cmd in ("去掉前2秒", "加字幕\"开场\"", "在第2秒切一刀", ""):
            self.assertIsNone(expected_semantic_ranges(cmd, self._index()),
                              f"`{cmd}` 不是语义删除 → 返回 None，不做核对")

    def test_discrepancy_free_when_ops_match_facts(self) -> None:
        from ai_video_agent.agent import semantic_delete_discrepancies
        self.assertEqual(semantic_delete_discrepancies(
            "删掉说公交车的片段", self._index(), self._ops(self.GOOD)), [])

    def test_missing_range_reported(self) -> None:
        """模型漏删第 3 个镜头 → 必须报出来（真机 0.6B 就是这么答的）。"""
        from ai_video_agent.agent import semantic_delete_discrepancies
        msgs = semantic_delete_discrepancies(
            "删掉说公交车的片段", self._index(), self._ops(self.GOOD[:2]))
        self.assertEqual(len(msgs), 1)
        self.assertIn("漏删", msgs[0])
        self.assertIn("16.00-20.00s", msgs[0])

    def test_extra_range_reported(self) -> None:
        """模型多删一段没人要的 → 同样必须报出来（真机 0.6B 也这么答过）。"""
        from ai_video_agent.agent import semantic_delete_discrepancies
        msgs = semantic_delete_discrepancies(
            "删掉说公交车的片段", self._index(), self._ops(self.GOOD + [(20.0, 24.0)]))
        self.assertEqual(len(msgs), 1)
        self.assertIn("多删", msgs[0])
        self.assertIn("20.00-24.00s", msgs[0])

    def test_tolerance_absorbs_rounding_noise(self) -> None:
        """模型给的秒数可能差一点点（如 7.999），别报虚假差异。"""
        from ai_video_agent.agent import semantic_delete_discrepancies
        noisy = [(0.001, 4.001), (8.0, 12.0), (16.0, 20.0)]
        self.assertEqual(semantic_delete_discrepancies(
            "删掉说公交车的片段", self._index(), self._ops(noisy)), [])

    def test_discrepancy_check_skipped_for_non_semantic(self) -> None:
        from ai_video_agent.agent import semantic_delete_discrepancies
        self.assertEqual(semantic_delete_discrepancies(
            "去掉前2秒", self._index(),
            [{"type": "trim", "payload": {"targetClipId": "src_video", "in": 2.0}}]), [])

    def test_discrepancy_check_handles_malformed_ops(self) -> None:
        """payload 缺字段 / 不是数字 → 忽略而不是崩；能解析的那条照常参与核对。

        注意：**必须混入至少一条能解析的 op**，否则守卫会在"没有 delete_range"时提前返回，
        `all(...)` 作用在空列表上恒真 —— 测试会变成永真 assertions（假绿）。
        """
        from ai_video_agent.agent import semantic_delete_discrepancies
        mixed = [{"type": "delete_range", "payload": {}},           # 缺 start/end
                 {"type": "delete_range"},                          # 缺 payload
                 {"type": "cut", "payload": {"targetClipId": "src_video", "at": 2.0}},
                 {"type": "delete_range", "payload": {"start": 0.0, "end": 4.0}}]  # 唯一合法的一条
        msgs = semantic_delete_discrepancies("删掉说公交车的片段", self._index(), mixed)
        self.assertEqual(len(msgs), 2)                      # 漏了 8-12 与 16-20
        self.assertTrue(all("漏删" in m for m in msgs))
        self.assertNotIn("多删", "\n".join(msgs), "畸形 op 不该被算成多删")

    # --- 死角：手上没有 M2 事实时，模型只能臆造，必须指向第 ② 步 ---

    def _bare_index(self):
        """已摄取但没跑 M2：没有镜头、没有视觉标签、没有台词。"""
        return FootageIndex(mediaId="x", filepath="dummy.mp4",
                            metadata={"duration": 24.0})

    def test_no_m2_facts_points_user_to_step_two(self) -> None:
        """没有 M2 事实时，报"多删"是误导 —— 要指向「先跑 ② 素材理解」。"""
        from ai_video_agent.agent import semantic_delete_discrepancies
        msgs = semantic_delete_discrepancies(
            "删掉有公交车的片段", self._bare_index(), self._ops([(0.0, 4.0)]))
        self.assertEqual(len(msgs), 1)
        self.assertIn("没有可用的 M2 结果", msgs[0])
        self.assertIn("臆测", msgs[0])
        self.assertNotIn("多删", msgs[0], "真因不是算错区间，而是根本没有事实")
        self.assertNotIn("漏删", msgs[0])

    def test_none_index_still_warns(self) -> None:
        """index 为 None 也不能静默 —— 这是上一版完全不核对的死角。"""
        from ai_video_agent.agent import semantic_delete_discrepancies
        msgs = semantic_delete_discrepancies(
            "删掉有公交车的片段", None, self._ops([(0.0, 4.0), (8.0, 12.0)]))
        self.assertEqual(len(msgs), 1)
        self.assertIn("没有可用的 M2 结果", msgs[0])

    def test_nothing_matched_says_so_instead_of_listing_extra(self) -> None:
        """有事实但关键词一个都没命中 → 说"没命中"，而不是逐条说"多删"。"""
        from ai_video_agent.agent import semantic_delete_discrepancies
        msgs = semantic_delete_discrepancies(
            "删掉说火锅的片段", self._index(), self._ops([(0.0, 4.0), (8.0, 12.0)]))
        self.assertEqual(len(msgs), 1)
        self.assertIn("没有", msgs[0])
        self.assertIn("火锅", msgs[0])
        self.assertNotIn("多删", msgs[0])

    def test_no_ops_means_nothing_to_check(self) -> None:
        """模型压根没给删除区间 → 这里不重复报（由规划层报"未匹配到操作"）。"""
        from ai_video_agent.agent import semantic_delete_discrepancies
        for idx in (self._bare_index(), None, self._index()):
            self.assertEqual(semantic_delete_discrepancies(
                "删掉有公交车的片段", idx, []), [])


if __name__ == "__main__":
    unittest.main()
