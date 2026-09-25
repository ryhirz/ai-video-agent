"""鲁棒性回归测试（2026-09-23 修两个实测 bug 后补）。

覆盖两条实测踩坑链：
1) **操作链引用已消失的 id**：LLM/规则 planner 在 cut 之后继续引用旧 id
   （如 src_video），旧实现直接抛裸 KeyError('clip not found: src_video')，
   整批操作全废（M5 界面表现为「编译失败」）。
   → 现在由 operations.apply_operations 的派生 id 别名表自动解析 / 跳过并告警。
2) **LLM 输出不可直接 json.loads**：带代码块围栏、前后解释文字、多个片段，
   旧实现抛 json.JSONDecodeError('Extra data')。
   → 现在由 agent._extract_json_array 容错提取，并自动重试一次。
"""
import json
import unittest

from ai_video_agent.agent import (JsonPlanner, _extract_json_array, _validate_ops,
                                 facts_block, timeline_digest)
from ai_video_agent.footage_index import FootageIndex, VisualSegment
from ai_video_agent.ingest import build_initial_timeline
from ai_video_agent.operations import OperationError, apply_operations
from ai_video_agent.compiler import compile as compile_timeline


def _index(dur: float = 15.0, visual: list = None) -> FootageIndex:
    return FootageIndex(
        mediaId="t", filepath="dummy.mp4",
        metadata={"duration": dur, "fps": 30, "width": 320, "height": 240,
                  "video_codec": "h264", "audio_codec": "aac", "has_audio": True},
        scenes=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 10.5},
                {"start": 10.5, "end": dur}],
        visual_segments=visual or [],
    )


class TestStaleClipIdResolution(unittest.TestCase):
    """场景 1：操作链里引用了「已被 cut 拆掉的旧 id」。"""

    def setUp(self) -> None:
        self.tl = build_initial_timeline(_index())

    def test_cut_then_cut_on_stale_id_resolves_by_time(self) -> None:
        """两次 cut 都写 src_video（第二次实际指向 src_video__b）——按 at 自动解析。"""
        ops = [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}},
               {"type": "cut", "payload": {"targetClipId": "src_video", "at": 8.0}}]
        tl, logs, warns = apply_operations(self.tl, ops)
        ids = sorted(c.id for c in tl.clips)
        self.assertEqual(warns, [])
        self.assertIn("src_video__a", ids)          # 0–3
        self.assertIn("src_video__b__a", ids)       # 3–8
        self.assertIn("src_video__b__b", ids)       # 8–15
        self.assertTrue(any("自动解析" in x for x in logs))
        # 位置也对：三段时长 3 / 5 / 7
        spans = sorted(round(c.out - c.in_, 2) for c in tl.clips if c.track == "v_main")
        self.assertEqual(spans, [3.0, 5.0, 7.0])

    def test_semantic_remove_chain_from_stale_id(self) -> None:
        """「删掉有bus的片段」的三步链（LLM 把第二刀也写成 src_video）也能正确落成保留 2 段。"""
        ops = [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 0.5}},
               {"type": "cut", "payload": {"targetClipId": "src_video", "at": 4.0}},
               {"type": "delete_clip", "payload": {"targetClipId": "src_video__b__a"}}]
        tl, _logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(warns, [])
        vspans = sorted((round(c.in_, 2), round(c.out, 2)) for c in tl.clips
                        if c.track == "v_main")
        self.assertEqual(vspans, [(0.0, 0.5), (4.0, 15.0)])
        self.assertIn("ffmpeg", compile_timeline(tl)["command"])

    def test_unknown_id_warns_without_aborting_batch(self) -> None:
        """臆造 id：只跳过该条并列出可用 id，其余操作照常生效（旧实现会整批崩）。"""
        ops = [{"type": "delete_clip", "payload": {"targetClipId": "nonexistent"}},
               {"type": "add_subtitle", "payload": {"text": "开场", "start": 0.0, "end": 3.0}}]
        tl, logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(len(warns), 1)
        self.assertIn("nonexistent", warns[0])
        self.assertIn("src_video", warns[0])        # 错误信息里给出可用 id
        self.assertEqual(len(logs), 1)              # 字幕这条正常应用
        self.assertEqual(len(tl.subtitles), 1)

    def test_ambiguous_delete_is_reported_not_silently_dropped(self) -> None:
        """删一个被切开的旧 id：语义歧义（是删一段还是全删）→ 明确报错而不是猜。"""
        ops = [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 5.0}},
               {"type": "delete_clip", "payload": {"targetClipId": "src_video"}}]
        tl, _logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(len(warns), 1)
        self.assertIn("多个候选", warns[0])
        self.assertEqual(len([c for c in tl.clips if c.track == "v_main"]), 2)

    def test_strict_mode_raises_operation_error(self) -> None:
        """strict=True 时抛 OperationError（带上下文），供 Agent.execute 使用。"""
        with self.assertRaises(OperationError):
            apply_operations(self.tl, [{"type": "trim",
                                        "payload": {"targetClipId": "ghost", "in": 1.0}}],
                             strict=True)


class TestJsonExtraction(unittest.TestCase):
    """场景 2：LLM 输出容错解析。"""

    def test_plain_array(self) -> None:
        self.assertEqual(_extract_json_array('[{"type":"cut","payload":{}}]'),
                         [{"type": "cut", "payload": {}}])

    def test_fenced(self) -> None:
        self.assertEqual(_extract_json_array('```json\n[{"type":"cut"}]\n```'),
                         [{"type": "cut"}])

    def test_leading_and_trailing_prose(self) -> None:
        raw = '好的，这是我给出的操作：\n[{"type":"cut","payload":{"at":2}}]\n以上。'
        self.assertEqual(_extract_json_array(raw), [{"type": "cut", "payload": {"at": 2}}])

    def test_extra_data_two_arrays_takes_first(self) -> None:
        """实测报错 'Extra data: line 1 column 3' 的形态：模型吐了两段。"""
        self.assertEqual(_extract_json_array('[]\n[{"type":"trim","payload":{"in":2}}]'), [])

    def test_brackets_inside_string_do_not_break_scan(self) -> None:
        raw = '结果：[{"type":"add_subtitle","payload":{"text":"[副标题] 开场","start":0,"end":2}}]'
        self.assertEqual(_extract_json_array(raw)[0]["payload"]["text"], "[副标题] 开场")

    def test_no_array_raises(self) -> None:
        with self.assertRaises(ValueError):
            _extract_json_array("抱歉，我无法处理这个指令。")

    def test_validate_rejects_hallucinated_type(self) -> None:
        with self.assertRaises(ValueError):
            _validate_ops([{"type": "speed_up", "payload": {}}])


class _SeqProvider:
    """按顺序返回预设文本的假 provider（用于验证重试与提示词内容）。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt: str, *, system=None) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class TestJsonPlannerRobustness(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = _index(visual=[VisualSegment(label="bus", start=0.5, end=4.0,
                                                source="yolo")])
        self.tl = build_initial_timeline(self.idx)

    def test_retry_once_on_unparsable_output(self) -> None:
        prov = _SeqProvider(["抱歉，我无法处理。",
                             '[{"type":"cut","payload":{"targetClipId":"src_video","at":2.0}}]'])
        ops = JsonPlanner(prov).plan("切一刀", self.tl, self.idx)
        self.assertEqual(ops[0]["type"], "cut")
        self.assertEqual(len(prov.prompts), 2, "首次解析失败应重试一次")

    def test_error_message_carries_raw_output(self) -> None:
        prov = _SeqProvider(["垃圾输出", "还是垃圾"])
        with self.assertRaises(ValueError) as cm:
            JsonPlanner(prov).plan("x", self.tl, self.idx)
        msg = str(cm.exception)
        self.assertIn("垃圾输出", msg)      # 首次输出
        self.assertIn("还是垃圾", msg)      # 重试输出
        self.assertIn("无法解析", msg)

    def test_prompt_contains_clip_ids_and_facts(self) -> None:
        """提示词必须含「可用 id 清单」与 M2 事实（否则模型只能臆造 id/时间）。"""
        prov = _SeqProvider(['[]'])
        JsonPlanner(prov).plan("删掉有bus的片段", self.tl, self.idx)
        p = prov.prompts[0]
        self.assertIn("`src_video`", p)
        self.assertIn("`src_audio`", p)
        self.assertIn("bus: 0.50-4.00s", p)
        self.assertIn("__a", p)           # 讲清了 cut 的派生命名规则

    def test_facts_block_marks_missing_m2(self) -> None:
        txt = facts_block(None)
        self.assertIn("未摄取", txt)

    def test_timeline_digest_lists_ids(self) -> None:
        d = timeline_digest(self.tl)
        self.assertIn("`src_video`", d)
        self.assertIn("0.00–15.00s", d)


class TestTrimSyncAudio(unittest.TestCase):
    """场景 3：trim 一个 clip 时，同源、同初始时窗、不同轨的配对 clip（如视频的音频）
    必须一起裁，否则混流后音频超前、尾部出现定格帧。"""

    def setUp(self) -> None:
        self.tl = build_initial_timeline(_index())

    def _get(self, tl, cid):
        return next(c for c in tl.clips if c.id == cid)

    def test_trim_video_head_also_shifts_paired_audio(self) -> None:
        ops = [{"type": "trim", "payload": {"targetClipId": "src_video", "in": 2.0}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        v = self._get(tl2, "src_video")
        a = self._get(tl2, "src_audio")
        self.assertEqual((v.in_, v.out), (2.0, 15.0))
        self.assertEqual((a.in_, a.out), (2.0, 15.0))   # 音频同步裁头
        self.assertAlmostEqual(v.sourceIn, 2.0)
        self.assertAlmostEqual(a.sourceIn, 2.0)           # 音频 sourceIn 也前移

    def test_trim_video_tail_also_shifts_paired_audio(self) -> None:
        ops = [{"type": "trim", "payload": {"targetClipId": "src_video", "out": 10.0}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        v = self._get(tl2, "src_video")
        a = self._get(tl2, "src_audio")
        self.assertEqual((v.in_, v.out), (0.0, 10.0))
        self.assertEqual((a.in_, a.out), (0.0, 10.0))   # 音频同步裁尾
        self.assertAlmostEqual(v.sourceOut, 10.0)
        self.assertAlmostEqual(a.sourceOut, 10.0)


class TestAvSyncOnVideoEdit(unittest.TestCase):
    """场景 4（2026-09-23 实跑新增）：视频剪辑必须带动配对音频。

    M5 实跑里「删掉有bus的片段」渲染出的成片音频仍是**原始时间轴**（-shortest 只按
    长度截断），与画面完全对不上。根因：cut / delete_clip 只作用于视频 clip。
    """

    def setUp(self) -> None:
        self.tl = build_initial_timeline(_index())

    @staticmethod
    def _spans(tl, track):
        return sorted((round(c.sourceIn, 3), round(c.sourceOut, 3))
                      for c in tl.clips if c.track == track)

    def test_delete_range_removes_video_and_paired_audio(self) -> None:
        ops = [{"type": "delete_range", "payload": {"start": 10.5, "end": 14.0}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(warns, [])
        self.assertEqual(self._spans(tl2, "v_main"), [(0.0, 10.5), (14.0, 15.0)])
        self.assertEqual(self._spans(tl2, "a_main"), [(0.0, 10.5), (14.0, 15.0)])

    def test_two_intervals_both_deleted__the_real_bug(self) -> None:
        """两段 bus 都要删掉。

        旧写法（cut+cut+delete_clip 手推派生 id）在第二个区间时 id 已失效，
        非严格模式静默跳过 -> 实测只删掉了后一段。
        """
        ops = [{"type": "delete_range", "payload": {"start": 0.5, "end": 4.0}},
               {"type": "delete_range", "payload": {"start": 10.5, "end": 14.0}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(warns, [])
        expect = [(0.0, 0.5), (4.0, 10.5), (14.0, 15.0)]
        self.assertEqual(self._spans(tl2, "v_main"), expect)
        self.assertEqual(self._spans(tl2, "a_main"), expect)   # 音频逐段对齐
        total = sum(b - a for a, b in expect)
        self.assertAlmostEqual(total, 8.0)

    def test_delete_range_on_missing_interval_warns_not_silent(self) -> None:
        """区间已被删过 -> 必须告警，而不是静默什么都不做。"""
        ops = [{"type": "delete_range", "payload": {"start": 10.5, "end": 14.0}},
               {"type": "delete_range", "payload": {"start": 10.5, "end": 14.0}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(warns), 1)
        self.assertIn("delete_range", warns[0])

    def test_delete_clip_on_video_also_removes_paired_audio(self) -> None:
        ops = [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 4.0}},
               {"type": "delete_clip", "payload": {"targetClipId": "src_video__a"}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        # 视频删掉 [0,4]；音频也必须只剩 [4,15]，否则 A/V 失步
        self.assertEqual(self._spans(tl2, "v_main"), [(4.0, 15.0)])
        self.assertEqual(self._spans(tl2, "a_main"), [(4.0, 15.0)])

    def test_delete_clip_on_audio_does_not_touch_video(self) -> None:
        """反向不成立：删音轨不该把画面也删了（视频是主轨）。"""
        ops = [{"type": "delete_clip", "payload": {"targetClipId": "src_audio"}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(self._spans(tl2, "v_main"), [(0.0, 15.0)])
        self.assertEqual(self._spans(tl2, "a_main"), [])

    def test_cut_splits_paired_audio_symmetrically(self) -> None:
        ops = [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 6.0}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(self._spans(tl2, "v_main"), [(0.0, 6.0), (6.0, 15.0)])
        self.assertEqual(self._spans(tl2, "a_main"), [(0.0, 6.0), (6.0, 15.0)])

    def test_trim_after_cut_still_syncs_paired_audio(self) -> None:
        """cut 之后再 trim 某一段，音频仍要跟着裁。"""
        ops = [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 4.0}},
               {"type": "trim", "payload": {"targetClipId": "src_video__b", "in": 6.0}}]
        tl2, logs, warns = apply_operations(self.tl, ops)
        self.assertEqual(self._spans(tl2, "a_main"), [(0.0, 4.0), (6.0, 15.0)])


class TestCompiledAudioGraph(unittest.TestCase):
    """多段音频必须顺序拼接（concat），不能用 amix 并行混音。"""

    def test_multi_audio_segments_use_concat(self) -> None:
        tl = build_initial_timeline(_index())
        tl2, _, _ = apply_operations(tl, [
            {"type": "delete_range", "payload": {"start": 0.5, "end": 4.0}},
            {"type": "delete_range", "payload": {"start": 10.5, "end": 14.0}},
        ])
        res = compile_timeline(tl2)
        self.assertIn("concat=n=3:v=0:a=1[aout]", res["audio_filter"])
        self.assertNotIn("amix", res["audio_filter"])
        # 视频同样是 3 段拼接
        self.assertIn("concat=n=3:v=1:a=0[vout]", res["video_filter"])


class TestSubtitleDedupe(unittest.TestCase):
    """LLM 偶发把同一条字幕输出两遍 -> 叠加渲染会变成重影。"""

    def test_duplicate_subtitle_dropped(self) -> None:
        tl = build_initial_timeline(_index())
        sub = {"type": "add_subtitle",
               "payload": {"text": "开场", "start": 0.0, "end": 5.0}}
        tl2, logs, warns = apply_operations(tl, [sub, dict(sub)])
        self.assertEqual(len(tl2.subtitles), 1)
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(warns), 1)
        self.assertIn("重复", warns[0])

    def test_different_subtitles_both_kept(self) -> None:
        tl = build_initial_timeline(_index())
        ops = [{"type": "add_subtitle",
                "payload": {"text": "开场", "start": 0.0, "end": 5.0}},
               {"type": "add_subtitle",
                "payload": {"text": "结束", "start": 8.0, "end": 12.0}}]
        tl2, logs, warns = apply_operations(tl, ops)
        self.assertEqual(len(tl2.subtitles), 2)
        self.assertEqual(warns, [])


class TestSpeechPromptContract(unittest.TestCase):
    """「按台词删除」的**提示词契约**测试（2026-09-24 补）。

    背景：这条能力靠的是"提示词把事实讲对"。但提示词内容此前**没有任何断言** ——
    `facts_block` 的格式或 schema 规则被改坏时，所有测试照样绿，LLM 路径却已悄悄退化。
    （stub 测试只断言了 model / messages 非空 / 鉴权头，不看提示词里写了什么。）

    这里钉三件事：
    1) schema 里"按镜头聚合 / 未按镜头聚合"两个分支都在（模型要知道该看哪个）；
    2) 有镜头时，提示词真的带上了「镜头 X-Y」聚合区间，且不再带半句小时间戳；
    3) **没有镜头时不得宣称已聚合** —— 否则提示词对模型说假话，
       模型会去找不存在的「镜头 X-Y」，或违反规则改用台词小时间戳。
    """

    SPEECH = [(8.15, 9.34, "第三幕"), (9.34, 11.28, "又出现了一辆公交车")]
    SCENES = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0},
              {"start": 8.0, "end": 12.0}, {"start": 12.0, "end": 15.0}]

    def _index(self, with_scenes: bool = True) -> FootageIndex:
        from ai_video_agent.footage_index import TranscriptSegment
        idx = _index(dur=15.0)
        # 显式指定镜头：让两句台词都落在 8–12s 这一个镜头里，
        # 这样"聚合"的效果是唯一确定的（否则会跨镜合并，断言无从下手）。
        idx.scenes = list(self.SCENES) if with_scenes else []
        idx.transcript_segments = [TranscriptSegment(start=s, end=e, text=t)
                                   for (s, e, t) in self.SPEECH]
        return idx

    def _facts_section(self, prompt: str) -> str:
        """只取【素材理解事实（M2）】到【用户指令】之间的那一段。

        断言必须**限定到事实块**：schema 的【示例】区里也会出现 "镜头 8.00-12.00s"
        这类示例文本，对整条提示词做 assertNotIn 会误报（示例本就该是示例）。
        """
        start = prompt.index("【素材理解事实（M2）】")
        end = prompt.index("【用户指令】")
        return prompt[start:end]

    def test_schema_hint_declares_both_speech_branches(self) -> None:
        h = JsonPlanner._SCHEMA_HINT
        self.assertIn("3b.", h)
        self.assertIn("按镜头聚合", h)
        self.assertIn("未按镜头聚合", h)
        # 必须明确禁止"拿台词小时间戳去删" —— 那是留下半个镜头的直接原因
        self.assertIn("半个镜头", h)

    def test_prompt_carries_aggregated_shot_ranges(self) -> None:
        idx = self._index()
        tl = build_initial_timeline(idx)
        prov = _SeqProvider(["[]"])
        JsonPlanner(prov).plan("删掉说公交车的片段", tl, idx)
        p = prov.prompts[0]
        facts = self._facts_section(p)
        self.assertIn("镜头 8.00-12.00s", facts)   # 可直接删除的区间
        self.assertIn("第三幕", facts)              # 台词原文仍在（模型靠它判关键词）
        self.assertNotIn("8.15-9.34s", facts)      # 不该再出现半句小时间戳
        self.assertIn("删掉说公交车的片段", p)      # 用户指令原样带入

    def test_prompt_marks_unaggregated_when_no_scenes(self) -> None:
        """无镜头信息时，提示词必须**明说没有聚合**，而不是继续宣称已聚合。"""
        idx = self._index(with_scenes=False)
        tl = build_initial_timeline(idx)
        prov = _SeqProvider(["[]"])
        JsonPlanner(prov).plan("删掉说公交车的片段", tl, idx)
        facts = self._facts_section(prov.prompts[0])
        self.assertIn("未按镜头聚合", facts)
        self.assertIn("8.15-9.34s", facts)          # 退化后给的是台词自己的区间
        self.assertNotIn("镜头 8.00-12.00s", facts)  # 不能出现不存在的镜头区间

    def test_no_inconsistency_between_hint_and_facts(self) -> None:
        """交叉检查：schema 里承诺的两种标注，事实块必须恰好在对应场景下给出。

        这条是防"提示词与事实块各自演化、互相矛盾"的结构性护栏 ——
        两者由不同函数生成，最容易在后续重构里分家。
        """
        hint = JsonPlanner._SCHEMA_HINT
        # 用带前缀的完整标注做断言：'按镜头聚合' 是 '未按镜头聚合' 的子串，
        # 只用子串断言会在退化场景下"假通过"。
        for with_scenes, marker in ((True, "台词（**按镜头聚合"),
                                    (False, "台词（**未按镜头聚合")):
            facts = facts_block(self._index(with_scenes=with_scenes))
            self.assertIn(marker, facts, f"with_scenes={with_scenes} 时事实块缺少标注 {marker}")
            self.assertIn(marker.replace("台词（**", "").rstrip("："),
                          hint, f"schema 未声明该标注，模型无从判断")


if __name__ == "__main__":
    unittest.main()
