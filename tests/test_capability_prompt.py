"""T2 能力清单注入：模型看到的能力 == 编译器真正支持的能力。

两道防线：
1. 提示词侧 —— 清单由 registry 导出并注入，模型不必猜；
2. 校验侧 —— 模型若仍幻觉出清单外效果，validate_ops 必须拦下（不只靠提示）。
"""
from __future__ import annotations

import unittest

from ai_video_agent import agent, operations
from ai_video_agent.registry import (AUDIO_EFFECT_SPECS, OPERATIONS,
                                     VIDEO_EFFECT_SPECS)
from ai_video_agent.timeline import Clip, Metadata, Timeline, Track


class _Stub:
    def __init__(self, out: str) -> None:
        self.out = out
        self.seen_prompt = ""

    def complete(self, prompt: str) -> str:
        self.seen_prompt = prompt
        return self.out


def _tl() -> Timeline:
    return Timeline(
        metadata=Metadata(schemaVersion="1.0.0", fps=30,
                          resolution={"w": 640, "h": 360}, duration=8.0),
        tracks=[Track("v_main", "video")],
        clips=[Clip("c1", "v_main", "media_001", 0.0, 8.0, 0.0, 8.0)],
    )


class TestCapabilityBlock(unittest.TestCase):
    def test_lists_every_registered_capability(self) -> None:
        block = agent.capability_block()
        for spec in VIDEO_EFFECT_SPECS.values():
            self.assertIn(spec.canonical, block)
        for spec in AUDIO_EFFECT_SPECS.values():
            self.assertIn(spec.canonical, block)
        for op in OPERATIONS:
            self.assertIn(op, block)

    def test_declares_out_of_scope(self) -> None:
        self.assertIn("不在此清单内", agent.capability_block())


class TestPromptInjection(unittest.TestCase):
    def test_build_prompt_contains_capability_block(self) -> None:
        planner = agent.JsonPlanner(_Stub("[]"))
        prompt = planner.build_prompt("随便剪一下", _tl(), None)
        self.assertIn("可用能力清单", prompt)
        self.assertIn(agent.capability_block(), prompt)

    def test_prompt_lists_every_effect(self) -> None:
        planner = agent.JsonPlanner(_Stub("[]"))
        prompt = planner.build_prompt("x", _tl(), None)
        for spec in VIDEO_EFFECT_SPECS.values():
            self.assertIn(spec.canonical, prompt)


class TestHallucinationIsBlocked(unittest.TestCase):
    def test_effect_outside_manifest_is_rejected_downstream(self) -> None:
        """模型幻觉出清单外效果时，validate_ops 必须拦下。"""
        op = {"type": "effect",
              "payload": {"targetClipId": "c1", "effectType": "cinematic"}}
        errs = operations.validate_ops(_tl(), [op])
        self.assertTrue(errs)
        self.assertIn("不支持的效果", errs[0]["reason"])

    def test_unsupported_request_yields_no_op(self) -> None:
        """当前不支持变速/水印：planner 应输出 []，而不是编一个假操作。"""
        planner = agent.JsonPlanner(_Stub("[]"))
        ops = planner.plan("把这段视频加速到 2 倍并加个水印", _tl(), None)
        self.assertEqual(ops, [])
