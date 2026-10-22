"""T1 能力注册表：清单 ↔ 实现 的双向一致性。

红线：清单说有的，编译器必须真能编译；编译器支持的，清单必须列出。
任何一侧漏项，这里必须红 —— 这是防「提示词说谎」的机器闸。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai_video_agent import compiler, operations, registry


class TestOperationsConsistency(unittest.TestCase):
    def test_manifest_operations_equal_dispatch_handlers(self) -> None:
        """清单里的操作 == 真的能执行的操作（双向、动态断言，非抄写）。"""
        manifest_ops = set(registry.capability_manifest()["operations"])
        dispatch_ops = set(operations.OPERATION_HANDLERS)
        self.assertEqual(manifest_ops, dispatch_ops)

    def test_unknown_operation_still_fails_loud(self) -> None:
        from ai_video_agent.timeline import Timeline

        with self.assertRaises(ValueError):
            operations.apply_operation(Timeline(), {"type": "no_such_op", "payload": {}})


class TestVideoEffectsConsistency(unittest.TestCase):
    def test_every_manifest_effect_compiles_to_a_filter(self) -> None:
        manifest = registry.capability_manifest()["effects"]["video"]
        for canonical, spec in manifest.items():
            for name in (canonical, *spec.get("aliases", [])):
                f = compiler.video_effect_filter({"effectType": name}, clip_dur=4.0)
                self.assertIsNotNone(f, f"清单里的效果 `{name}` 编译器产不出滤镜")

    def test_every_compiler_effect_is_in_manifest(self) -> None:
        manifest = registry.capability_manifest()["effects"]["video"]
        declared = set()
        for canonical, spec in manifest.items():
            declared.add(canonical)
            declared.update(spec.get("aliases", []))
        self.assertEqual(set(compiler.VIDEO_EFFECTS), declared)

    def test_param_defaults_within_declared_range(self) -> None:
        for group in registry.capability_manifest()["effects"].values():
            for spec in group.values():
                for pname, p in (spec.get("params") or {}).items():
                    default = p.get("default")
                    if default is None:      # fade 时长运行时才定
                        continue
                    if "min" in p:
                        self.assertGreaterEqual(default, p["min"], (pname, default))
                    if "max" in p:
                        self.assertLessEqual(default, p["max"], (pname, default))


class TestAudioEffectsConsistency(unittest.TestCase):
    def test_every_manifest_audio_effect_compiles(self) -> None:
        manifest = registry.capability_manifest()["effects"]["audio"]
        for canonical, spec in manifest.items():
            for name in (canonical, *spec.get("aliases", [])):
                f = compiler.audio_effect_filter({"effectType": name}, 4.0)
                self.assertIsNotNone(f, f"音频效果 `{name}` 编译器产不出滤镜")

    def test_every_compiler_audio_effect_is_in_manifest(self) -> None:
        manifest = registry.capability_manifest()["effects"]["audio"]
        declared = set()
        for canonical, spec in manifest.items():
            declared.add(canonical)
            declared.update(spec.get("aliases", []))
        self.assertEqual(set(compiler.AUDIO_EFFECTS), declared)


class TestManifestShape(unittest.TestCase):
    def test_manifest_is_json_serializable(self) -> None:
        s = json.dumps(registry.capability_manifest(), ensure_ascii=False)
        self.assertIn("operations", s)
        self.assertIn("effects", s)

    def test_manifest_is_deterministic(self) -> None:
        self.assertEqual(registry.capability_manifest(), registry.capability_manifest())

    def test_transition_types_declared(self) -> None:
        self.assertEqual(set(registry.TRANSITION_TYPES), {"fade", "cross", "wipe"})

    def test_checked_in_manifest_file_matches_code(self) -> None:
        """仓里的 capability_manifest.json 必须与代码一致 —— 防手改。"""
        path = Path(__file__).resolve().parent.parent / "capability_manifest.json"
        if not path.exists():
            self.skipTest("未生成 capability_manifest.json（可选产物）")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data, registry.capability_manifest(),
                         "capability_manifest.json 被手改过，请用 registry.dump_manifest() 重新导出")
