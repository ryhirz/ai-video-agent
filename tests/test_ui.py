"""M5 Gradio UI 冒烟测试。

覆盖三类：
1) 装配不崩：build_app() 返回 gr.Blocks（Gradio 在构造期即校验组件接线）。
2) 处理逻辑不崩（不启动服务、不依赖真实视频文件）：
   用内存构造的 FootageIndex/Timeline 直调 do_plan / do_apply_compile，
   验证 "NL → 操作 → 确定性编译" 整链在无 Gradio 服务下也能跑通。
3) 真 LLM 端到端（离线、无真实 Key）：用标准库 http.server 起 stub 端点，
   验证 EditingAgent(provider=OpenAICompatibleProvider) → JsonPlanner 解析模型 JSON 整链；
   以及 HybridProvider 的「本地优先 + 云端兜底」路由逻辑。
"""
import json
import os
import threading
import typing
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import gradio as gr

from ai_video_agent.agent import EditingAgent, expected_semantic_ranges
from ai_video_agent.footage_index import FootageIndex, VisualSegment
from ai_video_agent.ingest import build_initial_timeline
from ai_video_agent.ui import (build_app, do_plan, do_apply_compile,
                             APPLY_OUTPUT_KEYS,
                             OpenAICompatibleProvider, HybridProvider)


def _apply(tl: object, ops: object) -> dict:
    """按输出契约把 do_apply_compile 的返回解成 {键: 值}。

    测试也走 APPLY_OUTPUT_KEYS，避免测试自己写死位置而跟着一起错位。
    """
    return dict(zip(APPLY_OUTPUT_KEYS, do_apply_compile(tl, ops)))



def _index(dur: float = 10.0,
           visual: list = None,
           speech: list = None,
           scenes: list = None) -> FootageIndex:
    idx = FootageIndex(
        mediaId="x", filepath="dummy.mp4",
        metadata={"duration": dur, "fps": 30, "width": 320, "height": 240,
                  "video_codec": "h264", "audio_codec": "", "has_audio": False},
        visual_segments=visual or [],
    )
    idx.scenes = scenes or []          # 默认空：保持既有测试的行为不变
    idx.transcript_segments = speech or []
    return idx


def _tl(idx: FootageIndex) -> "object":
    return build_initial_timeline(idx)


class TestUI(unittest.TestCase):
    # --- 装配不崩 ---
    def test_build_app_returns_blocks(self) -> None:
        app = build_app()
        self.assertIsInstance(app, gr.Blocks)

    # --- 基础指令：切一刀（规则 Planner，mode='rule'）---
    def test_plan_and_compile_cut(self) -> None:
        idx = _index()
        tl = _tl(idx)
        ops_state, ops_json, status = do_plan(
            "在第2秒切一刀", tl, idx, "rule", "", "", "", "", "")
        self.assertIsNotNone(ops_state)
        self.assertIn("cut", ops_json)
        self.assertIn("生成", status)

        res = _apply(tl, ops_state)
        self.assertIsNotNone(res["timeline_state"])
        self.assertIn("ffmpeg", res["cmd_box"])
        self.assertIn("[vout]", res["cmd_box"])  # 编译器产出视频滤镜链
        self.assertIn("✅", res["apply_status"])

    # --- 语义检索：删掉有 bus 的片段（需 index.visual_segments）---
    def test_plan_semantic_remove_label(self) -> None:
        idx = _index(visual=[VisualSegment(label="bus", start=3.0, end=5.0, source="yolo")])
        tl = _tl(idx)
        ops_state, _, _ = do_plan(
            "删掉有bus的片段", tl, idx, "rule", "", "", "", "", "")
        self.assertIsNotNone(ops_state)
        self.assertEqual(len(ops_state), 1)          # 每个区间一条 delete_range
        self.assertEqual(ops_state[0]["type"], "delete_range")

        res = _apply(tl, ops_state)
        self.assertIn("ffmpeg", res["cmd_box"])

    # --- 按台词删除：走 UI 处理器路径（需 index.transcript_segments + scenes）---
    def test_plan_semantic_remove_by_speech(self) -> None:
        """『删掉说"X"的片段』在 UI 链路上也要跑通，且区间必须外扩到镜头。"""
        from ai_video_agent.footage_index import TranscriptSegment
        scenes = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0},
                  {"start": 8.0, "end": 10.0}]
        speech = [TranscriptSegment(start=4.2, end=5.1, text="这里有一辆公交车"),
                  TranscriptSegment(start=8.3, end=9.0, text="结尾没有")]
        idx = _index(dur=10.0, speech=speech, scenes=scenes)
        tl = _tl(idx)
        ops_state, _, _ = do_plan(
            '删掉说"公交车"的片段', tl, idx, "rule", "", "", "", "", "")
        self.assertIsNotNone(ops_state)
        self.assertEqual(len(ops_state), 1)
        self.assertEqual(ops_state[0]["type"], "delete_range")
        # 关键：区间应是台词覆盖的**镜头** (4,8)，而不是台词本身那 0.9s
        self.assertEqual((ops_state[0]["payload"]["start"],
                          ops_state[0]["payload"]["end"]), (4.0, 8.0))

        res = _apply(tl, ops_state)
        self.assertIn("ffmpeg", res["cmd_box"])
        self.assertIn("✅", res["apply_status"])

    def test_plan_warns_when_speech_delete_degrades(self) -> None:
        """没有镜头信息时的按台词删除会退化成"半句区间"—— 状态里必须提示，不能静默降级。"""
        from ai_video_agent.footage_index import TranscriptSegment
        speech = [TranscriptSegment(start=4.2, end=5.1, text="这里有一辆公交车")]
        idx = _index(dur=10.0, speech=speech, scenes=[])   # 故意不给镜头
        tl = _tl(idx)
        ops_state, _, status = do_plan(
            '删掉说"公交车"的片段', tl, idx, "rule", "", "", "", "", "")
        self.assertIsNotNone(ops_state)
        self.assertIn("⚠️", status)
        self.assertIn("没有可用的镜头信息", status)
        # 区间退化到台词自己的 1–2s（而不是镜头 4–8s）
        self.assertEqual((ops_state[0]["payload"]["start"],
                          ops_state[0]["payload"]["end"]), (4.2, 5.1))

    def test_plan_no_warning_when_shots_present(self) -> None:
        """有镜头信息时不该出现降级提示（避免狼来了）。"""
        from ai_video_agent.footage_index import TranscriptSegment
        scenes = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0},
                  {"start": 8.0, "end": 10.0}]
        speech = [TranscriptSegment(start=4.2, end=5.1, text="这里有一辆公交车")]
        idx = _index(dur=10.0, speech=speech, scenes=scenes)
        _ops, _, status = do_plan(
            '删掉说"公交车"的片段', _tl(idx), idx, "rule", "", "", "", "", "")
        self.assertNotIn("没有可用的镜头信息", status)

    # --- 未摄取时安全降级 ---
    def test_plan_without_ingest(self) -> None:
        ops_state, ops_json, status = do_plan(
            "在第2秒切一刀", None, None, "rule", "", "", "", "", "")
        self.assertIsNone(ops_state)
        self.assertIn("⚠️", status)

    # --- 可选真 LLM 提供方可构造（不联网）---
    def test_llm_provider_constructible(self) -> None:
        p = OpenAICompatibleProvider("https://api.deepseek.com/v1", "fake-key", "deepseek-chat")
        self.assertEqual(p.endpoint, "https://api.deepseek.com/v1")
        self.assertEqual(p.model, "deepseek-chat")
        # complete() 会真实联网，此处不调用；仅验证热插拔点可实例化


class TestUIOutputContract(unittest.TestCase):
    """输出契约：处理函数的返回顺序必须与 click(outputs=[...]) 的组件顺序逐位对应。

    回归背景：给 ④ 的输出列表插入 gr.Video 时，返回元组却把新值加在末尾，导致整批后移
    一格 —— gr.Video 收到一段 Timeline JSON 当文件路径，界面上 ④ 整块渲染成错误卡。
    当时的旧测试按"函数自己的顺序"解包，对"接线顺序"完全无感，所以没拦住。
    """

    @staticmethod
    def _wired(fn: object) -> list:
        """取出某个处理函数在 build_app() 里实际接的输出组件列表。"""
        for bf in build_app().fns.values():
            if getattr(bf, "fn", None) is fn:
                return list(bf.outputs)
        raise AssertionError(f"未在 build_app() 中找到 {fn} 的接线")

    def test_apply_output_order_matches_component_types(self) -> None:
        """逐位按组件类型校验：键的顺序 == 组件的顺序。"""
        wired = self._wired(do_apply_compile)
        self.assertEqual(len(wired), len(APPLY_OUTPUT_KEYS))
        by_key = dict(zip(APPLY_OUTPUT_KEYS, wired))
        self.assertIsInstance(by_key["timeline_state"], gr.State)
        self.assertIsInstance(by_key["apply_status"], gr.Markdown)
        self.assertIsInstance(by_key["video_out"], gr.Video)
        self.assertIsInstance(by_key["tl_json"], gr.Code)
        self.assertIsInstance(by_key["cmd_box"], gr.Code)
        self.assertIsInstance(by_key["ass_box"], gr.Code)

    def test_apply_values_land_in_the_right_slots(self) -> None:
        """按内容校验每位归位：视频格只能是路径/空串，命令格才是 ffmpeg 命令。"""
        idx = _index()
        tl = _tl(idx)
        ops, _, _ = do_plan("去掉前2秒", tl, idx, "rule", "", "", "", "", "")
        res = _apply(tl, ops)

        video = res["video_out"]
        self.assertIsInstance(video, str)
        self.assertFalse(video.lstrip().startswith("{"), f"video_out 收到了 JSON：{video[:60]!r}")
        if video:                                    # 真出片了则必须是存在的文件
            self.assertTrue(os.path.exists(video), video)
        json.loads(res["tl_json"])                   # 第 4 格必须是合法 JSON
        self.assertIn("ffmpeg", res["cmd_box"])      # 第 5 格必须是命令，不是字幕
        self.assertNotIn("ffmpeg", res["ass_box"])   # 第 6 格不能是命令

    def test_all_click_outputs_arity_matches_return_annotation(self) -> None:
        """通用兜底：每个处理函数接的输出组件数 == 返回类型标注的元素个数。"""
        checked = 0
        for bf in build_app().fns.values():
            fn = getattr(bf, "fn", None)
            if fn is None:
                continue
            ret = typing.get_type_hints(fn).get("return")
            args = typing.get_args(ret)
            if not args or Ellipsis in args:         # 未标注或 Tuple[..., ...] 的跳过
                continue
            self.assertEqual(
                len(bf.outputs), len(args),
                f"{fn.__name__} 接了 {len(bf.outputs)} 个输出，但返回标注是 {len(args)} 元组")
            checked += 1
        self.assertGreaterEqual(checked, 4, "应至少校验到 4 个处理函数")


class _StubLLMHandler(BaseHTTPRequestHandler):
    """本地 stub：模拟 OpenAI 兼容 /chat/completions，回吐一条合法操作 JSON。"""

    last_body: dict = {}
    auth_seen: bool = False

    def do_POST(self) -> None:  # noqa: N802
        _StubLLMHandler.auth_seen = self.headers.get("Authorization") == "Bearer test-key"
        length = int(self.headers.get("Content-Length", 0))
        _StubLLMHandler.last_body = json.loads(self.rfile.read(length) or b"{}")
        content = json.dumps(
            [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}}])
        resp = {"choices": [{"message": {"content": content}}]}
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # 静默
        pass


class TestLLMEndToEnd(unittest.TestCase):
    """真 LLM 路径离线验证：stub 端点 + OpenAICompatibleProvider + JsonPlanner 整链。"""

    _server: HTTPServer

    @classmethod
    def setUpClass(cls) -> None:
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), _StubLLMHandler)
        cls._port = cls._server.server_address[1]
        threading.Thread(target=cls._server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()

    def test_json_planner_via_stub(self) -> None:
        idx = _index()
        tl = _tl(idx)
        provider = OpenAICompatibleProvider(
            f"http://127.0.0.1:{self._port}/v1", "test-key", "deepseek-chat")
        agent = EditingAgent(provider=provider)
        ops = agent.plan("随便说点什么", tl, idx)
        self.assertEqual(
            ops, [{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}}])
        # 验证 JsonPlanner 正确拼装请求（model + 非空 messages）且携带鉴权头
        self.assertTrue(_StubLLMHandler.auth_seen, "应携带 Authorization 头")
        self.assertEqual(_StubLLMHandler.last_body.get("model"), "deepseek-chat")
        self.assertTrue(_StubLLMHandler.last_body.get("messages"))

    def test_full_url_endpoint_accepted(self) -> None:
        # endpoint 直接填完整 URL 也应工作（兼容性）
        provider = OpenAICompatibleProvider(
            f"http://127.0.0.1:{self._port}/v1/chat/completions", "test-key", "deepseek-chat")
        agent = EditingAgent(provider=provider)
        ops = agent.plan("x", _tl(_index()), _index())
        self.assertEqual(ops[0]["type"], "cut")

    def test_wire_prompt_carries_shot_aggregated_facts(self) -> None:
        """走到线上层（真发 HTTP）确认提示词**内容**没在拼装/序列化环节被丢。

        前面 TestSpeechPromptContract 校验的是 build_prompt 的返回值；
        这里校验的是服务端**实际收到**的请求体 —— 覆盖 provider 拼 messages、
        json.dumps、传输这几步里可能出现的内容丢失（例如字段名写错、
        把 prompt 塞进了 system 却没塞进 user）。
        """
        from ai_video_agent.footage_index import TranscriptSegment
        scenes = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0},
                  {"start": 8.0, "end": 10.0}]
        speech = [TranscriptSegment(start=4.2, end=5.1, text="这里有一辆公交车")]
        idx = _index(dur=10.0, speech=speech, scenes=scenes)
        provider = OpenAICompatibleProvider(
            f"http://127.0.0.1:{self._port}/v1", "test-key", "deepseek-chat")
        EditingAgent(provider=provider).plan('删掉说"公交车"的片段', _tl(idx), idx)

        body = _StubLLMHandler.last_body
        text = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
        self.assertIn("镜头 4.00-8.00s", text, "线上提示词必须带上聚合后的镜头区间")
        self.assertIn("这里有一辆公交车", text, "台词原文必须在提示词里（模型靠它判关键词）")
        self.assertIn('删掉说"公交车"的片段', text, "用户指令必须原样带入")
        self.assertIn("<id>__a", text, "拆分命名约定必须在提示词里")
        self.assertIn("`src_video`", text, "可用 id 清单必须在提示词里")


class _LocalStubHandler(BaseHTTPRequestHandler):
    """本地 Qwen stub：返回 at=1.0 的 cut，并置 called 标记。"""
    called = False

    def do_POST(self) -> None:  # noqa: N802
        _LocalStubHandler.called = True
        data = json.dumps({"choices": [{"message": {"content":
                    json.dumps([{"type": "cut", "payload": {"targetClipId": "src_video", "at": 1.0}}])}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # 静默
        pass


class _CloudStubHandler(BaseHTTPRequestHandler):
    """云端 DeepSeek stub：返回 at=9.0 的 cut，并置 called 标记。"""
    called = False

    def do_POST(self) -> None:  # noqa: N802
        _CloudStubHandler.called = True
        data = json.dumps({"choices": [{"message": {"content":
                    json.dumps([{"type": "cut", "payload": {"targetClipId": "src_video", "at": 9.0}}])}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # 静默
        pass


class _LocalBrokenHandler(BaseHTTPRequestHandler):
    """模拟本地 Qwen 不可达：返回 500，迫使 HybridProvider 兜回云端。"""
    called = False

    def do_POST(self) -> None:  # noqa: N802
        _LocalBrokenHandler.called = True
        self.send_response(500)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args: object) -> None:  # 静默
        pass


class TestHybrid(unittest.TestCase):
    """HybridProvider 路由验证：本地优先 + 云端兜底（离线、双 stub）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.local = ThreadingHTTPServer(("127.0.0.1", 0), _LocalStubHandler)
        cls.cloud = ThreadingHTTPServer(("127.0.0.1", 0), _CloudStubHandler)
        cls.broken = ThreadingHTTPServer(("127.0.0.1", 0), _LocalBrokenHandler)
        for s in (cls.local, cls.cloud, cls.broken):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        cls.local_port = cls.local.server_address[1]
        cls.cloud_port = cls.cloud.server_address[1]
        cls.broken_port = cls.broken.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        for s in (cls.local, cls.cloud, cls.broken):
            s.shutdown()

    def setUp(self) -> None:
        _LocalStubHandler.called = False
        _CloudStubHandler.called = False
        _LocalBrokenHandler.called = False

    def _provs(self, local_port: int, cloud_port: int):
        local = OpenAICompatibleProvider(f"http://127.0.0.1:{local_port}/v1", "", "qwen")
        cloud = OpenAICompatibleProvider(f"http://127.0.0.1:{cloud_port}/v1", "k", "deepseek")
        return local, cloud

    def test_local_preferred(self) -> None:
        local, cloud = self._provs(self.local_port, self.cloud_port)
        prov = HybridProvider(local, cloud, prefer="local")
        ops = EditingAgent(provider=prov).plan("x", _tl(_index()), _index())
        self.assertEqual(ops[0]["payload"]["at"], 1.0)   # 走了本地
        self.assertTrue(_LocalStubHandler.called)
        self.assertFalse(_CloudStubHandler.called)        # 云端未被调用

    def test_cloud_fallback_on_local_error(self) -> None:
        local = OpenAICompatibleProvider(f"http://127.0.0.1:{self.broken_port}/v1", "", "qwen")
        cloud = OpenAICompatibleProvider(f"http://127.0.0.1:{self.cloud_port}/v1", "k", "deepseek")
        prov = HybridProvider(local, cloud, prefer="local")
        ops = EditingAgent(provider=prov).plan("x", _tl(_index()), _index())
        self.assertEqual(ops[0]["payload"]["at"], 9.0)   # 兜回云端
        self.assertTrue(_LocalBrokenHandler.called)
        self.assertTrue(_CloudStubHandler.called)


class _GarbageHandler(BaseHTTPRequestHandler):
    """模拟本地 Qwen 吐出非 JSON 文本（弱模型偶发）：应被 _is_valid_ops 判否并兜回云端。"""
    called = False

    def do_POST(self) -> None:  # noqa: N802
        _GarbageHandler.called = True
        data = json.dumps({"choices": [{"message": {"content": "抱歉，我无法处理该指令。"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # 静默
        pass


class TestHybridCloudPreferred(unittest.TestCase):
    """云端优先 + 本地兜底（推荐策略）路由验证（离线、双/多 stub）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.local = ThreadingHTTPServer(("127.0.0.1", 0), _LocalStubHandler)
        cls.cloud = ThreadingHTTPServer(("127.0.0.1", 0), _CloudStubHandler)
        cls.broken = ThreadingHTTPServer(("127.0.0.1", 0), _LocalBrokenHandler)
        cls.garbage = ThreadingHTTPServer(("127.0.0.1", 0), _GarbageHandler)
        for s in (cls.local, cls.cloud, cls.broken, cls.garbage):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        cls.local_port = cls.local.server_address[1]
        cls.cloud_port = cls.cloud.server_address[1]
        cls.broken_port = cls.broken.server_address[1]
        cls.garbage_port = cls.garbage.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        for s in (cls.local, cls.cloud, cls.broken, cls.garbage):
            s.shutdown()

    def setUp(self) -> None:
        for h in (_LocalStubHandler, _CloudStubHandler, _LocalBrokenHandler, _GarbageHandler):
            h.called = False

    def _provs(self, local_port: int, cloud_port: int):
        local = OpenAICompatibleProvider(f"http://127.0.0.1:{local_port}/v1", "", "qwen")
        cloud = OpenAICompatibleProvider(f"http://127.0.0.1:{cloud_port}/v1", "k", "deepseek")
        return local, cloud

    def test_cloud_preferred(self) -> None:
        local, cloud = self._provs(self.local_port, self.cloud_port)
        prov = HybridProvider(local, cloud, prefer="cloud")
        ops = EditingAgent(provider=prov).plan("x", _tl(_index()), _index())
        self.assertEqual(ops[0]["payload"]["at"], 9.0)   # 走了云端
        self.assertTrue(_CloudStubHandler.called)
        self.assertFalse(_LocalStubHandler.called)        # 本地未被调用

    def test_local_fallback_on_cloud_error(self) -> None:
        local = OpenAICompatibleProvider(f"http://127.0.0.1:{self.local_port}/v1", "", "qwen")
        cloud = OpenAICompatibleProvider(f"http://127.0.0.1:{self.broken_port}/v1", "k", "deepseek")
        prov = HybridProvider(local, cloud, prefer="cloud")
        ops = EditingAgent(provider=prov).plan("x", _tl(_index()), _index())
        self.assertEqual(ops[0]["payload"]["at"], 1.0)   # 兜回本地
        self.assertTrue(_LocalStubHandler.called)
        self.assertTrue(_LocalBrokenHandler.called)

    def test_local_garbage_falls_back_to_cloud(self) -> None:
        # 本地返回非 JSON 文本（弱模型偶发）：_is_valid_ops 判否 -> 兜回云端
        # （本地优先场景下才触发校验兜底：local 先跑且无效，再 try cloud）
        local = OpenAICompatibleProvider(f"http://127.0.0.1:{self.garbage_port}/v1", "", "qwen")
        cloud = OpenAICompatibleProvider(f"http://127.0.0.1:{self.cloud_port}/v1", "k", "deepseek")
        prov = HybridProvider(local, cloud, prefer="local")
        ops = EditingAgent(provider=prov).plan("x", _tl(_index()), _index())
        self.assertEqual(ops[0]["payload"]["at"], 9.0)
        self.assertTrue(_GarbageHandler.called)
        self.assertTrue(_CloudStubHandler.called)


class TestHybridValidator(unittest.TestCase):
    """_is_valid_ops：只接受『含 type 的 JSON 数组』。"""

    def test_accepts_valid(self) -> None:
        self.assertTrue(HybridProvider._is_valid_ops('[{"type":"cut","payload":{}}]'))

    def test_rejects_non_json(self) -> None:
        self.assertFalse(HybridProvider._is_valid_ops("抱歉我无法处理"))

    def test_rejects_non_list(self) -> None:
        self.assertFalse(HybridProvider._is_valid_ops('{"type":"cut"}'))

    def test_rejects_missing_type(self) -> None:
        self.assertFalse(HybridProvider._is_valid_ops('[{"payload":{}}]'))

    def test_accepts_fenced_json(self) -> None:
        self.assertTrue(HybridProvider._is_valid_ops('```json\n[{"type":"trim","payload":{"in":2}}]\n```'))


class _IncompleteDeleteHandler(BaseHTTPRequestHandler):
    """回放**真机 Qwen3-0.6B 的原样输出**：三个公交镜头只给了两个，还带着 ```json 围栏。

    这条回放数据来自 `_verify/verify_llm_real.py` 的真实运行结果，不是编的。
    """

    def do_POST(self) -> None:  # noqa: N802
        content = ('```json\n[\n  {"type": "delete_range",\n   "payload": {"start": 0.0, "end": 4.0}},\n'
                   '  {"type": "delete_range",\n   "payload": {"start": 4.0, "end": 8.0}}\n]\n```')
        data = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # 静默
        pass


class TestSemanticDeleteGuard(unittest.TestCase):
    """模型答错了，UI 必须**说出来并自动纠正** —— 而不是把"漏删/多删"当成成功交付。

    真机实测（本地 Qwen3-0.6B，见 `_verify/verify_llm_real.py`）：
    「删掉说公交车的片段」只删了 3 个镜头里的 2 个，「删掉有公交车的片段」还多删了
    用户没要求的一段 —— **两次都是操作成功、0 警告**。这是最难发现的一类失效：
    时间轴确实改了，成片也出来了，但和用户指令对不上。

    修复（2026-09-24）：发现不一致且能算出确定性标准答案时，**自动把删除区间纠正为
    规则 Planner 的精确结果**（保证交付物正确），同时把"哪里不对"原样摆给用户看。
    仅当算不出标准答案（没有 M2 事实 / 关键词从未命中）时才保持"只警告"。
    """

    _server: HTTPServer

    @classmethod
    def setUpClass(cls) -> None:
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), _IncompleteDeleteHandler)
        cls._port = cls._server.server_address[1]
        threading.Thread(target=cls._server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()

    def _index(self) -> FootageIndex:
        """素材2 的事实：3 个公交镜头（0–4 / 8–12 / 16–20s），台词与视觉都命中。"""
        from ai_video_agent.footage_index import TranscriptSegment
        scenes = [{"start": float(i * 4), "end": float(i * 4 + 4)} for i in range(6)]
        speech = [TranscriptSegment(start=1.28, end=3.35, text="画面里有一辆公交车"),
                  TranscriptSegment(start=9.34, end=11.28, text="又出现了一辆公交车"),
                  TranscriptSegment(start=17.36, end=19.05, text="公交车再次出现")]
        visuals = [VisualSegment(label="bus", start=0.0, end=4.0, source="yolo"),
                   VisualSegment(label="bus", start=8.0, end=12.0, source="yolo"),
                   VisualSegment(label="bus", start=16.0, end=20.0, source="yolo")]
        return _index(dur=24.0, speech=speech, scenes=scenes, visual=visuals)

    def test_ui_status_flags_model_misses_and_invents_ranges(self) -> None:
        idx = self._index()
        # 模型给的是 (0,4) 与 (4,8)：漏了 8–12 / 16–20，还多删了没人要的 4–8
        ops, _, status = do_plan(
            "删掉有公交车的片段", _tl(idx), idx, "cloud",
            f"http://127.0.0.1:{self._port}/v1", "test-key", "fake-model", "", "")
        # 仍要把"哪里不对"摆给用户看（透明，不静默替换）
        self.assertIn("删除区间与素材事实不一致", status)
        self.assertIn("漏删", status)
        self.assertIn("8.00-12.00s", status)
        self.assertIn("16.00-20.00s", status)
        self.assertIn("多删", status)
        self.assertIn("4.00-8.00s", status)
        # 关键修复：系统已自动把模型不一致的删除区间纠正为正确值，
        # 交付物不再是错误的成片（0-4 / 8-12 / 16-20 三段 bus 镜头被正确删光）
        got = [(o["payload"]["start"], o["payload"]["end"])
               for o in ops if o.get("type") == "delete_range"]
        self.assertEqual(got, [(0.0, 4.0), (8.0, 12.0), (16.0, 20.0)])

    def test_rule_result_automatically_applied_on_mismatch(self) -> None:
        """不一致时系统自动改用规则 Planner 的精确结果（不再只摇头）。"""
        idx = self._index()
        ops, _, status = do_plan(
            "删掉有公交车的片段", _tl(idx), idx, "cloud",
            f"http://127.0.0.1:{self._port}/v1", "test-key", "fake-model", "", "")
        self.assertIn("规则 Planner", status)
        # 自动纠正后，删除区间必须等于规则 Planner 确定性算出的精确结果
        self.assertEqual(
            [(o["payload"]["start"], o["payload"]["end"])
             for o in ops if o.get("type") == "delete_range"],
            [(float(s), float(e))
             for (s, e) in expected_semantic_ranges("删掉有公交车的片段", idx)])

    def test_guard_skipped_for_non_semantic_commands(self) -> None:
        """不是语义删除的指令不做这项核对（避免狼来了）。"""
        idx = _index(dur=24.0)
        idx.scenes = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 24.0}]
        ops, _json, status = do_plan(
            "在第2秒切一刀", _tl(idx), idx, "cloud",
            f"http://127.0.0.1:{self._port}/v1", "test-key", "fake-model", "", "")
        self.assertIsNotNone(ops)
        self.assertNotIn("删除区间与素材事实不一致", status)

    def test_ui_warns_and_points_to_m2_when_no_facts(self) -> None:
        """没有 M2 事实时（用户跳过 ②），模型只能臆造 —— UI 必须把它说破。

        这是核对闸最容易漏的一个死角：`expected` 算不出来就直接返回 []，
        于是"什么都没查、什么都没报"，和没有这道闸一模一样。
        """
        idx = _index(dur=24.0)              # 空索引：没跑过 ②，无镜头/标签/台词
        self.assertFalse(idx.scenes)
        _ops, _, status = do_plan(
            "删掉有公交车的片段", _tl(idx), idx, "cloud",
            f"http://127.0.0.1:{self._port}/v1", "test-key", "fake-model", "", "")
        self.assertIn("删除区间与素材事实不一致", status)
        self.assertIn("没有可用的 M2 结果", status)
        self.assertIn("臆测", status)


if __name__ == "__main__":
    unittest.main()
