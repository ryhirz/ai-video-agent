"""M5 Gradio 演示界面（最小可用离线演示）。

架在 M1-M4 契约之上：
- 摄取：ingest.build_footage_index() + build_initial_timeline()
- 可选视觉理解：understand.detect_objects / transcribe（依赖未装时优雅降级，不崩）
- 规划：EditingAgent.plan()  -> [{type, payload}]
    - 默认 RuleBasedPlanner（离线，无需 Key）；
    - 可选真 LLM：云端 DeepSeek / 本地 Qwen / 混合（本地优先+云端兜底）三种模式，
      由标准库 urllib 实现的 OpenAI 兼容 LLMProvider 热插拔进 JsonPlanner，零新依赖。

界面状态用 gr.State 持有 FootageIndex / Timeline / 上一次操作列表，保证多步编辑可叠加。
所有处理函数定义为模块级函数，便于 tests/test_ui.py 直调验证（无需启动 Gradio 服务）。
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .timeline import Timeline
from .footage_index import FootageIndex
from .ingest import build_footage_index, build_initial_timeline
from .understand import detect_objects, transcribe
from .operations import apply_operations
from .compiler import compile as compile_timeline
from .render import render as render_video
from .agent import (EditingAgent, LLMProvider,
                    speech_delete_degrades_to_raw_segments,
                    semantic_delete_discrepancies, expected_semantic_ranges)


# ---------------------------------------------------------------------------
# 可选真 LLM 提供方：OpenAI / DeepSeek / GLM / 本地 Qwen 兼容 /chat/completions 端点
# 纯标准库 urllib 实现，零新依赖；仅当用户在 UI 选择非规则模式并填好端点/Key 时启用。
# ---------------------------------------------------------------------------
class OpenAICompatibleProvider(LLMProvider):
    """实现 LLMProvider.complete()：向 OpenAI 兼容端点 post 聊天补全，返回助手文本。"""

    def __init__(self, endpoint: str, api_key: str, model: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"

    def complete(self, prompt: str, *, system: Optional[str] = None, timeout: int = 60) -> str:
        from urllib import request as ureq, error as uerr

        # endpoint 兼容两种填法：裸 base（自动补 /chat/completions）或完整 URL
        base = self.endpoint.rstrip("/")
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }).encode("utf-8")

        req = ureq.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        ctx = ssl.create_default_context()
        try:
            with ureq.urlopen(req, timeout=timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except uerr.HTTPError as e:  # 端点返回非 2xx：把错误体透出，便于排查
            detail = e.read().decode("utf-8", "ignore")[:500]
            raise RuntimeError(f"LLM 端点返回 HTTP {e.code}: {detail}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"LLM 返回结构异常，无法解析：{data}") from e


class HybridProvider(LLMProvider):
    """云端+本地混合调度：prefer='local' 时本地优先，本地不可达/报错则兜回云端。

    零新依赖（复用 OpenAICompatibleProvider + 标准库）。适用于「本地 Qwen 离线免费 /
    云端 DeepSeek 兜底提质」的组合场景；双路任一就绪即可用，单路也能独立工作。
    """

    def __init__(self, local: OpenAICompatibleProvider, cloud: OpenAICompatibleProvider,
                 prefer: str = "local") -> None:
        self.local = local
        self.cloud = cloud
        self.prefer = prefer

    @staticmethod
    def _is_valid_ops(text: str) -> bool:
        """本地弱模型可能吐出非 JSON 文本；严格校验其输出可解析为操作列表，
        否则视为失败、兜回云端（仅在首选=本地时触发）。"""
        t = text.strip()
        if t.startswith("```"):                      # 容忍 ```json 代码块包裹
            import re as _re
            t = _re.sub(r"^```[a-zA-Z]*\n?", "", t)
            t = _re.sub(r"\n?```$", "", t).strip()
        try:
            data = json.loads(t)
        except Exception:
            return False
        if not isinstance(data, list):
            return False
        return all(isinstance(o, dict) and "type" in o for o in data)

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        order = (self.local, self.cloud) if self.prefer == "local" else (self.cloud, self.local)
        last_err: Optional[Exception] = None
        for prov in order:
            try:
                # 本地优先给 30s（含首次懒加载模型），云端用完整 60s
                to = 30 if prov is self.local else 60
                text = prov.complete(prompt, system=system, timeout=to)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            # 仅对本地输出做严格可解析校验：不可解析则兜回云端（弱模型容错）
            if prov is self.local and not self._is_valid_ops(text):
                last_err = RuntimeError("本地 Qwen 输出无法解析为操作 JSON，兜回云端")
                continue
            return text
        raise RuntimeError(f"HybridProvider 双路均失败：{last_err}")


def _make_agent(*, mode: str = "rule",
                cloud_endpoint: str = "", cloud_key: str = "", cloud_model: str = "",
                local_endpoint: str = "", local_key: str = "", local_model: str = "") -> EditingAgent:
    """按 mode 选择 Planner 后端：
    - rule   : RuleBasedPlanner（离线，无需 Key）
    - local  : 仅本地 Qwen（OpenAI 兼容端点）
    - cloud  : 仅云端（DeepSeek/GLM...）——默认
    - hybrid : 云端优先 + 本地兜底（离线免费 + 云端提质）
    """
    cloud = OpenAICompatibleProvider(cloud_endpoint, cloud_key, cloud_model) if (cloud_endpoint and cloud_key) else None
    local = OpenAICompatibleProvider(local_endpoint, local_key, local_model) if local_endpoint else None
    if mode == "cloud" and cloud:
        return EditingAgent(provider=cloud)
    if mode == "local" and local:
        return EditingAgent(provider=local)
    if mode == "hybrid" and local and cloud:
        # 实测定稿：云端优先（保证编辑正确）+ 本地兜底（断网/离线仍可用）
        return EditingAgent(provider=HybridProvider(local, cloud, prefer="cloud"))
    if mode == "hybrid" and cloud:            # 本地未启动时，混合模式自动退化为纯云端
        return EditingAgent(provider=cloud)
    return EditingAgent()


# ---------------------------------------------------------------------------
# 展示辅助
# ---------------------------------------------------------------------------
def _fmt_metadata(idx: FootageIndex) -> str:
    m = idx.metadata
    return "\n".join([
        f"- 文件：`{os.path.basename(idx.filepath)}`",
        f"- 时长：**{m.get('duration', 0):.2f}s**",
        f"- 分辨率：{m.get('width', '?')} × {m.get('height', '?')}",
        f"- 帧率：{m.get('fps', '?')} fps",
        f"- 视频编码：{m.get('video_codec', '-')}",
        f"- 音频：{m.get('audio_codec', '-') or '无'}（{'有音轨' if m.get('has_audio') else '无音轨'}）",
        f"- 镜头数：{len(idx.scenes)} · 关键帧数：{len(idx.keyframes)}",
    ])


# ---------------------------------------------------------------------------
# 处理函数（模块级，供 UI 接线 & 测试直调）
# ---------------------------------------------------------------------------
def do_ingest(video_path: Optional[str]) -> Tuple[Any, Any, str, str, str]:
    """摄取：ffprobe 元数据 + 镜头检测 + 初始 Timeline。"""
    if not video_path:
        return (None, None, "⚠️ 请先上传视频。", "", "")
    try:
        index = build_footage_index(video_path)
        tl = build_initial_timeline(index)
        meta = _fmt_metadata(index)
        scenes = "\n".join(
            f"- 镜头 {i + 1}: {s['start']:.2f}s – {s['end']:.2f}s"
            for i, s in enumerate(index.scenes)
        ) or "（无镜头切变）"
        kf = ", ".join(f"{t:.2f}s" for t in index.keyframes) or "（无）"
        return (index, tl, meta, scenes, kf)
    except Exception as e:  # noqa: BLE001
        return (None, None, f"❌ 摄取失败：{e}", "", "")


def do_understand(index: Optional[FootageIndex]) -> Tuple[Any, str]:
    """可选 M2：YOLOv8 视觉段 + FunASR 转写；任一依赖缺失均优雅降级。

    耗时提示：首次调用要加载 YOLO 权重 + Paraformer 权重（约 20–40s，含一次性
    模型下载），之后同进程复用缓存（见 understand._YOLO_CACHE / _ASR_CACHE），
    重复点击约 3–8s。这里把耗时写进返回文案，避免用户以为"点了没反应"。
    """
    if index is None:
        return (None, "⚠️ 请先点上面的『① 摄取（ffprobe+镜头）』生成时间轴，再点本按钮。"
                      "（M2 依赖摄取结果，单独点它不会执行）")
    parts: List[str] = []
    t_all = time.time()
    try:
        t0 = time.time()
        segs = detect_objects(index)
        index.visual_segments = segs
        if segs:
            parts.append("**视觉段（YOLOv8）：**\n" + "\n".join(
                f"- `{s.label}` {s.start:.2f}s–{s.end:.2f}s (conf={s.confidence:.2f})"
                for s in segs))
        else:
            parts.append("（未检出物体；可换含人物的视频重试）")
        parts.append(f"*(YOLOv8 用时 {time.time() - t0:.1f}s)*")
    except ImportError as e:
        parts.append(f"⚠️ YOLOv8 依赖未安装（ultralytics/torch）：{e}\n在 venv 安装后可重试。")
    except Exception as e:  # noqa: BLE001
        parts.append(f"❌ 视觉检测失败：{type(e).__name__}: {e}")
    try:
        t0 = time.time()
        tr = transcribe(index)
        index.transcript_segments = tr
        if tr:
            parts.append("**转写段（FunASR，时间戳 = VAD 切出的语音段落边界）：**\n" + "\n".join(
                f"- {t.start:.2f}s–{t.end:.2f}s: {t.text}" for t in tr))
        elif not index.metadata.get("has_audio"):
            parts.append("ℹ️ 该素材无音轨，转写跳过。")
        else:
            parts.append("（未转出文本：该音轨没有检出有效语音，或结果被判为幻觉"
                         "——纯音乐/无语音素材常见此情况）")
        parts.append(f"*(FunASR 用时 {time.time() - t0:.1f}s；模型已缓存，重复点击更快)*")
    except ImportError as e:
        parts.append(f"ℹ️ FunASR 未安装（转写跳过）：{e}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"❌ 转写失败：{type(e).__name__}: {e}")
    parts.append(f"*(M2 总用时 {time.time() - t_all:.1f}s)*")
    return (index, "\n\n".join(parts))


def do_plan(nl: str, timeline: Optional[Timeline], index: Optional[FootageIndex],
            mode: str, cloud_endpoint: str, cloud_key: str, cloud_model: str,
            local_endpoint: str, local_model: str
            ) -> Tuple[Any, str, str]:
    """自然语言 -> 结构化操作列表 [{type, payload}]。按 mode 选后端。"""
    if timeline is None:
        return (None, "", "⚠️ 请先『摄取』视频。")
    if not nl or not nl.strip():
        return (None, "", "⚠️ 请输入自然语言剪辑指令。")
    try:
        # 云端 Key 允许通过环境变量 DEEPSEEK_API_KEY 注入，UI 不强制填写
        cloud_key = cloud_key or os.environ.get("DEEPSEEK_API_KEY", "")
        agent = _make_agent(mode=mode, cloud_endpoint=cloud_endpoint, cloud_key=cloud_key,
                            cloud_model=cloud_model, local_endpoint=local_endpoint,
                            local_model=local_model)
        label = {"rule": "规则 Planner（离线）", "local": "本地 Qwen（JsonPlanner）",
                 "cloud": "云端 DeepSeek（JsonPlanner）",
                 "hybrid": "混合（云端优先+本地兜底）"}.get(mode, mode)
        ops = agent.plan(nl, timeline, index)
        if not ops:
            return (None, "", f"**（{label}）** 未匹配到操作；可换种说法或参考示例指令。")
        # 两道"看得见的兜底"（不要把模型/退化的输出当成可信结果直接交付）：
        notes: List[str] = []
        # ① 按台词删除但没有镜头信息时，区间只能取语音段落（约 1–2s）→ 会留下半个镜头。
        if speech_delete_degrades_to_raw_segments(nl, index):
            notes.append("⚠️ 本次没有可用的镜头信息，删除区间只能取**语音段落**（约 1–2 秒），"
                         "成片可能残留同一镜头的其余画面。建议先跑 ② M2 理解（或检查镜头检测）。")
        # ② 语义删除：拿 M2 事实核对模型给出的区间 —— 真机实测模型会漏删/多删，
        #    而这两者**都不报错、0 警告**，只能靠这道闸把它们摆到台面上。
        diff = semantic_delete_discrepancies(nl, index, ops)
        if diff:
            expected = expected_semantic_ranges(nl, index)
            if expected:
                # 有确定性标准答案（由 YOLO 标签 + ASR 事实算出"该删哪些区间"）→
                # 直接把模型不一致的删除区间**自动纠正**为正确值，保证交付物是对的；
                # 同时把"哪里不对"原样摆给用户看（透明，不是静默替换）。
                non_delete = [o for o in ops
                              if not (isinstance(o, dict) and o.get("type") == "delete_range")]
                corrected = [{"type": "delete_range",
                              "payload": {"start": float(s), "end": float(e)}}
                             for (s, e) in expected]
                ops = non_delete + corrected
                notes.append(
                    "⚠️ **已自动改用精确（规则 Planner）结果**：模型规划的删除区间与素材事实"
                    f"不一致（{len(diff)} 处），系统已按 YOLO 标签 + ASR 事实把删除区间修正为正确值。\n"
                    + "\n".join(f"  · {d}" for d in diff))
            else:
                # 算不出标准答案（没有 M2 事实 / 关键词从未命中）→ 无法自动纠正，
                # 只能提示用户先跑 ② 理解或换说法，保持"只警告"。
                notes.append("⚠️ **删除区间与素材事实不一致**（按实测期望值核对）：\n"
                             + "\n".join(f"  · {d}" for d in diff)
                             + "\n  建议先跑 ② M2 理解（或检查镜头检测）拿到事实依据，再回来规划。")
        return (ops,
                json.dumps(ops, ensure_ascii=False, indent=2),
                f"**（{label}）** 生成 {len(ops)} 条操作。"
                + (("\n\n" + "\n\n".join(notes)) if notes else ""))
    except Exception as e:  # noqa: BLE001
        return (None, "",
                f"❌ 生成操作失败：{e}\n```\n{traceback.format_exc()}\n```")


# ---------------------------------------------------------------------------
# ④「应用并编译」的输出契约
# 返回值元组的顺序必须与下方 btn_apply.click(outputs=[...]) 的组件顺序逐位一致。
# 用这一个常量承载顺序、返回值按它摊平，避免「输出列表里插了个新组件、返回元组却把新值
# 追加在末尾」造成整批错位 —— gr.Video 会拿到一段 JSON 当文件路径，界面整块渲染成错误卡。
# tests/test_ui.py::TestUIOutputContract 会按接线组件类型校验该顺序。
# ---------------------------------------------------------------------------
APPLY_OUTPUT_KEYS: Tuple[str, ...] = (
    "timeline_state",   # gr.State    -> 新 Timeline
    "apply_status",     # gr.Markdown -> 状态文案
    "video_out",        # gr.Video    -> 成片路径（"" 表示未出片）
    "tl_json",          # gr.Code     -> 结果 Timeline JSON
    "cmd_box",          # gr.Code     -> FFmpeg 命令
    "ass_box",          # gr.Code     -> 字幕 ASS
)


def _video_clip_count(tl: Timeline) -> int:
    """统计时间轴上还剩几条视频 clip（用于「操作把视频删光了」的安全闸）。"""
    ttype = {t.id: t.type for t in tl.tracks}
    return sum(1 for c in tl.clips if ttype.get(c.track, "video") == "video")


def do_apply_compile(timeline: Optional[Timeline], ops: Optional[List[Dict[str, Any]]]
                     ) -> Tuple[Any, str, str, str, str, str]:
    """应用操作 -> 新 Timeline -> 确定性编译 -> 真正渲染成 MP4。

    走 operations.apply_operations（非严格模式）：
    - 自动把「被 cut 拆掉的旧 id」（如 src_video）解析到当前真实 id；
    - 单条操作失败只跳过并列出原因，不再让整批以裸 KeyError 崩掉；
    - 若操作把视频片段全删光，直接判失败并说明，不产出无视频的空命令。
    编译通过后用 render 真正调用 ffmpeg 出片（尽力而为，失败不影响编译状态）。

    返回值顺序严格由 APPLY_OUTPUT_KEYS 决定（= 下方 click 的 outputs 组件顺序），
    不要手写元组字面量，否则插入新组件时极易整批错位。
    """
    def out(**kw: Any) -> Tuple[Any, ...]:
        """按 APPLY_OUTPUT_KEYS 摊平成 Gradio 需要的元组（顺序的唯一来源）。"""
        return tuple(kw[k] for k in APPLY_OUTPUT_KEYS)

    empty = dict(video_out="", tl_json="", cmd_box="", ass_box="")
    if timeline is None:
        return out(timeline_state=timeline, apply_status="⚠️ 请先『摄取』视频。", **empty)
    if not ops:
        return out(timeline_state=timeline, apply_status="⚠️ 请先『生成操作』得到操作列表。",
                   **empty)
    try:
        tl, logs, warns = apply_operations(timeline, ops, strict=False)

        if _video_clip_count(tl) == 0:
            return out(timeline_state=timeline,
                       apply_status=(
                           "❌ 未编译：这些操作把时间轴上的视频片段全部删除了。\n"
                           "多半是规划器把 `src_video` 整体 delete 掉了（应当只删目标区间）。\n"
                           "请改指令重试，例如：`删掉有bus的片段`（需先跑 M2）或 `去掉前2秒`。\n\n"
                           + "\n".join(logs + warns)),
                       **empty)

        result = compile_timeline(tl)
        head = f"✅ 已应用并编译（视频片段 {_video_clip_count(timeline)} → {_video_clip_count(tl)}）。"
        status = head + "\n" + "\n".join(logs)
        if warns:
            status += ("\n\n⚠️ 有 " + str(len(warns)) + " 条操作被跳过（其余操作已生效）：\n"
                       + "\n".join(f"- {w}" for w in warns))
        # 编译器认不出/表达不了的东西必须回显，不能让人以为"已应用"就万事大吉
        unsupported = result.get("unsupported") or []
        if unsupported:
            status += ("\n\n⚠️ 以下内容不会体现在成片里：\n"
                       + "\n".join(f"- {u}" for u in unsupported))

        # 真正出片：render 调用 ffmpeg；失败则在状态里说明，但保留已成功的编译结果
        video_path, render_msg = render_video(tl)
        if video_path:
            status += "\n\n" + render_msg
        else:
            status += ("\n\n⚠️ 视频渲染未执行（编译指令已正确生成，可手动跑 command 出片）："
                       + render_msg)

        return out(timeline_state=tl, apply_status=status, video_out=video_path or "",
                   tl_json=tl.dumps(), cmd_box=result["command"],
                   ass_box=result.get("ass", "") or "（无字幕）")
    except Exception as e:  # noqa: BLE001
        return out(timeline_state=timeline,
                   apply_status=(f"❌ 应用/编译失败：{type(e).__name__}: {e}\n"
                                 f"```\n{traceback.format_exc()}\n```"),
                   **empty)


def do_reset() -> Tuple[Any, Any, Any, Any, str, str, str, str, str, str, str, str, str, str, str]:
    """清空全部状态与展示（video 用 None 复位，避免 gr.update 版本依赖）。"""
    return (None, None, None, None,
            "摄取后显示元数据…", "", "", "运行 M2 后显示视觉段…",
            "", "", "", "", "", "", "")


# ---------------------------------------------------------------------------
# 界面装配
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI 视频剪辑智能体 · M5 离线演示") as demo:
        gr.Markdown("# 🎬 AI 视频剪辑智能体 · 离线演示 (M5)")
        gr.Markdown(
            "自然语言 → 结构化 Edit Operation → 确定性 FFmpeg 编译。"
            "**默认走仅云端 SiliconFlow（DeepSeek-V4-Flash），联网即用、编辑正确率高**；"
            "也可切『混合』（云端优先+本地兜底）/『规则』（离线无需 Key）/『仅本地 Qwen』。")

        # 状态
        index_state = gr.State(None)
        timeline_state = gr.State(None)
        ops_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 素材摄取与理解")
                video = gr.Video(label="① 上传视频", sources=["upload"], height=240)
                with gr.Row():
                    btn_ingest = gr.Button("① 摄取（ffprobe+镜头）", variant="primary")
                    btn_m2 = gr.Button("② 跑视觉理解（M2·可选）", variant="secondary")
                gr.Markdown("*② 需先完成①。首次运行要加载 YOLO + Paraformer 权重"
                            "（约 20–40 秒，会显示进度）；之后模型有缓存，重复点击约 3–8 秒。*")
                md_meta = gr.Markdown("摄取后显示元数据…")
                scenes_box = gr.Code(label="镜头区间")
                kf_box = gr.Code(label="关键帧时间戳")
                vis_box = gr.Markdown("运行 M2 后显示视觉段…")

                with gr.Accordion("真 LLM（可选·云端+本地混合）", open=False):
                    gr.Markdown("默认仅云端 SiliconFlow（DeepSeek-V4-Flash），联网即用、编辑正确率高。"
                                "也可选『混合』（云端优先+本地兜底）/『规则（离线）』/『仅本地 Qwen』。"
                                "本地 Qwen 已就位：`E:\\AI_models\\Qwen_Model\\start_qwen.bat` 启动即监听"
                                " `http://127.0.0.1:8001/v1`（与 PAI 文生图 8000 错开）。"
                                "云端 Key 也可用环境变量 `DEEPSEEK_API_KEY` 注入，无需在此填写。")
                    llm_mode = gr.Radio(
                        choices=[("仅云端 SiliconFlow（推荐）", "cloud"),
                                 ("混合：云端优先+本地兜底", "hybrid"),
                                 ("规则（离线·无需Key）", "rule"),
                                 ("仅本地 Qwen", "local")],
                        value="cloud", label="LLM 后端模式")
                    with gr.Row():
                        cloud_ep = gr.Textbox(
                            label="云端 Endpoint（兼容 /chat/completions）",
                            value="https://api.siliconflow.cn/v1")
                        cloud_model = gr.Textbox(
                            label="云端 Model", value="deepseek-ai/DeepSeek-V4-Flash")
                    cloud_key = gr.Textbox(
                        label="云端 API Key（或留空读 DEEPSEEK_API_KEY 环境变量）",
                        type="password", placeholder="sk-...")
                    with gr.Row():
                        local_ep = gr.Textbox(
                            label="本地 Qwen Endpoint（OpenAI 兼容）",
                            value="http://127.0.0.1:8001/v1")
                        local_model = gr.Textbox(
                            label="本地 Model", value="qwen3-0.6b")

            with gr.Column(scale=1):
                gr.Markdown("### 编辑指令与编译")
                nl = gr.Textbox(
                    label="③ 自然语言剪辑指令",
                    placeholder='例如：在第2秒切一刀 / 去掉前2秒 / 加字幕"标题" / '
                                '删掉有bus的片段 / 删掉说"公交车"的片段',
                    lines=2)
                gr.Markdown(
                    "示例：①`在第2秒切一刀` ②`去掉前2秒` ③`去掉后3秒` "
                    "④`加字幕\"开场\"` ⑤`删掉有bus的片段` "
                    "⑥`删掉说\"公交车\"的片段`（按**台词**删，需先跑 ② M2） ⑦`删除片段:src_audio`"
                    "\n\n⑤⑥ 都**需先跑 ② M2**：⑤ 靠 YOLO 找到 bus 在第几秒，"
                    "⑥ 靠语音转写找到哪几秒说了这句话。**删除区间一律按镜头对齐**，"
                    "不会只删半句/半个画面。"
                    "\n\n若提示「片段已被切分，自动解析为 xxx」，说明规划器引用了切分前的旧 id，"
                    "系统已自动纠正，操作照常生效。")
                with gr.Row():
                    btn_plan = gr.Button("③ 生成操作", variant="primary")
                    btn_apply = gr.Button("④ 应用并编译", variant="primary")
                    btn_reset = gr.Button("重置")
                ops_json = gr.Code(label="③ 操作列表 [{type,payload}]", language="json")
                plan_status = gr.Markdown("")
                apply_status = gr.Markdown("")
                video_out = gr.Video(label="④ 成片预览 / 下载（真正渲染出的 MP4）", height=360)
                tl_json = gr.Code(label="④ 结果 Timeline (JSON)", language="json")
                cmd_box = gr.Code(label="④ FFmpeg 命令（可复制执行）", language="shell")
                ass_box = gr.Code(label="④ 字幕 ASS（如有）")

        # 接线（开启队列：长任务会显示进度，不会"点了没反应"）
        demo.queue()
        btn_ingest.click(
            do_ingest, [video],
            [index_state, timeline_state, md_meta, scenes_box, kf_box])
        btn_m2.click(do_understand, [index_state], [index_state, vis_box],
                     show_progress="full")
        btn_plan.click(
            do_plan, [nl, timeline_state, index_state, llm_mode, cloud_ep, cloud_key,
                      cloud_model, local_ep, local_model],
            [ops_state, ops_json, plan_status])
        btn_apply.click(
            do_apply_compile, [timeline_state, ops_state],
            # ⚠️ 顺序必须与 do_apply_compile 的 APPLY_OUTPUT_KEYS 逐位一致（改动请同步常量，
            #    否则 gr.Video 会拿到 JSON 之类的错值、整块渲染成错误卡）
            [timeline_state, apply_status, video_out, tl_json, cmd_box, ass_box],
            show_progress="full")
        btn_reset.click(
            do_reset, None,
            [index_state, timeline_state, ops_state, video, md_meta, scenes_box,
             kf_box, vis_box, ops_json, plan_status, apply_status, video_out,
             tl_json, cmd_box, ass_box])

        gr.Markdown(
            "---\n运行：`cd ai-video-agent && .venv/Scripts/python.exe -m ai_video_agent.ui`"
            " · 架构铁律：Timeline 单一真相 / AI 只发结构化操作 / 编译器确定性可单测。")
    return demo


def _pick_free_port(preferred: int = 7860, span: int = 20) -> int:
    """返回可用端口：优先 preferred，被占用则向后顺延。

    若设置了环境变量 GRADIO_SERVER_PORT 则直接采用（显式覆盖）。
    注意：Windows 下不设 SO_REUSEADDR —— 否则会"抢占"已在监听的端口导致冲突。
    """
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port and env_port.isdigit():
        return int(env_port)
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"未找到空闲端口：{preferred}-{preferred + span - 1}")


if __name__ == "__main__":
    port = _pick_free_port(7860)
    if port != 7860:
        print("=" * 52)
        print(f"[提示] 端口 7860 已被占用（多半是上次没关干净的旧实例）")
        print(f"[提示] 本次自动改用 {port}，但建议关掉旧窗口再重启以释放 7860")
        print("=" * 52)
    # show_error=True：出错时把真实异常显示在界面上，而不是只弹一张"错误"卡
    # （曾因一张无信息的错误卡排查了很久）。对外演示若不想露堆栈，设为 False 即可。
    build_app().launch(server_name="127.0.0.1", server_port=port, inbrowser=True,
                       show_error=True)
