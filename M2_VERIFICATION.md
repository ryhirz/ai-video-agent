# M2 视觉理解层 · 冒烟验证报告

**日期**：2026-09-18  ·  **里程碑**：M1→M2→M3 全链在真实媒体上闭环 ✅

## 验证结论

M2 冒烟测试 `tests/test_understand.py` **3 passed**；全量回归 **16 passed（8.82s）**。

| 层 | 测试 | 结果 |
|----|------|------|
| M3 契约（冻结） | test_contract.py ×9 | ✅ |
| M1 摄取 | test_ingest.py ×3 | ✅ |
| M1→M3 联调 | test_pipeline.py ×1 | ✅ |
| M2 理解 | test_understand.py ×3 | ✅ |

## 真实检测证据

YOLOv8（yolov8n.pt）在 `bus.jpg` 循环生成的 2s 无声测试片上跑通：

- keyframes = `[1.0]`（静态单场景，取场景中点的单关键帧）
- **DETECTED_LABELS = `['bus', 'dog', 'person']`**（3 个 visual_segments）
- `has_audio = False` → `transcribe()` 直接返回 `[]`（管线接线正确，未触发 FunASR 权重下载）

→ 证明「自然语言检索素材」所需的 visual_segments 已由真实模型产出，架构链路（摄取→理解→结构化操作→确定性编译）在真实帧上打通。

## 权重获取（关键网络踩坑，已沉淀到项目记忆）

- 本沙箱 `release-assets.githubusercontent.com`（GitHub 文件 CDN）**DNS 不可解析** → ultralytics 自动下载 yolov8n.pt 必挂。
- 公开代理 `ghproxy.net` 可慢速续传（~17KB/s，6.3MB 约 5min，支持 `-C -`）；`ghfast.top`/`gh-proxy.com` 不可用。
- HF 镜像 / ModelScope 均无官方 `ultralytics/yolov8n` 仓库。
- **已落地**：`ai-video-agent/yolov8n.pt`（根，6,549,796B，PK.. 有效）+ `weights/yolov8n.pt` 双份；`bus.jpg` 拷自 `site-packages/ultralytics/assets/`。

## 真实转写证据（2026-09-19 补）

带音轨样本 `tests/.cache/sample_with_audio.mp4`（ffmpeg 由 testsrc 视频 + edge-tts 中文语音混流，含 AAC 音轨，时长 6.24s）实跑 `scripts/paraformer_demo.py`：

- `has_audio = True`，`audio_codec = aac`，`duration = 6.24`
- Paraformer（`paraformer`/`v2.0.4`，ModelScope 权重下载成功）转写输出：
  `今天天气真好我们一起来剪辑这段视频把精彩的片段保留下来`
- 与合成原文（「今天天气真好，我们一起来剪辑这段视频，把精彩的片段保留下来。」）逐字一致，仅缺标点逗号（Paraformer 默认不出标点），**ASR 整链在本环境实跑通过**。
- 性能：`rtf ≈ 0.356`（CPU 推理 2.2s / 6.24s 音频）。
- 依赖：`torch`+`funasr`+`modelscope`+`kaldi-native-fbank`（fbank 后端，替代 torchaudio）。

## 待解 / 下一步

1. ~~**FunASR 真实转写**~~：✅ 已完成（见上「真实转写证据」）。
2. **M4/M5 真 LLM 后端**：已实现「云端+本地混合」`HybridProvider`（本地优先+云端兜底），离线双 stub 路由验证通过（见 `M5_VERIFICATION.md`）。**当前状态**：用户云端 Key 在 `api.deepseek.com` 返回 401（疑似第三方网关 Key，需填正确 Endpoint）；本地 Qwen 未运行（启动后填入即接入混合）。
3. **M5（Gradio UI）**：已交付并升级混合后端（见 `M5_VERIFICATION.md`）。
