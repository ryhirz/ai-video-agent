# AI 视频剪辑智能体

> 用自然语言指挥视频剪辑：从**素材理解 → 操作规划 → 事实校验 → 真实出片**的全链路本地智能体。

输入一句话，例如「删掉有公交车的片段」「把字幕改成竖屏 9:16」，系统自动规划剪辑操作、调用 ffmpeg 出片。
配套 **163 个单元测试** 与 **22 个可复跑验收脚本**，并有 14 页汇报 PPT。

---

## 核心能力

| 能力 | 说明 |
|---|---|
| 自然语言剪辑规划 | 一句话 → 结构化剪辑操作（`规则 Planner` 离线可用 / `LLM` 云端或本地端点双通道） |
| 语义删除 | 按**画面**（YOLO 目标检测）与**台词**（ASR 语音转写）定位并删除对应片段 |
| 确定性事实核对 | 把模型「漏删 / 多删」从幻觉里揪出来，并用可计算的精确结果**自动纠正** |
| 常用剪辑操作 | 切分、删除、拼接、竖屏 / 横屏适配、字幕处理 |
| 全链路本地可跑 | 不依赖云端；可选接入本地 LLM（如 Qwen3）走智能规划 |

## 快速开始

```bash
git clone https://github.com/ryhirz/ai-video-agent.git
cd ai-video-agent
```

然后依次**双击**三个脚本：

| 步骤 | 脚本 | 作用 |
|---|---|---|
| 1 | `一键配置环境.bat` | 建虚拟环境 `.venv` + 装依赖 + 校验/补齐 ffmpeg |
| 2 | `一键测试.bat` | 跑全部单元测试（按退出码判定成败） |
| 3 | `一键启动界面.bat` | 启动 Web 界面 → http://127.0.0.1:7860 |

> 需要完整 AI 能力（YOLO 视觉检测 / 语音转写）时，用 `一键配置环境_完整版.bat` 替代第 1 步（额外安装 torch / ultralytics / funasr）。

### 关于 ffmpeg 与模型权重

本仓库是**源码仓库**，按 GitHub 单文件 100MB 硬限制，**未包含**以下二进制（`.gitignore` 已排除）：

- `tools/ffmpeg/bin/`（ffmpeg / ffprobe，各约 156MB）
- `weights/`（yolov8n.pt 视觉权重）

`一键配置环境.bat` 在检测到 ffmpeg 缺失时会**自动下载**（约 80MB，需联网一次）并放入 `tools/ffmpeg/bin/`。
若自动下载失败，也可使用随附的**完整交付包** `AI视频剪辑智能体-源码.zip`（已含 ffmpeg + 权重 + 汇报 PPT，解压即用、完全离线）。

## 目录结构

```
ai-video-agent/
├─ ai_video_agent/        核心源码（11 个模块）
│   ├─ understand.py        素材理解：YOLO 视觉检测 + ASR 转写 + 镜头聚合
│   ├─ agent.py             规划核心：提示词契约 / 语义删除 / 确定性事实核对
│   ├─ operations.py        剪辑操作模型
│   ├─ compiler.py          操作 → ffmpeg 命令编译
│   ├─ render.py            渲染与 ffmpeg 调用
│   ├─ timeline.py          时间轴
│   ├─ ui.py                Gradio Web 界面
│   └─ …                    ingest / footage_index / model_client 等
├─ tests/                 163 个单元测试（pytest）
├─ _verify/               22 个可复跑验收脚本（含 _run_gates.sh 批跑器）
├─ scripts/               素材生成等辅助脚本
├─ tools/ffmpeg/bin/       ffmpeg 工具链（默认不入库，配置脚本自动补齐）
├─ weights/                YOLO 视觉权重（默认不入库）
├─ requirements.txt        基础依赖（必装）
├─ requirements-full.txt   完整依赖（含 AI 模型，可选）
├─ 一键配置环境.bat        一键装基础环境
├─ 一键配置环境_完整版.bat  一键装完整环境
├─ 一键测试.bat            跑全部测试
├─ 一键启动界面.bat        启动 Web 界面
├─ 汇报/                   汇报 PPT（14 页）
└─ 交付文档.md / 交付概览.md / overview.md / 配置说明.md   交付与设计文档
```

## 技术栈

| 层 | 技术 |
|---|---|
| 语言 / 运行 | Python 3.10+（开发验证用 3.13） |
| Web 界面 | Gradio |
| 视频处理 | ffmpeg（本地工具链）+ numpy + opencv |
| AI 理解（可选） | ultralytics（YOLOv8 目标检测）+ funasr（Paraformer 语音转写） |
| LLM 调用 | 标准库 `urllib`，兼容 OpenAI `/chat/completions` 协议（云端或本地端点） |

## 测试与验收

```bash
# 单元测试（完整环境应为 163 passed + 27 subtests）
.venv\Scripts\python.exe -m pytest tests -q

# 确定性验收脚本（离线，9 个，应为 GATES PASSED=9 FAILED=0）
bash _verify/_run_gates.sh
```

- 单元测试分布：`test_agent`(43) · `test_robust_ops`(33) · `test_understand`(29) · `test_ui`(28) · `test_contract`(14) · `test_render_ops`(9) · `test_render`(3) · `test_ingest`(3) · `test_pipeline`(1)
- 基础环境（未装 AI 依赖）下，2 个 YOLO 真实检测用例会**优雅跳过**而非报错

## 文档索引

| 文档 | 内容 |
|---|---|
| `配置说明.md` | 给验收者的一页纸：环境要求 → 一键配置 → 验证 → 启动 → FAQ |
| `交付文档.md` | 完整设计文档：架构、提示词契约、缺陷修复记录 |
| `交付概览.md` | 交付摘要 |
| `overview.md` | 项目全景与验证证据链 |
| `M5_VERIFICATION.md` | M5 阶段验证记录 |

## 工程亮点

- **5 类真实缺陷的闭环修复**（N：语义删除外扩到镜头 / O：ASS 字幕坐标系 / P：ASR 幻觉护栏 + VAD 段落级时间戳 / Q：提示词契约对模型说谎 / R：真机模型漏删多删）——均含回归测试与真机验证记录
- **性能优化约 3.7×**（5.02s → 1.36s）
- **确定性优先的工程理念**：凡是能用代码算出标准答案的，就不只依赖提示词去要求模型

## 环境要求

- **操作系统**：Windows 10/11（本包按 Windows 路径打包，ffmpeg 为 `.exe`）
- **Python**：3.10 及以上，安装时**务必勾选 "Add Python to PATH"**

## 交付物

完整离线包 `AI视频剪辑智能体-源码.zip`（约 134MB）：源码 + 163 测试 + 验收脚本 + 文档 + **ffmpeg 工具链** + 模型权重 + 4 个一键脚本 + 汇报 PPT。
