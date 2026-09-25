"""渲染层：把 compiler.compile() 产出的 ffmpeg 指令真正执行，输出 MP4。

这是 M5 演示"确定性编译 → 真实成片"的最后一段：UI 跑完 ④ 应用并编译后，
这里把时间线真正剪出来，返回一个可预览/下载的视频文件路径。

设计要点：
- ffmpeg 二进制定位顺序：① 显式参数 ② 环境变量 FFMPEG_BIN ③ 项目自带
  tools/ffmpeg/bin/ffmpeg(.exe)（本机 ffmpeg 没进 PATH，靠这个）④ 系统 PATH。
- 用按次时间戳子目录 + cwd 运行：输出文件名走相对名（output.mp4 / subs.ass），
  避开 Windows 绝对路径在 filtergraph 里的 `:` 转义坑。
- 渲染是"尽力而为"：ffmpeg 缺失或输入文件不存在时返回 (None, 原因)，
  不影响上层"应用+编译"已成功的状态展示。
"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from .compiler import compile as compile_timeline
from .timeline import Timeline


def resolve_ffmpeg(explicit: Optional[str] = None) -> Optional[str]:
    """按优先级定位 ffmpeg 可执行文件，找不到返回 None。"""
    root = Path(__file__).resolve().parent.parent  # ai-video-agent/
    candidates: list = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("FFMPEG_BIN")
    if env:
        candidates.append(env)
    candidates.append(str(root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"))
    candidates.append(str(root / "tools" / "ffmpeg" / "bin" / "ffmpeg"))
    for c in candidates:
        if c and Path(c).exists():
            return c
    # 最后回退到系统 PATH
    from shutil import which
    return which("ffmpeg") or which("ffmpeg.exe")


def _cleanup_empty(run_dir: Path) -> None:
    """渲染失败时清掉本次运行目录（只含我们的临时文件，无成片）。"""
    try:
        if run_dir.exists() and not (run_dir / "output.mp4").exists():
            shutil.rmtree(run_dir, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def render(tl: Timeline, ffmpeg_bin: Optional[str] = None,
           base_dir: Optional[str] = None) -> Tuple[Optional[str], str]:
    """把时间线真正渲染成 MP4。

    返回 (视频绝对路径, 说明)。视频为 None 表示未成功生成（说明里给原因）。
    """
    ffmpeg = resolve_ffmpeg(ffmpeg_bin)
    if not ffmpeg:
        return (None, "未找到 ffmpeg：请把 ffmpeg 放进 tools/ffmpeg/bin/ 或设置 FFMPEG_BIN。")

    root = Path(__file__).resolve().parent.parent
    base = Path(base_dir) if base_dir else (root / "outputs")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    result = compile_timeline(tl)
    argv = list(result["argv"])  # ["ffmpeg", "-i", ...]，-y output.mp4
    # compile() 的命令首元是字面量 "ffmpeg"（按 PATH 假设）；本机 ffmpeg 没进 PATH，
    # 这里用定位到的绝对路径替换 argv[0]，避免 FileNotFoundError。
    argv[0] = ffmpeg

    # 字幕：把 ASS 写到运行目录，filtergraph 用相对名引用，避免绝对路径转义问题
    if result.get("ass"):
        (run_dir / "subs.ass").write_text(result["ass"], encoding="utf-8")

    out_path = run_dir / "output.mp4"
    try:
        proc = subprocess.run(
            argv, cwd=str(run_dir), capture_output=True, text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        _cleanup_empty(run_dir)
        return (None, "ffmpeg 超时（>600s），可能是素材过大或滤镜过重。")
    except Exception as e:  # noqa: BLE001
        _cleanup_empty(run_dir)
        return (None, f"执行 ffmpeg 失败：{type(e).__name__}: {e}")

    if proc.returncode != 0 or not out_path.exists():
        # 取 ffmpeg 末尾报错信息，便于定位
        err = proc.stderr.strip().splitlines()[-15:] if proc.stderr else []
        _cleanup_empty(run_dir)
        return (None, "ffmpeg 渲染失败（返回码 %d）：\n%s" % (
            proc.returncode, "\n".join(err[-8:]) or "（无 stderr）"))

    return (str(out_path), f"🎬 成片已生成：{out_path}")
