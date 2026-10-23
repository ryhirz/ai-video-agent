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

第三步新增（流式进度 + 进程级取消，keyword-only 参数，向后兼容）：
- stderr 由读线程按块消费，解析 `time=HH:MM:SS.xx` → 结合 tl.duration 折算
  progress 0.0~1.0，经 on_progress 回调上报（≥0.25s 节流）。
- cancel_check 每 poll_interval 秒轮询一次；命中即两步杀
  （terminate → 宽限 5s → kill），删除本次运行目录，返回 (None, 取消说明)。
- 超时看门狗与 subprocess.run(timeout=600) 语义对齐（同样两步杀）。
"""
from __future__ import annotations

import datetime
import os
import re
import shutil
import subprocess
import threading
import time as _time
from pathlib import Path
from typing import Callable, Optional, Tuple

from .compiler import compile as compile_timeline
from .timeline import Timeline

_TIME_RE = re.compile(r"time=(\d+):(\d+):([\d.]+)")
_KILL_GRACE = 5.0            # terminate 后等待进程退出的宽限秒数
_PROGRESS_MIN_INTERVAL = 0.25  # 进度回调最小间隔（秒），再往下的节流交给上层


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


def _terminate_proc(proc: subprocess.Popen) -> None:
    """两步杀：terminate → 宽限等待 → kill（Windows 上 terminate 即硬杀）。"""
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=_KILL_GRACE)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=_KILL_GRACE)
    except (subprocess.TimeoutExpired, OSError):
        pass


def render(tl: Timeline, ffmpeg_bin: Optional[str] = None,
           base_dir: Optional[str] = None, *,
           on_progress: Optional[Callable[[float], None]] = None,
           cancel_check: Optional[Callable[[], bool]] = None,
           poll_interval: float = 0.5,
           timeout: float = 600.0) -> Tuple[Optional[str], str]:
    """把时间线真正渲染成 MP4。

    返回 (视频绝对路径, 说明)。视频为 None 表示未成功生成（说明里给原因）。
    on_progress(p)：渲染进度 0.0~1.0 回调（节流上报）；
    cancel_check()：返回 True 即两步杀 ffmpeg 并返回取消。
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
    # 进度分母：优先 metadata.duration；缺省时用视频/音频片段最大 out 点近似
    # （speed 效果会缩放实际时长，此处为近似值，足够驱动进度条）
    total_secs = float(tl.metadata.duration or 0.0)
    if total_secs <= 0 and tl.clips:
        total_secs = max((float(c.out or 0.0) for c in tl.clips), default=0.0)

    try:
        proc = subprocess.Popen(
            argv, cwd=str(run_dir), stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
    except Exception as e:  # noqa: BLE001
        _cleanup_empty(run_dir)
        return (None, f"执行 ffmpeg 失败：{type(e).__name__}: {e}")

    stderr_chunks: list = []          # 完整 stderr（失败时取末尾报错）
    last_reported = [0.0]             # 上次进度回调时刻（monotonic）

    def _read_stderr() -> None:
        """消费 ffmpeg stderr：留存全文 + 解析 time= 上报进度。

        按块 os.read（管道有数据即返回，不整块等待）；carry 保留末尾 64 字符
        以拼接跨块的 time= 片段。
        """
        carry = ""
        fd = proc.stderr.fileno()
        while True:
            try:
                chunk = os.read(fd, 4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            stderr_chunks.append(chunk)
            text = carry + chunk.decode("utf-8", errors="replace")
            carry = text[-64:]
            matches = list(_TIME_RE.finditer(text))
            if matches and on_progress and total_secs > 0:
                h, m, s = matches[-1].group(1, 2, 3)
                p = min((int(h) * 3600 + int(m) * 60 + float(s)) / total_secs, 1.0)
                now = _time.monotonic()
                if p >= 1.0 or now - last_reported[0] >= _PROGRESS_MIN_INTERVAL:
                    last_reported[0] = now
                    try:
                        on_progress(p)
                    except Exception:  # noqa: BLE001 —— 进度回调失败不拖垮渲染
                        pass

    reader = threading.Thread(target=_read_stderr, daemon=True,
                              name="ffmpeg-stderr")
    reader.start()

    # 看门狗：正常退出 / 取消 / 超时 三出口
    killed: Optional[str] = None
    start = _time.monotonic()
    while proc.poll() is None:
        if cancel_check is not None:
            try:
                want_cancel = bool(cancel_check())
            except Exception:  # noqa: BLE001 —— 检查器异常视作未取消
                want_cancel = False
            if want_cancel:
                killed = "cancelled"
                _terminate_proc(proc)
                break
        if timeout and _time.monotonic() - start > timeout:
            killed = "timeout"
            _terminate_proc(proc)
            break
        _time.sleep(poll_interval)

    try:
        reader.join(timeout=_KILL_GRACE)
    finally:
        try:
            proc.stderr.close()
        except OSError:
            pass

    if killed == "cancelled":
        shutil.rmtree(run_dir, ignore_errors=True)   # 半成品一并清掉
        return (None, "渲染已取消（用户中止）。")
    if killed == "timeout":
        _cleanup_empty(run_dir)
        return (None, "ffmpeg 超时（>600s），可能是素材过大或滤镜过重。")

    if proc.returncode != 0 or not out_path.exists():
        # 取 ffmpeg 末尾报错信息，便于定位
        full = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        err = full.strip().splitlines()[-15:] if full.strip() else []
        _cleanup_empty(run_dir)
        return (None, "ffmpeg 渲染失败（返回码 %d）：\n%s" % (
            proc.returncode, "\n".join(err[-8:]) or "（无 stderr）"))

    return (str(out_path), f"🎬 成片已生成：{out_path}")
