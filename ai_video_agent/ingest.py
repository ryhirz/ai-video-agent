"""M1 摄取理解层：把原始视频变成结构化 Footage Index。

职责（对应 PRD §7.1 M1）：
- ffprobe 取元数据（duration/fps/宽高/编码/声道）；
- 镜头边界检测（基于 OpenCV 帧直方图差，轻量可控，不依赖 scenedetect 版本 API）；
- 关键帧时间戳采样（场景中点 + 均匀采样），供 M2 的 YOLOv8 抽帧检测；
- 桥接：build_initial_timeline() 把索引变成 M3 的「原始 Timeline」，Agent 在其上做 cut/trim。

依赖：ffmpeg/ffprobe 静态版（置于 <项目>/tools/ffmpeg/bin），opencv-python（cv2）。
重型依赖（cv2）在函数内惰性 import，保证 footage_index 与 schema 层可独立单测。
"""
from __future__ import annotations

import os
import json
import subprocess
from typing import Any, Dict, List

from .timeline import Timeline, Track, Clip, Metadata
from .footage_index import FootageIndex


# ---------------------------------------------------------------------------
# ffmpeg/ffprobe 路径解析：优先 <项目>/tools/ffmpeg/bin，其次系统 PATH
# ---------------------------------------------------------------------------
def ffmpeg_bin_dir() -> str:
    """解析 ffmpeg/ffprobe 所在目录。

    约定（见模块 docstring）：置于 <项目>/tools/ffmpeg/bin。
    这里 <项目> = ai-video-agent 根（即本文件上两级目录），
    故正确路径为 <here>/tools/ffmpeg/bin；同时兼容旧布局 <here>/../tools/ffmpeg/bin。
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(here, "tools", "ffmpeg", "bin"),                       # <项目>/tools/ffmpeg/bin
        os.path.abspath(os.path.join(here, "..", "tools", "ffmpeg", "bin")),  # 兼容旧布局
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "ffprobe.exe")) or \
           os.path.exists(os.path.join(c, "ffprobe")):
            return c
    return ""


def _exe(name: str) -> str:
    d = ffmpeg_bin_dir()
    return os.path.join(d, name) if d else name


# ---------------------------------------------------------------------------
# 1) 元数据探针
# ---------------------------------------------------------------------------
def probe(path: str) -> Dict[str, Any]:
    """ffprobe 取媒体元数据。返回 duration/fps/width/height/编码/has_audio。"""
    exe = _exe("ffprobe")
    cmd = [exe, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    fmt = info.get("format", {})
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})

    fps = 0.0
    if v.get("avg_frame_rate") and v["avg_frame_rate"] != "0/0":
        try:
            n, d = v["avg_frame_rate"].split("/")
            fps = float(n) / float(d) if float(d) else 0.0
        except Exception:
            fps = 0.0

    return {
        "duration": float(fmt.get("duration", 0.0)),
        "fps": round(fps, 3) if fps else 30,
        "width": int(v.get("width", 0) or 0),
        "height": int(v.get("height", 0) or 0),
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name"),
        "has_audio": bool(a),
    }


# ---------------------------------------------------------------------------
# 2) 镜头边界检测（OpenCV 帧直方图差）
# ---------------------------------------------------------------------------
def scene_cuts(path: str, sample_fps: float = 1.0, threshold: float = 0.4) -> List[float]:
    """返回镜头切变时间戳（不含 0.0 起点）。

    - sample_fps：抽帧频率（默认 1Hz 足够检测硬切变，省算力）；
    - threshold：Bhattacharyya 距离阈值，>阈值即判为切变。
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    cuts: List[float] = []
    prev = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (64, 64))
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            if prev is not None:
                diff = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > threshold:
                    cuts.append(idx / fps)
            prev = hist
        idx += 1
    cap.release()
    return sorted(set(round(c, 3) for c in cuts))


def extract_frames(path: str, times: List[float], out_dir: str) -> List[str]:
    """在指定时间戳抽帧存为 jpg，返回路径列表（供 M2 YOLOv8 检测）。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    paths: List[str] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            p = os.path.join(out_dir, f"kf_{int(t * 1000):07d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
    cap.release()
    return paths


# ---------------------------------------------------------------------------
# 3) 组装 Footage Index
# ---------------------------------------------------------------------------
def build_footage_index(path: str, media_id: str | None = None,
                        keyframe_interval: float = 2.0) -> FootageIndex:
    """M1 摄取主入口：产出 Footage Index（元数据 + 场景 + 关键帧时间戳）。
    visual/transcript 段由 M2 填充。"""
    meta = probe(path)
    dur = meta["duration"]
    cuts = scene_cuts(path)

    # 场景区间
    if cuts:
        scenes = [{"start": 0.0, "end": cuts[0]}]
        for i in range(len(cuts)):
            scenes.append({"start": cuts[i],
                           "end": cuts[i + 1] if i + 1 < len(cuts) else dur})
    else:
        scenes = [{"start": 0.0, "end": dur}]

    # 关键帧：每个场景中点 + 均匀采样（避免 0.0 重复）
    kf = [round((s["start"] + s["end"]) / 2, 3) for s in scenes]
    if dur > 0 and keyframe_interval > 0:
        kf += [round(t, 3) for t in range(0, int(dur), int(keyframe_interval)) if t != 0]
    kf = sorted(set(kf))

    return FootageIndex(
        mediaId=media_id or os.path.splitext(os.path.basename(path))[0],
        filepath=path,
        metadata=meta,
        scenes=scenes,
        keyframes=kf,
    )


# ---------------------------------------------------------------------------
# 4) 桥接 M3：索引 -> 原始 Timeline
# ---------------------------------------------------------------------------
def build_initial_timeline(index: FootageIndex) -> Timeline:
    """把摄取结果桥接为 M3 的「原始 Timeline」：整段视频 + 整段音频各一条 clip，
    Agent 在其上做 cut/trim/add_subtitle/add_bgm。"""
    dur = float(index.metadata.get("duration", 0.0))
    tl = Timeline(
        metadata=Metadata(
            schemaVersion="1.0.0",
            fps=int(index.metadata.get("fps", 30)),
            resolution={"w": int(index.metadata.get("width", 1920)),
                        "h": int(index.metadata.get("height", 1080))},
            duration=dur,
        ),
        tracks=[Track("v_main", "video"),
                Track("a_main", "audio"),
                Track("s_main", "subtitle")],
    )
    if dur > 0:
        tl.clips.append(Clip(
            id="src_video", track="v_main", sourceRef=index.filepath,
            in_=0.0, out=dur, sourceIn=0.0, sourceOut=dur))
        if index.metadata.get("has_audio"):
            tl.clips.append(Clip(
                id="src_audio", track="a_main", sourceRef=index.filepath,
                in_=0.0, out=dur, sourceIn=0.0, sourceOut=dur))
    return tl
