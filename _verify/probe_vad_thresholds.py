"""实测（第二轮）：VAD 参数的正确传法。

第一轮用 `vad_post_args={"max_end_silence_time": th}` —— 四个阈值结果完全相同，
说明没生效。查源码发现：

    self.vad_opts = VADXOptions(**kwargs)      # fsmn_vad_streaming/model.py:397

VAD 参数是**直接作为关键字**传给 AutoModel 的（`vad_post_args` 虽然出现在
`FsmnVADStreaming.__init__` 签名里，但从未被使用）。

本轮用 `AutoModel(model="fsmn-vad", max_end_silence_time=th)` 重测，
并用 `max_single_segment_time` 做对照（它若生效会强行切长段，用来验证链路是否通）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
CLIP2 = ROOT / "测试素材2_多目标.mp4"
CLIP3 = ROOT / "测试素材3_竖屏音乐.mp4"

CONFIGS = [
    ("默认（不传参）", {}),
    ("max_end_silence_time=300", {"max_end_silence_time": 300}),
    ("max_end_silence_time=100", {"max_end_silence_time": 100}),
    ("max_single_segment_time=4000（对照，验证链路）", {"max_single_segment_time": 4000}),
]


def to_wav(clip: Path) -> str:
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run([str(FFMPEG), "-y", "-i", str(clip), "-vn", "-ac", "1",
                    "-ar", "16000", wav], capture_output=True, check=True)
    return wav


def main() -> None:
    from funasr import AutoModel

    out = {}
    for label, cfg in CONFIGS:
        vad = AutoModel(model="fsmn-vad", device="cpu", disable_update=True,
                        disable_pbar=True, **cfg)
        print(f"\n{'=' * 74}\n== {label}\n{'=' * 74}")
        for name, clip in (("素材2", CLIP2), ("素材3", CLIP3)):
            res = vad.generate(input=to_wav(clip))
            val = res[0].get("value") if res else []
            print(f"  {name}: {len(val)} 段  {val}")
            out[f"{label}|{name}"] = val
        del vad

    (ROOT / "_verify" / "vad_thresholds.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n明细已写入 _verify/vad_thresholds.json")


if __name__ == "__main__":
    main()
