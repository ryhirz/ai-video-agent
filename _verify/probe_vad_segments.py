"""探测：把 VAD 当**独立模型**用，能否拿到原始语音段边界（毫秒）。

上一轮把 VAD 挂在 AutoModel(vad_model=...) 里，它只是内部切段后又把结果拼成一条，
所以拿不到段边界。这次单独实例化 `AutoModel(model="fsmn-vad")`，看原始返回。

只 dump，不改业务代码。目标：确认 `value` 里是不是 [[beg_ms, end_ms], ...]。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
CLIPS = {"素材2_有旁白": ROOT / "测试素材2_多目标.mp4",
         "素材3_纯音乐": ROOT / "测试素材3_竖屏音乐.mp4"}


def to_wav(clip: Path) -> str:
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run([str(FFMPEG), "-y", "-i", str(clip), "-vn", "-ac", "1",
                    "-ar", "16000", wav], capture_output=True, check=True)
    return wav


def main() -> None:
    from funasr import AutoModel

    t0 = time.time()
    print("单独加载 AutoModel(model='fsmn-vad') …")
    vad = AutoModel(model="fsmn-vad", device="cpu",
                    disable_update=True, disable_pbar=True)
    print(f"加载完成，用时 {time.time() - t0:.1f}s")

    out = {}
    for name, clip in CLIPS.items():
        wav = to_wav(clip)
        print("\n" + "=" * 74)
        print(f"== {name}")
        raw = vad.generate(input=wav, cache={}, **{})
        print(f"返回 {len(raw)} 条；keys={list(raw[0].keys()) if raw else '—'}")
        for r in raw:
            v = r.get("value", r.get("text"))
            print(f"  类型={type(v).__name__}")
            if isinstance(v, list):
                print(f"  段数={len(v)}")
                for seg in v[:20]:
                    print(f"    {seg}")
            else:
                print(f"  value={v!r}")
        out[name] = raw

    (ROOT / "_verify" / "vad_segments.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n明细已写入 _verify/vad_segments.json")


if __name__ == "__main__":
    main()
