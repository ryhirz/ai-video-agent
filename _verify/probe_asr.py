"""诊断：FunASR 的原始返回结构（找时间戳与置信度字段）。

要回答两个问题：
  1. 有没有办法拿到**逐句时间戳**（现在永远退化成"整段一条 0–时长"）？
  2. 纯音乐上幻觉出的 `'the'` 有没有**分数/置信度**可以据此过滤？

不改任何业务代码，只把真实返回原样 dump 出来看。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.understand import _get_asr  # noqa: E402

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
CLIPS = {"素材2_有旁白": ROOT / "测试素材2_多目标.mp4",
         "素材3_纯音乐": ROOT / "测试素材3_竖屏音乐.mp4"}


def to_wav(clip: Path) -> str:
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run([str(FFMPEG), "-y", "-i", str(clip), "-vn",
                    "-ac", "1", "-ar", "16000", wav], capture_output=True, check=True)
    return wav


def dump(obj, depth: int = 0, maxdepth: int = 3):
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and depth < maxdepth:
                print(f"{pad}{k}: {type(v).__name__}(len={len(v)})")
                dump(v, depth + 1, maxdepth)
            else:
                s = repr(v)
                print(f"{pad}{k} = {s[:200]}")
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0] 样例：")
        dump(obj[0], depth + 1, maxdepth)
    else:
        print(f"{pad}{repr(obj)[:200]}")


def main() -> None:
    am = _get_asr("paraformer", "v2.0.4", "cpu")
    all_out = {}
    for name, clip in CLIPS.items():
        wav = to_wav(clip)
        print("\n" + "=" * 74)
        print(f"== {name}  ({clip.name})")
        print("=" * 74)

        for kw in ({}, {"sentence_timestamp": True}):
            tag = "sentence_timestamp=True" if kw else "（默认，当前代码用的）"
            print(f"\n--- generate(kw={kw}) {tag} ---")
            raw = am.generate(input=wav, batch_size_s=300, return_spk_res=False, **kw)
            dump(raw)
            all_out[f"{name}|{tag}"] = {
                "kwargs": {k: str(v) for k, v in kw.items()},
                "keys": [list(r.keys()) for r in raw],
                "text": [r.get("text") for r in raw],
                "sentence_info_len": [len(r.get("sentence_info") or []) for r in raw],
                "raw": raw,
            }

    (ROOT / "_verify" / "asr_probe.json").write_text(
        json.dumps(all_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n明细已写入 _verify/asr_probe.json")


if __name__ == "__main__":
    main()
