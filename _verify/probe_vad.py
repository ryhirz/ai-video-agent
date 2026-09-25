"""实测：给 FunASR 挂上 VAD（fsmn-vad，约几 MB）能否一次解决两件事。

假设（未经验证）：
  A. **无语音素材不再幻觉**：纯音乐音轨下 VAD 应该切不出语音段 → 返回空，而不是像现在吐出一个 'the'。
  B. **拿到逐句时间戳**：VAD 按静音切段，每段一个结果，段的边界就是台词的时间戳 ——
     这样就不必为了 `sentence_timestamp` 去下 ct-punc（几百 MB）。

只 dump 真实返回，不改业务代码。顺带确认模型下载落在哪块盘（用户 C 盘紧张）。
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
    print("加载 AutoModel(model=paraformer, vad_model=fsmn-vad) …（首次会下载 VAD 权重）")
    am = AutoModel(model="paraformer", model_revision="v2.0.4",
                   vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000},
                   device="cpu", disable_update=True, disable_pbar=True)
    print(f"加载完成，用时 {time.time() - t0:.1f}s")
    for name in ("vad_model", "model"):
        v = getattr(am, name, None)
        p = getattr(v, "model_path", None) if v is not None else None
        print(f"  {name} -> {p}")

    out = {}
    for name, clip in CLIPS.items():
        wav = to_wav(clip)
        print("\n" + "=" * 74)
        print(f"== {name}")
        print("=" * 74)
        raw = am.generate(input=wav, batch_size_s=300)
        print(f"返回 {len(raw)} 条结果")
        for i, r in enumerate(raw):
            keys = list(r.keys())
            print(f"  [{i}] keys={keys}")
            print(f"      text={r.get('text')!r}")
            if r.get("timestamp"):
                print(f"      timestamp={r['timestamp']}")
        out[name] = raw

    (ROOT / "_verify" / "vad_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n明细已写入 _verify/vad_probe.json")


if __name__ == "__main__":
    main()
