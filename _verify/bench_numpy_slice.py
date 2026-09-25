"""实测：用 numpy 内存切片替代 ffmpeg 切音频（省掉 11 次进程调用）。

背景：bench_transcribe 显示 VAD 分段耗时 5.03s，其中**转写只占 1.72s**，
剩下 **3.31s 全花在 11 次 ffmpeg 切音频**上 —— 优化重点在切段，不在转写。

`prepare_data_iterator` 的注释写着元素可以是 `[audio sample point, fbank, text]`，
所以理论上可以直接把 numpy 数组喂进去。

**要验证的是"结果与 ffmpeg 版本完全一致"** —— 数组的 dtype/取值范围若不对，
可能不报错但转出垃圾。所以必须逐条比对文本，而不是只看跑不跑得通。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index, ffmpeg_bin_dir
from ai_video_agent.understand import _get_asr, _get_vad, _slice_wav, _vad_spans

CLIP = ROOT / "测试素材2_多目标.mp4"
FF = str(Path(ffmpeg_bin_dir()) / "ffmpeg")
SR = 16000


def main() -> None:
    build_footage_index(str(CLIP))
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run([FF, "-y", "-i", str(CLIP), "-vn", "-ac", "1", "-ar", str(SR), wav],
                   capture_output=True, check=True)

    am = _get_asr("paraformer", "v2.0.4", "cpu")
    _get_vad("cpu")
    am.generate(input=wav)                                  # 预热

    spans = _vad_spans(wav, "cpu", 24.0)
    print(f"VAD 段数：{len(spans)}")

    # ---- A：ffmpeg 切段 + 批量转写 ----
    t0 = time.time()
    pieces = [_slice_wav(FF, wav, s, e) for (s, e) in spans]
    t_slice = time.time() - t0
    raw_a = am.generate(input=pieces, batch_size_s=300, return_spk_res=False)
    text_a = [(r.get("text") or "") for r in raw_a]
    print(f"A（ffmpeg 切段）：切段 {t_slice:.2f}s，转写 {(time.time() - t0 - t_slice):.2f}s")

    # ---- B：numpy 内存切片 + 批量转写 ----
    t0 = time.time()
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:                                      # 保险：强制单声道
        audio = audio.mean(axis=1)
    assert sr == SR, f"采样率不符：{sr}"
    chunks = [audio[int(s * SR):int(e * SR)] for (s, e) in spans]
    t_cut = time.time() - t0
    raw_b = am.generate(input=chunks, batch_size_s=300, return_spk_res=False)
    text_b = [(r.get("text") or "") for r in raw_b]
    t_all_b = time.time() - t0
    print(f"B（内存切片）：切片 {t_cut:.3f}s，合计 {t_all_b:.2f}s")

    print(f"\n条数一致：{len(text_a) == len(text_b)}（A={len(text_a)} / B={len(text_b)}）")
    same = text_a == text_b
    print(f"文本完全一致：{same}")
    if not same:
        for i, (a, b) in enumerate(zip(text_a, text_b)):
            if a != b:
                print(f"  ≠ [{i}] A={a!r}  B={b!r}")

    print(f"\n**总耗时对比**：A {t_slice + 1.72:.2f}s（切段+转写） vs B {t_all_b:.2f}s")

    (ROOT / "_verify" / "bench_numpy_slice.json").write_text(
        json.dumps({"spans": len(spans), "t_ffmpeg_slice": round(t_slice, 2),
                    "t_numpy_total": round(t_all_b, 2), "same": same,
                    "text_ffmpeg": text_a, "text_numpy": text_b},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for p in pieces:
        try:
            Path(p).unlink()
        except OSError:
            pass
    print("\n明细已写入 _verify/bench_numpy_slice.json")


if __name__ == "__main__":
    main()
