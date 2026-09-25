"""实测：批量转写（`generate(input=[wav1, wav2, ...])`）能否替代"逐段逐次转写"。

动机：VAD 分段把 1 次转写变成 N 次（素材2 → 11 次），实测耗时 1.38s → 5.03s（3.64×）。
每次调用都要重跑一遍特征提取等前后处理，批量应该能省掉这部分。

要验证两件事：
  ① **更快吗**（省下多少）
  ② **返回是否与输入一一对应**（条数 + 内容顺序）—— 如果对不上就不能用。

源码依据：`prepare_data_iterator` 对 list 输入走 `data_list = data_in`，逐条给 key，
所以理论上逐条返回。但"理论上"要实测。
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

from ai_video_agent.ingest import build_footage_index
from ai_video_agent.understand import _get_asr, _get_vad, _slice_wav, _vad_spans
from ai_video_agent.ingest import ffmpeg_bin_dir

CLIP = ROOT / "测试素材2_多目标.mp4"
FF = str(Path(ffmpeg_bin_dir()) / "ffmpeg")


def main() -> None:
    idx = build_footage_index(str(CLIP))
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run([FF, "-y", "-i", str(CLIP), "-vn", "-ac", "1", "-ar", "16000", wav],
                   capture_output=True, check=True)

    am = _get_asr("paraformer", "v2.0.4", "cpu")
    _get_vad("cpu")                                   # 预热
    am.generate(input=wav)                            # 预热

    spans = _vad_spans(wav, "cpu", 24.0)
    print(f"VAD 段数：{len(spans)}")
    pieces = [_slice_wav(FF, wav, s, e) for (s, e) in spans]

    # ---- 逐条 ----
    t0 = time.time()
    one_by_one = []
    for p in pieces:
        r = am.generate(input=p, batch_size_s=300, return_spk_res=False)
        one_by_one.append((r[0].get("text") if r else "") or "")
    t_one = time.time() - t0

    # ---- 批量 ----
    t0 = time.time()
    batch_raw = am.generate(input=pieces, batch_size_s=300, return_spk_res=False)
    t_batch = time.time() - t0
    batch = [(r.get("text") or "") for r in batch_raw]

    print(f"\n逐条：{t_one:.2f}s（{len(one_by_one)} 条）")
    print(f"批量：{t_batch:.2f}s（{len(batch)} 条）")
    print(f"\n条数一致：{len(one_by_one) == len(batch)}")
    same = one_by_one == batch
    print(f"内容与顺序完全一致：{same}")
    if not same:
        print("\n逐条 vs 批量 对照：")
        for i, (a, b) in enumerate(zip(one_by_one, batch)):
            mark = "  " if a == b else "≠ "
            print(f"  {mark}[{i}] {a!r}  |  {b!r}")
        if len(batch) != len(one_by_one):
            print(f"  （条数不同：逐条 {len(one_by_one)} / 批量 {len(batch)}）")

    # 批量返回体的 keys（看是否有额外字段）
    if batch_raw:
        print(f"\n批量返回体 keys：{list(batch_raw[0].keys())}")

    (ROOT / "_verify" / "bench_batch_asr.json").write_text(
        json.dumps({"spans": len(spans), "t_one_by_one": round(t_one, 2),
                    "t_batch": round(t_batch, 2), "same": same,
                    "one_by_one": one_by_one, "batch": batch},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for p in pieces:
        try:
            Path(p).unlink()
        except OSError:
            pass
    print("\n明细已写入 _verify/bench_batch_asr.json")


if __name__ == "__main__":
    main()
