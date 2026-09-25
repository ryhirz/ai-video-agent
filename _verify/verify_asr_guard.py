"""实测 ASR 幻觉护栏：素材3（纯音乐）不应再产出 'the' 这种整片段；
素材2（真人中文旁白）不能被误杀。直接复用被测代码，不另写一套实现。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from ai_video_agent.ingest import build_footage_index
from ai_video_agent.understand import transcribe

CASES = [
    ("素材2_多目标（真人中文旁白）", os.path.join(HERE, "测试素材2_多目标.mp4"), 24.0),
    ("素材3_竖屏音乐（纯音乐，无语音）", os.path.join(HERE, "测试素材3_竖屏音乐.mp4"), 20.0),
]

for name, path, dur in CASES:
    idx = build_footage_index(path)
    segs = transcribe(idx)
    print(f"== {name}  (dur={idx.metadata.get('duration')}, "
          f"has_audio={idx.metadata.get('has_audio')})")
    if not segs:
        print("   转写段：空")
    for s in segs:
        print(f"   [{s.start:.2f}-{s.end:.2f}] {s.text!r}")
    print()
