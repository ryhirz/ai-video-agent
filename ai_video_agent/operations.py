"""Edit Operation 应用层：把结构化操作增量应用到 Timeline（不可变）。

对应规格书 §2。每条操作都是纯函数式 apply：读取入参 Timeline 的拷贝，
返回新 Timeline，不修改入参。这样 op log 可回放、可单测、同操作同结果。

本文件另提供 apply_operations()（批量应用 + 鲁棒解析），供 UI / Agent 使用：
- cut 会把目标 clip 拆成 "<id>__a"/"<id>__b"（见 _apply_cut），**原 id 随即消失**；
  但 LLM / 规则 planner 常继续引用原 id（如 src_video），于是整批操作在 _find_clip
  处抛裸 KeyError('clip not found: src_video') 而全部失败（M5 实测踩到）。
- apply_operations 维护「派生 id 别名表」，按 at / in / out 时间提示把旧 id 重写
  到当前真实 id；非严格模式下跳过无法修复的单条操作，保证整批不中断且可解释。

A/V 对齐铁律（2026-09-23 新增，M5 实跑踩到「视频删了音频没删」）：
- **视频是主轨，音频跟随**。cut / delete_clip / delete_range 作用在视频上时，会
  同步作用于同 sourceRef 的配对音频 clip，否则渲染出来音频还按原时间轴播放，
  与画面对不上（`-shortest` 只会把音频按视频长度截断，不会让它跟着剪辑走）。
- **delete_range**（推荐给 planner 用）：按「素材绝对秒数区间」删除，不依赖
  cut 派生出的 id。LLM 手推 `__b__b__a` 这类 id 极易失效，而非严格模式只会把它
  静默跳过 —— 实测导致「有两段 bus 只删掉了后面那段」。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .timeline import Timeline, Clip, Subtitle, Track, Transition
from .compiler import AUDIO_EFFECTS, BGM_MARKER, VIDEO_EFFECTS


class OperationError(RuntimeError):
    """操作无法应用（带「第几条 / 类型 / 可用 id」可读上下文，替代裸 KeyError）。"""


def _find_clip(tl: Timeline, cid: str) -> Clip:
    for c in tl.clips:
        if c.id == cid:
            return c
    raise KeyError(f"clip not found: {cid}")


def _track_type(tl: Timeline, c: Clip) -> str:
    for tr in tl.tracks:
        if tr.id == c.track:
            return tr.type
    return "video"


def _is_bgm_clip(c: Clip) -> bool:
    """是否为 add_bgm 铺的背景音乐（叠加轨，不参与主轨的配对/拼接逻辑）。"""
    return any((e or {}).get("effectType") == BGM_MARKER for e in (c.effects or []))


def apply_operation(tl: Timeline, op: Dict[str, Any]) -> Timeline:
    """返回应用后的新 Timeline（不修改入参）。op 形如 {id,type,actor,payload}。"""
    t = tl.copy()
    typ = op["type"]
    p = op.get("payload", {})
    handlers = {
        "cut": _apply_cut,
        "trim": _apply_trim,
        "add_subtitle": _apply_subtitle,
        "add_bgm": _apply_bgm,
        "delete_clip": _apply_delete,
        "delete_range": _apply_delete_range,
        "move_clip": _apply_move,
        "transition": _apply_transition,
        "effect": _apply_effect,
    }
    if typ not in handlers:
        raise ValueError(f"UnsupportedOperationError: {typ}")
    handlers[typ](t, p)
    return t


# ===========================================================================
# 批量应用 + 派生 id 解析（鲁棒层）
# ===========================================================================
# 别名表结构：{名字: [当前存活的 clip id 列表]}
# - 初始：每个 clip 的 id 映射到自身；
# - 执行 cut 后：旧 id 展开为其两个派生 id（如 src_video -> [src_video__a, src_video__b]），
#   同时注册两个新 id 指向自身；因此后续引用旧 id 能解析到「当前还活着的那些段」。
AliasTable = Dict[str, List[str]]


def _live_ids(tl: Timeline) -> List[str]:
    return [c.id for c in tl.clips]


def _resolve_names(tl: Timeline, alias: AliasTable, name: str) -> List[str]:
    """名字 -> 当前存活 clip id 列表（去重、保持顺序）。"""
    live = {c.id for c in tl.clips}
    if name in alias:
        return [i for i in dict.fromkeys(alias[name]) if i in live]
    return [name] if name in live else []


def _pick_by_time(tl: Timeline, cands: List[str], *, at: Optional[float] = None,
                  span: Optional[Tuple[float, float]] = None, strict: bool = True
                  ) -> List[str]:
    """在候选 id 中按时间提示挑出「最匹配」的那一个/那些。"""
    want = set(cands)
    hits: List[str] = []
    for c in tl.clips:
        if c.id not in want:
            continue
        if at is not None:
            ok = (c.in_ < at < c.out) if strict else (c.in_ <= at <= c.out)
            if ok:
                hits.append(c.id)
        elif span is not None:
            lo, hi = span
            if lo == hi:
                if c.in_ <= lo <= c.out:
                    hits.append(c.id)
            elif min(hi, c.out) - max(lo, c.in_) > 0:
                hits.append(c.id)
    return hits


def _expand_alias(alias: AliasTable, old_id: str, children: List[str]) -> None:
    """把别名表里所有指向 old_id 的记录展开为 children，并注册 children 自身。"""
    for k in list(alias.keys()):
        new: List[str] = []
        for i in alias[k]:
            new.extend(children if i == old_id else [i])
        alias[k] = list(dict.fromkeys(new))
    alias[old_id] = list(children)
    for ch in children:
        alias[ch] = [ch]


def _resolve_name(tl: Timeline, alias: AliasTable, name: str,
                  hint: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str]]:
    """解析单个 clip 引用。返回 (解析出的 id 列表, 说明)。无法解析时抛 OperationError。"""
    cands = _resolve_names(tl, alias, name)
    if len(cands) == 1:
        notes = [f"`{name}` → `{cands[0]}`"] if cands[0] != name else []
        return cands, notes
    if not cands:
        raise OperationError(
            f"片段 `{name}` 不存在（可能已被前序操作删除或从未存在）；"
            f"当前可用 id：{_live_ids(tl)}")
    # 多个候选（该 id 被 cut 拆过）：按时间提示消歧
    hint = hint or {}
    at, span = hint.get("at"), hint.get("span")
    for strict in (True, False):
        hits = _pick_by_time(tl, cands, at=at, span=span, strict=strict) if (
            at is not None or span is not None) else []
        if len(hits) == 1:
            return hits, [f"`{name}` 已被切分，按时间提示自动解析为 `{hits[0]}`"]
    raise OperationError(
        f"片段 `{name}` 有多个候选 {cands}（被 cut 拆分过），无法按时间消歧；"
        f"请直接引用具体 id")


def _hint_of(typ: str, p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 payload 提取用于消歧的时间提示。"""
    try:
        if typ == "cut" and "at" in p:
            return {"at": float(p["at"])}
        if typ in ("trim", "delete_clip") and ("in" in p or "out" in p):
            lo = float(p["in"]) if "in" in p else float("-inf")
            hi = float(p["out"]) if "out" in p else float("inf")
            return {"span": (lo, hi)}
        if typ in ("delete_clip", "delete_range") and "at" in p:
            return {"at": float(p["at"])}
    except (TypeError, ValueError):
        return None
    return None


def _resolve_op(tl: Timeline, op: Dict[str, Any], alias: AliasTable
                ) -> Tuple[Dict[str, Any], List[str]]:
    """把一条操作里的 clip 引用重写为当前真实 id。返回 (新操作, 说明列表)。"""
    typ = op.get("type")
    p = copy.deepcopy(op.get("payload") or {})
    notes: List[str] = []

    if typ == "transition":                       # 唯一引用多个 clip 的操作
        ids: List[str] = []
        for name in list(p.get("clips") or []):
            got, nt = _resolve_name(tl, alias, str(name))
            ids.extend(got)
            notes.extend(nt)
        p["clips"] = list(dict.fromkeys(ids))
        if not p["clips"]:
            raise OperationError("transition 的 clips 为空，无法锚定转场")
        return {**op, "payload": p}, notes

    if "targetClipId" not in p:
        return {**op, "payload": p}, notes      # add_subtitle / add_bgm 等不引用已有 clip

    got, nt = _resolve_name(tl, alias, str(p["targetClipId"]), _hint_of(typ, p))
    notes.extend(nt)
    if len(got) != 1:
        raise OperationError(
            f"`{typ}` 需要唯一的 targetClipId，但解析出 {got}；请改用具体 id")
    p["targetClipId"] = got[0]
    return {**op, "payload": p}, notes


def _update_alias(alias: AliasTable, tl: Timeline, op: Dict[str, Any]) -> None:
    """应用一条操作后同步别名表。

    任何操作产生的新派生 id（cut / delete_range / 区间摘除 都会造 `X__a`/`X__b`）
    都要登记；同时把别名表里所有指向其「父 id」的记录展开到这些新 id，
    父 id 随即失效 —— 与 _split_at_source 的命名契约保持一致。
    """
    typ, p = op.get("type"), op.get("payload") or {}
    live = set(_live_ids(tl))
    # 按派生层级由浅到深处理，保证 X__a 先于 X__a__b 登记
    fresh = sorted((i for i in live if i not in alias and "__" in i),
                   key=lambda s: (s.count("__"), s))
    for fid in fresh:
        parent = fid.rpartition("__")[0]
        if parent in alias:
            # 只取「直接子代」（rpartition 后父名相同），避免把孙子也一并登记
            children = sorted(x for x in live if x.rpartition("__")[0] == parent)
            _expand_alias(alias, parent, children or [fid])
        else:
            alias[fid] = [fid]
    for i in live:
        alias.setdefault(i, [i])
    if typ == "add_bgm":
        nid = p.get("id")
        if nid:
            alias[str(nid)] = [str(nid)]


def _sub_key(text: Any, start: Any, end: Any, track: Any, pos: Any) -> Optional[Tuple]:
    """字幕去重键（同一文本 + 同一时间窗 + 同一位置 = 重复）。"""
    try:
        return (str(text), round(float(start), 3), round(float(end), 3),
                str(track or "s_main"), str(pos or "bottom-center"))
    except (TypeError, ValueError):
        return None


def apply_operations(tl: Timeline, ops: List[Dict[str, Any]], *, strict: bool = False
                     ) -> Tuple[Timeline, List[str], List[str]]:
    """顺序应用操作列表，返回 (新 Timeline, 应用日志, 警告列表)。

    与逐条 apply_operation 的区别（M5 实测暴露的鲁棒性需求）：
    - 自动把「被 cut 拆掉的旧 id」解析到当前真实 id（见模块 docstring）；
    - strict=False（默认）时单条失败只跳过并记入警告，其余操作照常生效；
    - 错误信息带可用 id，用户可直接照着改指令，而不是看到裸 KeyError；
    - 完全重复的 add_subtitle 自动丢弃（LLM 偶发把同一条字幕输出两遍，
      叠加渲染会变成「加粗/重影」）。

    ops 为空或全部被跳过时，返回的 Timeline 与入参等价（不修改入参）。
    """
    cur = tl.copy()
    logs: List[str] = []
    warns: List[str] = []
    alias: AliasTable = {c.id: [c.id] for c in cur.clips}
    seen_subs = {
        k for k in (
            _sub_key(s.text, s.start, s.end, s.track, (s.style or {}).get("position"))
            for s in cur.subtitles
        ) if k
    }

    for i, op in enumerate(ops or [], 1):
        typ = op.get("type") if isinstance(op, dict) else None
        if not isinstance(op, dict) or not typ:
            warns.append(f"{i}. 非法操作（缺 type）：{op!r} 已跳过")
            continue
        if typ == "add_subtitle":
            p = op.get("payload") or {}
            key = _sub_key(p.get("text"), p.get("start"), p.get("end"),
                           p.get("track"), (p.get("style") or {}).get("position"))
            if key and key in seen_subs:
                warns.append(f"{i}. `add_subtitle` 已跳过：与已有字幕重复"
                             f"（“{key[0]}” {key[1]}–{key[2]}s，叠加会变成重影）")
                continue
        try:
            fixed, notes = _resolve_op(cur, op, alias)
        except OperationError as e:
            if strict:
                raise
            warns.append(f"{i}. `{typ}` 已跳过：{e}")
            continue
        try:
            cur = apply_operation(cur, fixed)
        except Exception as e:  # noqa: BLE001 —— 单条失败不拖垮整批
            if strict:
                raise OperationError(f"第 {i} 条 `{typ}` 应用失败：{e}") from e
            warns.append(f"{i}. `{typ}` 已跳过：{e}")
            continue
        if typ == "add_subtitle":
            fp = fixed.get("payload") or {}
            k2 = _sub_key(fp.get("text"), fp.get("start"), fp.get("end"),
                          fp.get("track"), (fp.get("style") or {}).get("position"))
            if k2:
                seen_subs.add(k2)
        _update_alias(alias, cur, fixed)
        logs.append(f"{i}. `{typ}` → 已应用" + (f"（{'；'.join(notes)}）" if notes else ""))

    return cur, logs, warns


# ---------------- 时间轴/素材映射与拆分（A/V 对齐的基础设施） ----------------

def _source_at(c: Clip, at: float) -> float:
    """时间轴时刻 at 对应的「素材绝对秒数」。"""
    return c.sourceIn + (at - c.in_)


def _piece(src: Clip, new_id: str, s_from: float, s_to: float) -> Clip:
    """以 clip src 为模板，截取其素材区间 [s_from, s_to] 造一段新 clip。"""
    return Clip(
        id=new_id, track=src.track, sourceRef=src.sourceRef,
        in_=src.in_ + (s_from - src.sourceIn),
        out=src.in_ + (s_to - src.sourceIn),
        sourceIn=s_from, sourceOut=s_to,
        transform=copy.deepcopy(src.transform),
        effects=copy.deepcopy(src.effects),
    )


def _split_at_source(t: Timeline, c: Clip, src_at: float) -> Tuple[Clip, Clip]:
    """在「素材绝对秒数」src_at 处把 clip c 拆成 __a / __b 两段（原地替换）。

    与 _apply_cut 共用同一命名约定（公开契约）：c -> c__a（前）/ c__b（后），原 id 消失。
    """
    if not (c.sourceIn + 1e-9 < src_at < c.sourceOut - 1e-9):
        raise ValueError(
            f"拆分点 {src_at:.3f}s 必须严格落在 clip 素材区间 "
            f"({c.sourceIn:.3f},{c.sourceOut:.3f})")
    at = c.in_ + (src_at - c.sourceIn)
    a = _piece(c, f"{c.id}__a", c.sourceIn, src_at)
    b = _piece(c, f"{c.id}__b", src_at, c.sourceOut)
    idx = t.clips.index(c)
    t.clips[idx:idx + 1] = [a, b]
    # Q1 转场锚定：被切分且引用该 clip 的转场，重锚到新相邻边界（保留）
    for tr in t.transitions:
        if c.id in tr.attachedClips:
            tr.attachedClips = [a.id, b.id]
    return a, b


def _remove_source_range(t: Timeline, source_ref: str, start: float, end: float,
                         *, exclude_track: Optional[str] = None) -> List[str]:
    """把同源 clip 上的素材区间 [start,end] 摘掉（需要时先切两刀）。

    返回被摘除的原始 clip id 列表。`exclude_track` 用于「只同步其它轨」的场景
    （如 delete_clip 已经删掉视频本身，只需再清理配对音频）。
    """
    removed: List[str] = []
    for c in list(t.clips):
        if c.sourceRef != source_ref or (exclude_track and c.track == exclude_track):
            continue
        if _is_bgm_clip(c):
            continue                     # BGM 是叠加音轨，不参与「按素材区间摘除」
        lo, hi = c.sourceIn, c.sourceOut
        if hi <= start + 1e-9 or lo >= end - 1e-9:
            continue                                   # 与该区间无交集
        keep: List[Clip] = []
        if lo < start - 1e-9:
            keep.append(_piece(c, f"{c.id}__a", lo, start))
        if hi > end + 1e-9:
            nid = f"{c.id}__b" if keep else f"{c.id}__a"
            keep.append(_piece(c, nid, end, hi))
        idx = t.clips.index(c)
        t.clips[idx:idx + 1] = keep
        for tr in t.transitions:
            if c.id in tr.attachedClips:
                tr.attachedClips = [k.id for k in keep]
        t.transitions = [tr for tr in t.transitions if tr.attachedClips]
        removed.append(c.id)
    return removed


def _primary_source_ref(t: Timeline, start: float, end: float) -> Optional[str]:
    """找一个能代表区间 [start,end] 的素材源：优先覆盖该区间的视频 clip，其次任意视频 clip。"""
    mid = (start + end) / 2
    fallback: Optional[str] = None
    for c in t.clips:
        if _track_type(t, c) != "video":
            continue
        if fallback is None:
            fallback = c.sourceRef
        if c.sourceIn <= mid <= c.sourceOut:
            return c.sourceRef
    return fallback


# ---------------- P0 操作 ----------------

def _apply_cut(t: Timeline, p: Dict[str, Any]) -> None:
    c = _find_clip(t, p["targetClipId"])
    at = float(p["at"])
    if not (c.in_ < at < c.out):
        raise ValueError(f"cut at={at} 必须严格落在 ({c.in_},{c.out}) 开区间")
    src_at = _source_at(c, at)
    _split_at_source(t, c, src_at)
    # A/V 对齐：同源、其它轨的配对 clip 在同一素材时刻一起切，
    # 这样视频后来的 trim / delete 才能精确对应到音频的同一段。
    for o in list(t.clips):
        if o.track == c.track or o.sourceRef != c.sourceRef:
            continue
        if _is_bgm_clip(o):
            continue
        if o.sourceIn + 1e-9 < src_at < o.sourceOut - 1e-9:
            _split_at_source(t, o, src_at)


def _apply_trim(t: Timeline, p: Dict[str, Any]) -> None:
    c = _find_clip(t, p["targetClipId"])
    new_in = float(p.get("in", c.in_))
    new_out = float(p.get("out", c.out))
    if not (new_in < new_out):
        raise ValueError("trim: 必须 in < out")
    if new_in < c.in_ or new_out > c.out:
        raise ValueError("trim: 只能裁头/裁尾，不能扩展 clip 边界")
    # 同步同源、同素材区间、不同轨的配对 clip（如视频的配对音频），保持 A/V 对齐：
    # 否则只裁视频不裁音频，混流后音频会超前、尾部出现定格帧。
    new_sin = c.sourceIn + (new_in - c.in_)
    new_sout = c.sourceOut - (c.out - new_out)
    for o in t.clips:
        if o is c or o.track == c.track or o.sourceRef != c.sourceRef:
            continue
        if _is_bgm_clip(o):
            continue
        # 按「素材区间」配对（比按时间轴 in_/out 配对更稳：cut 之后两者可能错位）
        if abs(o.sourceIn - c.sourceIn) < 1e-6 and abs(o.sourceOut - c.sourceOut) < 1e-6:
            o.sourceIn, o.sourceOut = new_sin, new_sout
            o.in_, o.out = new_in, new_out
    c.sourceIn, c.sourceOut = new_sin, new_sout
    c.in_, c.out = new_in, new_out


def _apply_subtitle(t: Timeline, p: Dict[str, Any]) -> None:
    sub = Subtitle(
        id=p.get("id", f"sub_{len(t.subtitles) + 1}"),
        track=p.get("track", "s_main"),
        text=p["text"],
        start=float(p["start"]),
        end=float(p["end"]),
        style=p.get("style", {}),
    )
    if not (sub.start < sub.end):
        raise ValueError("subtitle: 必须 start < end")
    if sub.start < 0 or sub.end > t.metadata.duration:
        raise ValueError("subtitle: 超出时间轴范围")
    t.subtitles.append(sub)                # v1 允许重叠（warning 级）


def _apply_bgm(t: Timeline, p: Dict[str, Any]) -> None:
    """铺背景音乐。

    BGM 放在**独立的 a_bgm 轨**上：它是"叠加"而非"主音频"，若与主音频同轨，
    一边是按素材区间配对删除、一边是按序拼接，语义会互相污染。编译器按轨把它
    单独抽出来做 amix 混音。
    """
    if not (float(p["in"]) < float(p["out"])):
        raise ValueError("bgm: 必须 in < out")
    track = p.get("track", "a_bgm")
    if not any(tr.id == track for tr in t.tracks):
        t.tracks.append(Track(track, "audio"))
    clip = Clip(
        id=p.get("id", f"bgm_{len(t.clips) + 1}"),
        track=track,
        sourceRef=p["sourceRef"],
        in_=float(p["in"]),
        out=float(p["out"]),
        sourceIn=0.0,
        sourceOut=float(p["out"]) - float(p["in"]),
        effects=[{
            "effectType": BGM_MARKER,
            "params": {
                "volume": float(p.get("volume", 1.0)),
                "fadeIn": float(p.get("fadeIn", 0.0)),
                "fadeOut": float(p.get("fadeOut", 0.0)),
            },
        }],
    )
    t.clips.append(clip)


# ---------------- 基础操作 ----------------

def _apply_delete(t: Timeline, p: Dict[str, Any]) -> None:
    c = _find_clip(t, p["targetClipId"])
    # Q1：若被转场引用，丢弃该转场（warning 级，不静默保留）
    t.transitions = [tr for tr in t.transitions if c.id not in tr.attachedClips]
    t.clips.remove(c)
    # A/V 对齐：视频是主轨。删掉一段视频时，同源同素材区间的音频段必须一并删掉，
    # 否则渲染出的音频仍在原时间轴播放（-shortest 只按长度截断，不会跟着剪辑走）。
    if _track_type(t, c) == "video":
        _remove_source_range(t, c.sourceRef, c.sourceIn, c.sourceOut,
                             exclude_track=c.track)


def _apply_delete_range(t: Timeline, p: Dict[str, Any]) -> None:
    """按「素材绝对秒数区间」删除 [start,end]（视频与其配对音频一起删）。

    这是给 planner 用的稳健删除：不依赖 cut 派生出的 id，多条区间彼此独立、
    顺序无关，天然避免「LLM 手推 __b__b__a 失效 -> 删除被静默跳过 -> 只删了一半」。
    """
    start, end = float(p["start"]), float(p["end"])
    if not (start < end):
        raise ValueError("delete_range: 必须 start < end")
    ref = p.get("sourceRef") or _primary_source_ref(t, start, end)
    if not ref:
        raise ValueError("delete_range: 时间轴上没有可用的视频素材，无法删除")
    removed = _remove_source_range(t, str(ref), start, end)
    if not removed:
        raise ValueError(
            f"delete_range: 素材 {start:.2f}–{end:.2f}s 区间已被删除或不存在（无 clip 覆盖）")


def _apply_move(t: Timeline, p: Dict[str, Any]) -> None:
    """移动片段。

    `toIn` 采用**重排（reflow）语义**：把目标片段插到新位置后，整条轨重排为
    首尾相接。为什么不直接改 in_ 了事？—— 编译器的画面是「按入点排序后顺序拼接」，
    只改被移动片段自己的 in_ 会与邻段撞车（两者 in_ 相同则排序退化为原顺序），
    成片毫无变化，等于静默失效。重排能保证顺序真的体现在成片里。
    """
    c = _find_clip(t, p["targetClipId"])
    if "toTrack" in p:
        c.track = p["toTrack"]
    if "toIn" not in p:
        return
    ni = float(p["toIn"])
    dur = c.out - c.in_
    if ni < 0 or ni + dur > t.metadata.duration + 1e-6:
        raise ValueError("move: 越界")
    # 同轨其他片段，按原入点排序；把 c 摘出来，再按 toIn 决定插到哪个位置
    others = sorted([x for x in t.clips if x.track == c.track and x is not c],
                    key=lambda x: x.in_)
    pos = sum(1 for x in others if x.in_ < ni)
    order = others[:pos] + [c] + others[pos:]
    # 重排为连续时间轴
    at = 0.0
    for x in order:
        d = x.out - x.in_
        x.in_, x.out = at, at + d
        at += d


def _apply_transition(t: Timeline, p: Dict[str, Any]) -> None:
    tr = Transition(
        id=p.get("id", f"trans_{len(t.transitions) + 1}"),
        type=p["transitionType"],
        duration=float(p["duration"]),
        attachedClips=list(p["clips"]),
    )
    t.transitions.append(tr)               # 叠加渲染指令，不改 clip in/out


def _apply_effect(t: Timeline, p: Dict[str, Any]) -> None:
    """给片段挂效果。

    这里**校验效果名**：编译器只认一批确定的效果，认不出的会渲染不出任何变化。
    与其让操作"成功"却对成片毫无影响（静默失效），不如在这里直接拒绝 ——
    apply_operations 非严格模式下会把它变成一条可见的警告。
    """
    c = _find_clip(t, p["targetClipId"])
    et = str(p["effectType"])
    known = AUDIO_EFFECTS if _track_type(t, c) == "audio" else VIDEO_EFFECTS
    if et not in known:
        raise ValueError(
            f"effect: 不支持的效果 `{et}`（{'音轨' if _track_type(t, c) == 'audio' else '画面'}"
            f"可用：{list(known)}）")
    c.effects.append({"effectType": et, "params": p.get("params", {})})
