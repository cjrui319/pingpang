"""离线精度调参工具（**开发用，提交前可整文件删除**）。

要解决的问题：精度分 75 分，而 0° 视角只有 21 个有效 GT，每错一个点就动 ~4% F1；
没有可测量的回路就只能盲调，极易把噪声当信号。本工具给出**官方口径**的本地评分：

    F1 命中判据（与细则一致）：
      * 预测与 GT「帧号差 ≤ 1 帧」且「欧氏距离 ≤ 50px」才算命中（严格一对一匹配）
      * valid=false 的 GT 不参与评分；**命中无效点的预测直接丢弃**（既不算 TP 也不算 FP）
        —— 细则明确「误检或遗漏这类点不影响评分」
      * P = TP/(TP+FP)，R = TP/(TP+FN)，F1 = 2PR/(P+R)
      * 精度分 = 25*F1_0° + 25*F1_45° + 25*F1_90°

三个子命令：
  1) dump   —— 在能跑模型的环境里把公开集跑一遍，逐帧检测结果落盘（只跑一次）
       python3 participant/_tune_eval.py dump --input public_data --out /tmp/pp_traj
     评测机路径（TRT + NVDEC）就是上面的默认值。**没有 TRT/GPU 的开发机也能测**：
       python3 participant/_tune_eval.py dump --input public_data --out /tmp/pp_traj \
           --backend ort --decoder cpu --ort-dir <解包后的 onnxruntime 目录>
       （--decoder cv2 是本机既无 PyAV 也无 ffmpeg 时的兜底；ORT 与 TRT fp16 的数值
         有极小差异，故 ORT 上只用于**横向比较参数**，绝对值以评测机为准）
  2) score  —— 纯离线：重放 EmitRule + 官方口径评分 + 参数扫描（不需要模型/GPU）
       python3 participant/_tune_eval.py score --traj /tmp/pp_traj --input public_data \
           --one-way --per-video
  3) selftest —— 不需要模型：用 GT 合成「理想检测器」轨迹，验证评分器，并直接算出
       MIN_GAP 过紧会丢多少真实落点（给出 0°/45°/90° 的 F1 天花板）

设计原则：落点判定只有一份实现（participant/emit_rule.py）。本工具 import 它，
所以「扫出来的参数」与「线上真正跑的判定」不可能对不上。

评分器的一处不可避免的假设：官方是「一对一匹配」，但未公开具体匹配算法（贪心/匈牙利）。
本工具按「距离升序贪心」匹配；该假设对所有候选参数完全一致，故 A/B 结论不受影响。
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

def _find_task_root(start: Path) -> Path:
    """定位「任务根」= 同时含 core/ 与 participant/ 的目录（即 .../participant/pingpang）。

    刻意不写死 `parent.parent`：本工具是**开发期脚本、不属于提交产物**，允许放在仓库
    任意位置（推荐放到 participant/ 之外，避免被打进提交镜像）。搜索式定位让移动文件
    后无需改代码。

    注意不能只逐级向上查一层：仓库根下的 `participant/` 是**容器路径锚点**（内含
    pingpang/、basketball/），任务根藏在它里面，所以要同时探测常见嵌套位置。
    """
    p = start.resolve()
    for cand in (p, *p.parents):
        for rel in (".", "pingpang", "participant/pingpang", "participant"):
            t = cand / rel if rel != "." else cand
            if (t / "core").is_dir() and (t / "participant").is_dir():
                return t
    return p.parent      # 兜底：维持旧行为


ROOT = _find_task_root(Path(__file__).resolve())
# 顺序很重要：脚本自身所在目录会占 sys.path[0]，若那里恰好也有个 participant/
# （仓库根就是这种情况——它是容器路径锚点），会遮蔽任务内的 participant 包。
# 所以必须把这些路径插到**最前面**。
for _p in (str(ROOT / "participant"), str(ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Windows 控制台/重定向默认 GBK：报表里的 ✅/❌ 会直接抛 UnicodeEncodeError（实测踩到），
# 这里强制 UTF-8 并允许替换字符，保证任何终端下都能把报表打完。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from participant.emit_rule import (EmitRule, PARAMS, describe,    # noqa: E402
                                   params_from_env)

HIT_FRAMES = 1         # 帧号差 ≤ 1
HIT_DIST = 50.0        # 欧氏距离 ≤ 50px
VIEWS = ("0度", "45度", "90度")


# ── 数据加载 ────────────────────────────────────────────────────────────────
def load_gt(path: Path) -> dict:
    """→ {video_id: {"valid": [(f,x,y)], "invalid": [(f,x,y)]}}"""
    out = defaultdict(lambda: {"valid": [], "invalid": []})
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        key = "valid" if d.get("valid") else "invalid"
        out[d["video_id"]][key].append((int(d["frame_id"]), float(d["x"]), float(d["y"])))
    for v in out.values():
        v["valid"].sort()
        v["invalid"].sort()
    return out


def load_videos(input_dir: Path) -> tuple[dict, dict]:
    """→ ({video_id: VideoMeta}, {video_id: angle})；复用 core.dataset 的 schema 校验。"""
    from core.dataset import load_manifest, resolve_manifest

    man = resolve_manifest(str(input_dir))
    videos = load_manifest(man)
    return {v.video_id: v for v in videos}, {v.video_id: v.angle for v in videos}


def load_traj(traj_dir: Path, vid: str) -> list:
    """→ [(frame_id, det|None)]，det=(x,y,conf)。"""
    rows = []
    for line in (traj_dir / f"traj_{vid}.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        det = None if d.get("x") is None else (float(d["x"]), float(d["y"]), float(d["c"]))
        rows.append((int(d["f"]), det))
    return rows


# ── 判定重放（唯一实现 = EmitRule）──────────────────────────────────────────
def replay(rows: list, width: int, height: int, params: dict) -> list:
    """按 EmitRule 语义重放一条轨迹 → [(frame_id, x, y)] 预测。"""
    rule = EmitRule(**params)
    rule.reset(width, height)
    preds = []
    for fid, det in rows:
        for lp in rule.tick(fid, det):
            preds.append((lp.frame_id, lp.x, lp.y))
    for lp in rule.flush():
        preds.append((lp.frame_id, lp.x, lp.y))
    return preds


# ── 官方口径评分 ────────────────────────────────────────────────────────────
def match_counts(preds: list, valid: list, invalid: list) -> tuple[int, int, int]:
    """→ (TP, FP, FN)。一对一：命中无效点的预测被丢弃、不计 FP。"""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(valid):
            if abs(p[0] - g[0]) <= HIT_FRAMES:
                d = math.hypot(p[1] - g[1], p[2] - g[2])
                if d <= HIT_DIST:
                    pairs.append((d, pi, gi))
    pairs.sort()
    used_p, used_g = set(), set()
    tp = 0
    for _d, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1
    fp = 0
    for pi, p in enumerate(preds):
        if pi in used_p:
            continue
        hit_invalid = any(
            abs(p[0] - g[0]) <= HIT_FRAMES and math.hypot(p[1] - g[1], p[2] - g[2]) <= HIT_DIST
            for g in invalid)
        if not hit_invalid:
            fp += 1
    fn = len(valid) - tp
    return tp, fp, fn


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def explain(preds: list, valid: list, invalid: list, times=None) -> dict:
    """与 match_counts 同一套匹配，但返回**逐点**分类，用于逐点复盘。

    每个未被匹配的预测/GT 都同时给出「帧号差」与「像素距离」，因为这两者分别对应
    两类完全不同的失效模式：
      * 帧号差在容差内、像素距离超限  → 落点**位置**判偏（坐标/解码/球台先验问题）
      * 帧号差超容差                  → 落点**时刻**判偏（峰值/门控/帧对齐问题）
      * 该帧根本没有检测              → 检测**漏检**（模型侧，判定逻辑救不了）
    分不清这三类就会往错误的方向调参。
    """
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(valid):
            if abs(p[0] - g[0]) <= HIT_FRAMES:
                d = math.hypot(p[1] - g[1], p[2] - g[2])
                if d <= HIT_DIST:
                    pairs.append((d, pi, gi))
    pairs.sort()
    used_p, used_g = set(), set()
    tp_list = []
    for d, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp_list.append((preds[pi], valid[gi], d))

    det_frames = set(times) if times else None

    def closest(p, pool):
        """最近候选：先按帧差，再按像素距离。返回 (frame_diff, dist, cand)。"""
        best = None
        for g in pool:
            fd = abs(p[0] - g[0])
            d = math.hypot(p[1] - g[1], p[2] - g[2])
            if best is None or (fd, d) < (best[0], best[1]):
                best = (fd, d, g)
        return best

    fp_list = []
    for pi, p in enumerate(preds):
        if pi in used_p:
            continue
        nv = closest(p, valid)
        ni = closest(p, invalid)
        hit_invalid = ni is not None and ni[0] <= HIT_FRAMES and ni[1] <= HIT_DIST
        fp_list.append((p, nv, hit_invalid))
    fn_list = []
    for gi, g in enumerate(valid):
        if gi in used_g:
            continue
        near = None
        for q in preds:
            fd = abs(q[0] - g[0])
            d = math.hypot(q[1] - g[1], q[2] - g[2])
            if near is None or (fd, d) < (near[0], near[1]):
                near = (fd, d, q)
        has_det = (det_frames is None) or (g[0] in det_frames)
        fn_list.append((g, near, has_det))
    return {"tp": tp_list, "fp": fp_list, "fn": fn_list}


def _verdict_fp(p, nv) -> str:
    """给一个 FP 定性，指出该往哪调。"""
    if nv is None:
        return "无任何 GT，纯误检"
    fd, d, g = nv
    if fd <= HIT_FRAMES:
        return f"位置偏（帧差{fd} 在其内，距离{d:.0f}px > 50）"
    return f"时刻偏（最近GT帧差{fd} > 1）"


def _verdict_fn(g, near, has_det) -> str:
    if not has_det:
        return "★该帧无检测（模型漏检，判定逻辑救不了）"
    if near is None:
        return "★该帧无检测（模型漏检）"
    fd, d, q = near
    if fd <= HIT_FRAMES:
        return f"位置偏（帧差{fd} 在其内，距离{d:.0f}px > 50）"
    return f"时刻偏（最近预测帧差{fd} > 1）"


def print_detail(rows: dict, gt: dict, videos: dict, params: dict, title: str) -> None:
    """逐点 FP/FN 明细：0° 只有 21 个有效 GT，逐点复盘是唯一可靠的改进依据。"""
    print(f"\n{title}")
    for vid, m in videos.items():
        g = gt.get(vid, {"valid": [], "invalid": []})
        preds = replay(rows[vid], m.width, m.height, params)
        det_frames = {f for f, d in rows[vid] if d is not None}
        ex = explain(preds, g["valid"], g["invalid"], times=det_frames)
        tp, fp, fn = len(ex["tp"]), len(ex["fp"]), len(ex["fn"])
        if not (fp or fn):
            continue
        print(f"  {vid} [{m.angle}] {m.width}x{m.height} {m.n_frames}帧 "
              f"GT={len(g['valid'])} TP={tp} FP={fp} FN={fn} F1={prf1(tp, fp, fn)[2]:.3f}"
              f"  检测帧={len(det_frames)}")
        for (p, nv, hit_invalid) in ex["fp"]:
            tag = "  [命中无效GT，官方不计FP]" if hit_invalid else ""
            print(f"    FP f={p[0]:>4} ({p[1]:>7.1f},{p[2]:>7.1f})  {_verdict_fp(p, nv)}{tag}")
        for (gg, near, has_det) in ex["fn"]:
            print(f"    FN f={gg[0]:>4} ({gg[1]:>7.1f},{gg[2]:>7.1f})  "
                  f"{_verdict_fn(gg, near, has_det)}")


def load_all(traj_dir: Path, videos: dict) -> dict:
    """一次性把所有轨迹读进内存（扫参时反复重放，避免重复 IO）。"""
    return {vid: load_traj(traj_dir, vid) for vid in videos}


def evaluate(rows_by_vid: dict, gt: dict, videos: dict, params: dict) -> dict:
    """重放参数 + 评分（rows → preds → 官方口径 P/R/F1）。"""
    preds = {vid: replay(rows_by_vid[vid], m.width, m.height, params)
             for vid, m in videos.items()}
    return evaluate_preds(preds, gt, videos)


def evaluate_preds(preds_by_vid: dict, gt: dict, videos: dict) -> dict:
    """→ {view: {tp,fp,fn,p,r,f1,n_gt}, ..., "score": float}；score = Σ 25*F1_视角。"""
    acc = {v: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for v in VIEWS}
    for vid, meta in videos.items():
        g = gt.get(vid, {"valid": [], "invalid": []})
        tp, fp, fn = match_counts(preds_by_vid[vid], g["valid"], g["invalid"])
        a = acc[meta.angle]
        a["tp"] += tp
        a["fp"] += fp
        a["fn"] += fn
        a["n_gt"] += len(g["valid"])
    score = 0.0
    for v in VIEWS:
        a = acc[v]
        a["p"], a["r"], a["f1"] = prf1(a["tp"], a["fp"], a["fn"])
        score += 25.0 * a["f1"]
    acc["score"] = score
    return acc


# ── selftest：无需模型，验证评分器 + 量化 MIN_GAP 的 F1 天花板 ───────────────
def _undetectable(videos: dict, gt: dict) -> dict:
    """结构上不可检出的落点：t >= n_frames-2。

    EmitRule 上报的是「峰值帧 - 1」，而峰值点后面必须还有一个轨迹点（仅尾帧降级为 1 个），
    所以 t = n_frames-2 / n_frames-1 的落点在**任何**实现下都上报不了（参考实现同一写法）。
    这是官方指标里固有的 FN，任何算法都躲不掉，只能如实计入天花板。
    """
    out = {v: 0 for v in VIEWS}
    for vid, meta in videos.items():
        for (t, _x, _y) in gt.get(vid, {"valid": []})["valid"]:
            if t >= meta.n_frames - 2:
                out[meta.angle] += 1
    return out


def synth_rows(meta, valid, slope=3.0, conf=0.9, drift=1.5) -> list:
    """由 GT 合成「理想检测器」轨迹：每个有效落点处 y 都是严格局部极大。

    峰值放在 fid = t_k + 1（正好抵消 ch1 的固有滞后 → 上报帧号 == GT 帧号）。
    两个峰值之间按「前半段以 slope 下降、后半段以 slope 上升」拼接（中间允许跳变）：
    这样每个峰值两侧都恰好下降 slope（严格局部极大 ✅），内部不可能冒出假峰（不会产生 FP）。
    想用包络函数（min/max）会踩坑：min 会造出假峰、max 会盖掉较低的相邻峰值。
    """
    if not valid:
        return [(f, None) for f in range(meta.n_frames)]
    apex = [(t + 1, gx, gy) for (t, gx, gy) in valid if t + 1 < meta.n_frames]
    if not apex:
        return [(f, None) for f in range(meta.n_frames)]
    frames = [a[0] for a in apex]
    rows = []
    for f in range(meta.n_frames):
        i = bisect.bisect_left(frames, f)
        if i < len(apex) and frames[i] == f:
            ref, y = i, apex[i][2]
        else:
            lo, hi = i - 1, i
            if lo < 0:
                ref, y = hi, apex[hi][2] - slope * (frames[hi] - f)
            elif hi >= len(apex):
                ref, y = lo, apex[lo][2] - slope * (f - frames[lo])
            else:
                half = max(1, (frames[hi] - frames[lo]) // 2)
                if f - frames[lo] <= half:
                    ref, y = lo, apex[lo][2] - slope * (f - frames[lo])
                else:
                    ref, y = hi, apex[hi][2] - slope * (frames[hi] - f)
        x = apex[ref][1] + drift * (f - frames[ref])
        rows.append((f, (x, y, conf)))
    return rows


def cmd_selftest(a) -> int:
    gt = load_gt(Path(a.gt) if a.gt else Path(a.input) / "gt_public.jsonl")
    videos, _ = load_videos(Path(a.input))
    rows = {vid: synth_rows(meta, gt.get(vid, {"valid": [], "invalid": []})["valid"])
            for vid, meta in videos.items()}
    base = params_from_env()
    print("=" * 78)
    print("selftest：把所有帧都当成「检测到了且在正确位置」→ 当前判定规则的 F1 天花板")
    print("=" * 78)

    # 1) 端到端自检：门控放到最松，理想检测器只应差「结构上不可检出的末帧落点」
    loose = dict(base, min_gap=2)
    acc = evaluate(rows, gt, videos, loose)
    und = _undetectable(videos, gt)
    print(format_report(acc, loose, "[1] 门控最松（min_gap=2）：理想检测器应只剩末帧落点检不出"))
    for v in VIEWS:
        print(f"    {v}: FN={acc[v]['fn']}（其中结构上不可检出 {und[v]} 个）、FP={acc[v]['fp']}")
    ok = all(acc[v]["fp"] == 0 and acc[v]["fn"] == und[v] for v in VIEWS)
    print(f"    → 评分器/重放/帧对齐自检：{'通过 ✅' if ok else '失败 ❌'}")
    if not ok:
        return 1

    # 2) 负控：整体错 2 帧 / 整体错 60px 必须全部落空
    for name, kind in (("帧号 +2（超出 ±1 容差）", "f"), ("x +60px（超出 50px 容差）", "x")):
        preds = {}
        for vid, m in videos.items():
            ps = replay(rows[vid], m.width, m.height, loose)
            ps = ([(f + 2, x, y) for (f, x, y) in ps] if kind == "f"
                  else [(f, x + 60.0, y) for (f, x, y) in ps])
            preds[vid] = ps
        a2 = evaluate_preds(preds, gt, videos)
        bad = all(a2[v]["tp"] == 0 for v in VIEWS)
        print(f"[2] 负控 {name}: TP="
              f"{ {v: a2[v]['tp'] for v in VIEWS} } → {'全部落空 ✅' if bad else '仍有命中 ❌'}")
        if not bad:
            return 1

    # 3) 天花板：min_gap 从松到紧，理想检测器丢掉的正是「相邻间隔 < min_gap」的真实落点
    print()
    print("[3] MIN_GAP 对 F1 天花板的影响（理想检测器；其余参数 = " + describe(base) + "）")
    print(f"{'min_gap':>8}{'F1_0°':>9}{'F1_45°':>9}{'F1_90°':>9}{'折分/75':>10}{'丢的TP':>8}")
    for mg in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15):
        a3 = evaluate(rows, gt, videos, dict(base, min_gap=mg))
        lost = sum(a3[v]["fn"] for v in VIEWS)
        print(f"{mg:>8}{a3['0度']['f1']:>9.3f}{a3['45度']['f1']:>9.3f}"
              f"{a3['90度']['f1']:>9.3f}{a3['score']:>10.2f}{lost:>8}")
    print("说明：'丢的TP' = 理想检测器下因门控被吞掉的真实落点总数（0 = 天花板无损）。")
    print("      只要 GT 里存在相邻间隔 < min_gap 的落点，这列必然 >0 —— 与模型好坏无关。")
    return 0


# ── score：真正要用的离线扫参（不需要模型）─────────────────────────────────
SWEEP_GRID = {                      # 单参数敏感性（--one-way）用的完整区间
    "min_gap": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    "conf_th": [0.70, 0.75, 0.80, 0.85, 0.90],
    "fall_amp": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
    "rise_amp": [0.0, 0.5, 1.0, 2.0, 3.0],
    "peak_win": [1, 2, 3, 4],
    "peak_conf": [0.0, 0.75, 0.80, 0.85, 0.90],
    "smooth": [0, 1, 2, 3],
}
# 全组合（--sweep）用**粗网格**：7 个参数全量组合会到几万组，既慢又更容易过拟合
FULL_GRID = {
    "min_gap": [8, 10, 12],
    "conf_th": [0.75, 0.80, 0.85],
    "fall_amp": [0.0, 2.0, 4.0],
    "rise_amp": [0.0, 1.0, 2.0],
    "peak_win": [1, 2, 3],
    "peak_conf": [0.0, 0.85, 0.90],
    "smooth": [0, 1],
}


def _grid_product(grid: dict):
    import itertools
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def cmd_score(a) -> int:
    gt_path = Path(a.gt) if a.gt else Path(a.input) / "gt_public.jsonl"
    gt = load_gt(gt_path)
    videos, _ = load_videos(Path(a.input))
    traj = Path(a.traj)
    missing = [vid for vid in videos if not (traj / f"traj_{vid}.jsonl").exists()]
    if missing:
        print(f"缺少轨迹文件（先跑 dump）：{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        return 1
    rows = load_all(traj, videos)
    n_rows = sum(len(r) for r in rows.values())
    n_det = sum(1 for r in rows.values() for _f, d in r if d is not None)
    print(f"轨迹：{len(videos)} 条视频 / {n_rows} 帧 / 有检测 {n_det} 帧"
          f"（{100.0 * n_det / max(n_rows, 1):.0f}%）  GT: {gt_path.name}")
    base = params_from_env()
    acc0 = evaluate(rows, gt, videos, base)
    print(format_report(acc0, base, "\n[A] 当前默认参数（= 参考实现原值，除非设了 PP_* 环境变量）"))
    print(f"    ★ 精度分（满分 75）：{acc0['score']:.2f}")

    if a.per_video:
        print("\n[A2] 逐视频明细（当前参数）——用于核对「不是靠某一条视频刷分」")
        print(f"  {'video':<10}{'视角':<6}{'GT':>4}{'TP':>4}{'FP':>4}{'FN':>4}{'F1':>8}"
              f"{'帧数':>7}")
        for vid, m in videos.items():
            g = gt.get(vid, {"valid": [], "invalid": []})
            preds = replay(rows[vid], m.width, m.height, base)
            tp, fp, fn = match_counts(preds, g["valid"], g["invalid"])
            print(f"  {vid:<10}{m.angle:<6}{len(g['valid']):>4}{tp:>4}{fp:>4}{fn:>4}"
                  f"{prf1(tp, fp, fn)[2]:>8.3f}{len(rows[vid]):>7}")

    if a.detail:
        print_detail(rows, gt, videos, base,
                     "[A3] 逐点 FP/FN 明细（当前参数）——0° 逐点复盘用")

    if a.one_way:
        print("\n[B] 单参数敏感性（其余参数固定默认值）——用于判断「动哪个最值」")
        for name, vals in SWEEP_GRID.items():
            print(f"\n  -- {name}（默认 {base[name]}）--")
            print(f"  {'值':>8}{'F1_0°':>9}{'F1_45°':>9}{'F1_90°':>9}{'得分/75':>10}{'Δ':>9}")
            for v in vals:
                acc = evaluate(rows, gt, videos, dict(base, **{name: v}))
                print(f"  {str(v):>8}{acc['0度']['f1']:>9.3f}{acc['45度']['f1']:>9.3f}"
                      f"{acc['90度']['f1']:>9.3f}{acc['score']:>10.2f}"
                      f"{acc['score'] - acc0['score']:>+9.2f}")

    if a.sweep:
        print("\n[C] 全组合扫描（多维一起动，避免单变量局部最优误判）")
        results = []
        for p in _grid_product(FULL_GRID):
            acc = evaluate(rows, gt, videos, p)
            results.append((acc["score"], dict(p), acc))
        results.sort(key=lambda t: -t[0])
        top = results[:max(1, a.top)]
        print(f"  共 {len(results)} 组；Top {len(top)}：")
        print(f"  {'得分/75':>9}{'Δ基线':>9}  {'min_gap':>8}{'conf_th':>9}"
              f"{'fall_amp':>10}{'rise_amp':>10}")
        for sc, p, _acc in top:
            print(f"  {sc:>9.2f}{sc - acc0['score']:>+9.2f}  {p['min_gap']:>8}"
                  f"{p['conf_th']:>9.2f}{p['fall_amp']:>10.1f}{p['rise_amp']:>10.1f}")
        best_sc, best_p, best_acc = results[0]
        print()
        print(format_report(best_acc, best_p, "[D] 全组合最优"))
        print(f"\n  复现方式（无需改代码）："
              + " ".join(f"{PARAMS[k][0]}={best_p[k]}" for k in PARAMS))
        print("  过拟合警告：公开集仅 15 条视频，0° 有效 GT 只有 21 个，±1 个点 = ±4.8% recall。")
        print("    建议只采纳「三视角都不掉分」的改动（如纯放宽 min_gap），并逐视频核对，")
        print("    而不是直接套用全组合最优参数。")
        if a.json:
            Path(a.json).write_text(json.dumps(
                {"baseline": {k: acc0[k]["f1"] for k in VIEWS} | {"score": acc0["score"]},
                 "top": [{"params": p, "score": sc} for sc, p, _ in top]},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  已写 {a.json}")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="落点检测离线精度调参（开发工具）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="跑一遍模型，逐帧检测结果落盘")
    d.add_argument("--input", required=True, help="数据集根目录（含 manifest.json 与 videos/）")
    d.add_argument("--out", default="/tmp/pp_traj", help="轨迹输出目录")
    d.add_argument("--decoder", choices=["gpu", "cpu", "cv2"], default="gpu",
                   help="gpu=ffmpeg+NVDEC（评测机）；cpu=PyAV；cv2=本机缺前两者时的兜底")
    d.add_argument("--backend", choices=["trt", "ort"], default="trt",
                   help="trt=评测机路径；ort=onnxruntime CPU（本机无 TRT 时做本地测量）")
    d.add_argument("--cache", default=None, help="TRT engine 缓存目录")
    d.add_argument("--ort-dir", default=None,
                   help="onnxruntime 所在目录（未装进 site-packages 时，如解包的 wheel）")
    d.add_argument("--limit", type=int, default=0, help="只跑前 N 条视频（调试用）")
    d.set_defaults(func=cmd_dump)

    s = sub.add_parser("score", help="离线重放 + 官方口径评分 + 扫参")
    s.add_argument("--traj", required=True, help="dump 出来的轨迹目录")
    s.add_argument("--input", required=True)
    s.add_argument("--gt", default=None, help="默认 <input>/gt_public.jsonl")
    s.add_argument("--sweep", action="store_true", help="跑全组合扫描（较慢）")
    s.add_argument("--one-way", action="store_true", help="单参数敏感性表")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--detail", action="store_true",
                   help="逐点打印 FP/FN（帧号+坐标+最近 GT 距离）——0° 逐点复盘必备")
    s.add_argument("--per-video", action="store_true",
                   help="逐视频打印 GT/TP/FP/FN/F1（核对是否靠某一条视频刷分）")
    s.add_argument("--json", default=None, help="把结果写成 json")
    s.set_defaults(func=cmd_score)

    t = sub.add_parser("selftest", help="不需要模型：验证评分器 + 量化 MIN_GAP 天花板")
    t.add_argument("--input", required=True)
    t.add_argument("--gt", default=None)
    t.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


# 入口守卫放在**文件最末**：build_parser 引用的 cmd_* / format_report 必须都已定义。
# （本文件把「报表 + dump」两个次要函数排在 CLI 段之后，故不能在这里就 __main__。）


# ── 报表 ────────────────────────────────────────────────────────────────────
def format_report(acc: dict, params: dict, title: str = "") -> str:
    out = []
    if title:
        out.append(title)
    out.append(f"参数: {describe(params)}")
    out.append(f"{'视角':<6}{'GT':>5}{'TP':>5}{'FP':>5}{'FN':>5}"
               f"{'P':>8}{'R':>8}{'F1':>8}{'折分/25':>9}")
    for v in VIEWS:
        a = acc[v]
        out.append(f"{v:<6}{a['n_gt']:>5}{a['tp']:>5}{a['fp']:>5}{a['fn']:>5}"
                   f"{a['p']:>8.3f}{a['r']:>8.3f}{a['f1']:>8.3f}{25 * a['f1']:>9.2f}")
    out.append(f"{'合计':<6}{'':>5}{'':>5}{'':>5}{'':>5}{'':>8}{'':>8}{'':>8}"
               f"{acc['score']:>9.2f}")
    return "\n".join(out)


# ── dump：跑一遍模型，逐帧检测结果落盘（只需跑一次）────────────────────────
class OrtInfer:
    """onnxruntime 推理——**仅本机测量用的兜底**（评测机走 TRT fp16，数值有极小差异）。

    provider 自动协商：装了 onnxruntime-gpu 就优先 CUDA（本机 2109 帧 CPU 太慢），
    否则退回 CPU。**只影响测速**，不影响解码语义。

    预处理/解码**复用提交代码本身**（FrameTiler + decode_heat），所以测的是本方案真正的
    输入分布与解码语义，而不是另写一套近似实现。
    """

    def __init__(self, onnx_path, ort_dir=None, log=print):
        if ort_dir:
            sys.path.insert(0, str(ort_dir))
        import onnxruntime as ort
        from participant.ball_detector import OUT_CHANNEL

        avail = ort.get_available_providers()
        # TensorRT EP 构建耗时且会与稍后的官方 TRT 数值混淆，这里只用 CUDA/CPU
        want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        if not want:
            want = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0                    # 用满物理核
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=want)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.out_channel = int(OUT_CHANNEL)
        self.provider = self.sess.get_providers()[0]
        self.how = f"ort_{ort.__version__}_{self.provider.replace('ExecutionProvider','').lower()}"
        log(f"[dump] {self.how}：in={self.in_name} out={self.out_name} "
            f"providers={self.sess.get_providers()}")

    def infer(self, x_np):
        """(1,9,288,512) float32 -> ch=1 的 (288,512) 热图。"""
        y = self.sess.run([self.out_name], {self.in_name: x_np})[0]
        return y[0, self.out_channel]


def iter_frames_cv2(path):
    """cv2(FFmpeg) 解码 → RGB uint8——本机既无 PyAV 也无 ffmpeg.exe 时的兜底。"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 打不开 {path}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def iter_frames(meta, input_dir: Path, mode: str):
    """统一帧源：gpu/cpu 走 core.decoder（会校验帧数==manifest），cv2 为本地兜底。"""
    if mode == "cv2":
        yield from enumerate(iter_frames_cv2(input_dir / meta.path))
    else:
        from core.decoder import decode_video
        yield from decode_video(meta, input_dir, mode)


def cmd_dump(a) -> int:
    from participant.pp_fast import FrameTiler, decode_heat, gpu_index_from_env

    videos, _ = load_videos(Path(a.input))
    input_dir = Path(a.input)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = ROOT / "participant" / "weights" / "ball.onnx"

    if a.backend == "ort":
        infer = OrtInfer(onnx, a.ort_dir or os.environ.get("PP_ORT_DIR"), log=print)
    else:
        from participant.pp_fast import FastTrt, RefTrt
        cache = Path(a.cache) if a.cache else out_dir / "trt_cache"
        cache.mkdir(parents=True, exist_ok=True)
        dev = gpu_index_from_env()
        try:
            infer = FastTrt(onnx, cache, device=dev, log=print)
        except Exception as exc:  # noqa: BLE001 —— dump 工具只求拿到数据，回退即可
            print(f"[dump] FastTrt 不可用（{exc}），回退 RefTrt")
            infer = RefTrt(onnx, cache, device=dev, log=print)

    ids = list(videos)
    if a.limit:
        ids = ids[:a.limit]
    for vid in ids:
        meta = videos[vid]
        tiler = FrameTiler()
        n = 0
        with (out_dir / f"traj_{vid}.jsonl").open("w", encoding="utf-8") as fh:
            for fid, img in iter_frames(meta, input_dir, a.decoder):
                if img.shape[0] != meta.height or img.shape[1] != meta.width:
                    raise RuntimeError(f"{vid}: 解码尺寸 {img.shape[1]}x{img.shape[0]}"
                                       f" != manifest {meta.width}x{meta.height}")
                heat = infer.infer(tiler.push(img))
                det = decode_heat(heat, meta.width, meta.height)
                row = {"f": fid, "m": round(float(heat.max()), 6)}
                if det is None:
                    row.update({"x": None, "y": None, "c": None})
                else:
                    row.update({"x": det[0], "y": det[1], "c": det[2]})
                fh.write(json.dumps(row) + "\n")
                n += 1
        flag = "" if n == meta.n_frames else f"  !! 帧数 {n} != manifest {meta.n_frames}"
        print(f"[dump] {vid} [{meta.angle}] {n} 帧 -> traj_{vid}.jsonl{flag}", flush=True)

    (out_dir / "dump_meta.json").write_text(json.dumps({
        "input": str(a.input), "decoder": a.decoder, "backend": getattr(infer, "how", "?"),
        "n_videos": len(ids), "params_for_replay": params_from_env(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[dump] 完成。离线扫参：\n"
          f"  python participant/_tune_eval.py score --traj {out_dir} --input {a.input} --sweep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
