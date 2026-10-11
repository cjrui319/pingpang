"""离线精度评测 + 调参工具（开发用，**不属于提交产物**）。

本文件在仓库根，刻意不放进 ``participant/``——它会读 ``gt_public.jsonl`` 真值，
而 participant/ 整个目录会被打进提交镜像。

口径（与赛题细则一致）：
  * 帧号容差 ±1、位置容差 50px，同一视频内**一对一**匹配（Kuhn 最大基数匹配，
    保证容差内命中数最大——这是官方口径，贪心会低估 TP）；
  * 只有 ``valid=true`` 的 GT 参与召回统计（算 FN）；
  * 命中 ``valid=false`` 干扰点（二次弹跳/擦边/遮挡）的预测被"忽略"，
    既不计 TP 也不计 FP；
  * 精度分 = F1_0×25 + F1_45×25 + F1_90×25。

子命令：
  1) dump —— 跑一遍模型（复用提交的 BallDetector 多通道解码），把每帧的冗余
     观测落盘成 ``traj_<vid>.jsonl``（每行一个观测：f/x/y/c/offset）。
  2) score —— 纯离线：重放 landing.find_landings + 官方口径评分 + 参数扫描。
  3) selftest —— 不需要模型：验证评分器与重放链路。

设计原则：落点判定只有一份实现（participant/pingpang/participant/landing.py）。
本工具 import 它，所以「扫出来的参数」与「线上真正跑的判定」不可能对不上。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find_task_root(start: Path) -> Path:
    """定位任务根 = 同时含 core/ 与 participant/ 的目录（.../participant/pingpang）。"""
    p = start.resolve()
    for cand in (p, *p.parents):
        for rel in (".", "pingpang", "participant/pingpang", "participant"):
            t = cand / rel if rel != "." else cand
            if (t / "core").is_dir() and (t / "participant").is_dir():
                return t
    return p.parent


ROOT = _find_task_root(_HERE)
for _p in (str(ROOT / "participant"), str(ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from participant.landing import find_landings, clean_track, fuse_obs  # noqa: E402
from participant.ball_detector import BallDetector, IN_H, IN_W          # noqa: E402

HIT_FRAMES = 1
HIT_DIST = 50.0
VIEWS = ("0度", "45度", "90度")

# ── 数据加载 ────────────────────────────────────────────────────────────────
def load_gt(path: Path) -> dict:
    """→ {video_id: {"valid": [(f,x,y)], "invalid": [(f,x,y)]}}"""
    out = defaultdict(lambda: {"valid": [], "invalid": []})
    for line in Path(path).read_text(encoding="utf-8").splitlines():
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
    from core.dataset import load_manifest, resolve_manifest
    man = resolve_manifest(str(input_dir))
    videos = load_manifest(man)
    return {v.video_id: v for v in videos}, {v.video_id: v.angle for v in videos}


def load_obs(traj_dir: Path, vid: str) -> list:
    """读 dump 的观测文件 → 按帧号升序的 [(f, x, y, conf)]（已按帧融合）。"""
    per_frame: dict[int, list[tuple[float, float, float]]] = {}
    for line in (traj_dir / f"traj_{vid}.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        per_frame.setdefault(int(d["f"]), []).append((float(d["x"]), float(d["y"]), float(d["c"])))
    fused = fuse_obs(per_frame)
    return sorted((f, x, y, c) for f, (x, y, c) in fused.items())


# ── 判定重放（唯一实现 = landing.find_landings）─────────────────────────────
def replay(points: list, cfg: dict, conf_gate: float = 0.6) -> list:
    """按 landing 语义重放 → [(frame_id, x, y)] 预测。

    与提交的 solution._analyze 对齐：先 conf_gate 过滤（低于门限的轨迹点不参与
    拟合），再做 clean_track + find_landings。
    """
    gated = [p for p in points if p[3] >= conf_gate]
    if len(gated) >= 3:
        points = gated
    cleaned = clean_track(points,
                          max_resid=float(cfg.get("max_resid", 30.0)),
                          max_gap=int(cfg.get("max_gap", 6)))
    raw = find_landings(cleaned, cfg)
    return [(f, x, y) for f, x, y, _s in raw]


# ── 官方口径评分（Kuhn 最大基数匹配）────────────────────────────────────────
def max_matching(adj: list, n_right: int) -> list:
    """Kuhn 最大基数匹配；返回长度 n_right 的 match（右侧点 -> 左侧点）。"""
    match_r = [-1] * n_right

    def dfs(u: int, seen: list) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    for u in range(len(adj)):
        dfs(u, [False] * n_right)
    return match_r


def match_counts(P: list, G: list) -> tuple[int, int, int, int]:
    """单视频匹配，返回 (tp, fp, fn, ignored)。G = [(fid,x,y,valid)]。"""
    adj = [[] for _ in P]
    for i, (pf, px, py) in enumerate(P):
        for j, (gf, gx, gy, _v) in enumerate(G):
            if abs(pf - gf) <= HIT_FRAMES and math.hypot(px - gx, py - gy) <= HIT_DIST:
                adj[i].append(j)
    match_r = max_matching(adj, len(G)) if G else []
    matched = {u for u in match_r if u != -1}
    tp = sum(1 for j, u in enumerate(match_r) if u != -1 and G[j][3])
    ign = sum(1 for j, u in enumerate(match_r) if u != -1 and not G[j][3])
    fn = sum(1 for g in G if g[3]) - tp
    fp = len(P) - len(matched)
    return tp, fp, fn, ign


def prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(points_by_vid: dict, gt: dict, videos: dict, cfg: dict) -> dict:
    acc = {v: {"tp": 0, "fp": 0, "fn": 0, "ign": 0, "n_gt": 0} for v in VIEWS}
    for vid, meta in videos.items():
        g = gt.get(vid, {"valid": [], "invalid": []})
        G = [(f, x, y, True) for f, x, y in g["valid"]] + \
            [(f, x, y, False) for f, x, y in g["invalid"]]
        preds = replay(points_by_vid[vid], cfg)
        tp, fp, fn, ign = match_counts(preds, G)
        a = acc[meta.angle]
        a["tp"] += tp
        a["fp"] += fp
        a["fn"] += fn
        a["ign"] += ign
        a["n_gt"] += len(g["valid"])
    score = 0.0
    for v in VIEWS:
        a = acc[v]
        a["p"], a["r"], a["f1"] = prf1(a["tp"], a["fp"], a["fn"])
        score += 25.0 * a["f1"]
    acc["score"] = score
    return acc


def format_report(acc: dict, cfg: dict, title: str = "") -> str:
    out = []
    if title:
        out.append(title)
    out.append(f"参数: {json.dumps(cfg, ensure_ascii=False)}")
    out.append(f"{'视角':<6}{'GT':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'忽略':>6}"
               f"{'P':>8}{'R':>8}{'F1':>8}{'折分/25':>9}")
    for v in VIEWS:
        a = acc[v]
        out.append(f"{v:<6}{a['n_gt']:>5}{a['tp']:>5}{a['fp']:>5}{a['fn']:>5}{a['ign']:>6}"
                   f"{a['p']:>8.3f}{a['r']:>8.3f}{a['f1']:>8.3f}{25 * a['f1']:>9.2f}")
    out.append(f"{'合计':<6}{'':>5}{'':>5}{'':>5}{'':>5}{'':>6}{'':>8}{'':>8}{'':>8}"
               f"{acc['score']:>9.2f}")
    return "\n".join(out)


# ── dump：跑一遍模型，落盘逐帧观测 ──────────────────────────────────────────
def _ort_infer(onnx_path: Path, decoder: str, ort_dir=None, log=print):
    if ort_dir:
        sys.path.insert(0, str(ort_dir))
    import onnxruntime as ort
    avail = ort.get_available_providers()
    want = [p for p in (["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if decoder == "gpu" else ["CPUExecutionProvider"]) if p in avail]
    if not want:
        want = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(str(onnx_path), so, providers=want)
    in_name = sess.get_inputs()[0].name
    log(f"[dump] ort providers={sess.get_providers()} in={in_name}")

    def infer(x):
        return sess.run(None, {in_name: x})[0]
    return infer


def _trt_infer(onnx_path: Path, cache: Path, decoder: str, log=print):
    from core.trt import load_or_build
    model, how = load_or_build(onnx_path, cache, input_name="input",
                               shape=(1, 9, IN_H, IN_W), log=log)
    log(f"[dump] tensorrt {how}")
    return model.infer


def iter_frames_cv2(path):
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


def cmd_dump(a) -> int:
    videos, _ = load_videos(Path(a.input))
    input_dir = Path(a.input)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = ROOT / "participant" / "weights" / "ball.onnx"

    if a.backend == "ort":
        infer = _ort_infer(onnx, a.decoder, a.ort_dir, log=print)
    else:
        cache = Path(a.cache) if a.cache else Path(a.out) / "trt_cache"
        cache.mkdir(parents=True, exist_ok=True)
        infer = _trt_infer(onnx, cache, a.decoder, log=print)

    # 复用提交的 BallDetector（多通道，channels=[0,1,2]）
    det = BallDetector(infer, onnx, score_th=0.55, channels=(0, 1, 2), peak_radius=14)

    ids = list(videos)
    if a.limit:
        ids = ids[:a.limit]
    for vid in ids:
        meta = videos[vid]
        det.reset()
        n = 0
        with (out_dir / f"traj_{vid}.jsonl").open("w", encoding="utf-8") as fh:
            for fid, img in enumerate(iter_frames_cv2(input_dir / meta.path)):
                for off, x, y, c in det.detect(img, meta.width, meta.height):
                    fh.write(json.dumps({"f": fid + off, "x": x, "y": y, "c": c,
                                         "offset": off}) + "\n")
                n += 1
        print(f"[dump] {vid} [{meta.angle}] {n} 帧 -> traj_{vid}.jsonl", flush=True)
    print(f"[dump] 完成。评分：python tune_eval.py score --traj {out_dir} --input {a.input}")
    return 0


def _iter_gpu_cpu(meta, input_dir: Path, mode: str):
    from core.decoder import decode_video
    yield from decode_video(meta, input_dir, mode)


# ── score ───────────────────────────────────────────────────────────────────
def cmd_score(a) -> int:
    gt_path = Path(a.gt) if a.gt else Path(a.input) / "gt_public.jsonl"
    gt = load_gt(gt_path)
    videos, _ = load_videos(Path(a.input))
    traj = Path(a.traj)
    missing = [v for v in videos if not (traj / f"traj_{v}.jsonl").exists()]
    if missing:
        print(f"缺少轨迹文件（先跑 dump）：{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        return 1
    points = {vid: load_obs(traj, vid) for vid in videos}

    # 分视角配置（与提交的 configs/landing.json 一致）
    base_cfg = {"dv_y": -12.0, "max_dvx_ratio": 0.5, "min_gap": 5.0, "nms": "maxy",
                "k": 2, "span": 4.0, "max_step": 1.0, "max_resid": 30.0, "max_gap": 6}
    per_view = {"0度": dict(base_cfg, dv_y=-10.0),
                "45度": dict(base_cfg),
                "90度": dict(base_cfg, max_dvx_ratio=1e9, min_gap=6.0)}

    # 按视频分视角评估（提交也是按视角取配置）
    acc = {v: {"tp": 0, "fp": 0, "fn": 0, "ign": 0, "n_gt": 0} for v in VIEWS}
    for vid, meta in videos.items():
        cfg = per_view[meta.angle]
        g = gt.get(vid, {"valid": [], "invalid": []})
        G = [(f, x, y, True) for f, x, y in g["valid"]] + \
            [(f, x, y, False) for f, x, y in g["invalid"]]
        preds = replay(points[vid], cfg)
        tp, fp, fn, ign = match_counts(preds, G)
        st = acc[meta.angle]
        st["tp"] += tp
        st["fp"] += fp
        st["fn"] += fn
        st["ign"] += ign
        st["n_gt"] += len(g["valid"])
        if a.per_video:
            p, r, f1 = prf1(tp, fp, fn)
            print(f"  {vid:<10}{meta.angle:<6}{len(g['valid']):>4}{tp:>4}{fp:>4}{fn:>4}"
                  f"{ign:>6}{p:>8.3f}{r:>8.3f}{f1:>8.3f}")
    score = 0.0
    for v in VIEWS:
        st = acc[v]
        st["p"], st["r"], st["f1"] = prf1(st["tp"], st["fp"], st["fn"])
        score += 25.0 * st["f1"]
    acc["score"] = score
    print(format_report(acc, {"per_view": True}, "\n官方口径（Kuhn 匹配）"))
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="落点检测离线精度评测（开发工具）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="跑一遍模型，落盘逐帧观测")
    d.add_argument("--input", required=True)
    d.add_argument("--out", default="/tmp/pp_traj")
    d.add_argument("--decoder", choices=["gpu", "cpu", "cv2"], default="cv2")
    d.add_argument("--backend", choices=["trt", "ort"], default="ort")
    d.add_argument("--cache", default=None)
    d.add_argument("--ort-dir", default=None)
    d.add_argument("--limit", type=int, default=0)
    d.set_defaults(func=cmd_dump)

    s = sub.add_parser("score", help="离线重放 + 官方口径评分")
    s.add_argument("--traj", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--gt", default=None)
    s.add_argument("--per-video", action="store_true")
    s.set_defaults(func=cmd_score)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
