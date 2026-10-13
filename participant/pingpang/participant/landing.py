"""落点判定：轨迹清洗 + 触台"折角"检测（纯 numpy，无框架依赖）。

数据流：``{frame_id: [(x, y, conf), ...]}``（多通道冗余观测）
        → ``fuse_obs`` 合并成 ``{frame_id: (x, y, conf)}``
        → ``clean_track`` 剔除离群点得到按帧排序的 ``[(f, x, y, conf)]``
        → ``find_landings`` 用"球心速度矢量的剧烈折转"定位触台帧。

物理依据：触台瞬间只有**竖直速度反向**（水平速度基本不变），该突变在图像上
表现为球心速度矢量的一次折转，且与视角无关：

* 90 度（侧视）：折角最大，此时恰好等价于"y 的局部极大"；
* 0/45 度（沿球台纵深俯视）：触台后球同时远离相机，``y`` 可能继续增大，
  "y 取极大"这一朴素判据会整体偏移 3~8 帧（实测 F1 只有 0.05），而 ``Δv_y``
  的负向尖峰仍精确落在触台帧上。

判据：``Δv_y = v_y(后) - v_y(前)`` 足够负（下落被瞬间抬升），且
``|Δv_x| <= ratio·|Δv_y|``——后者是"竖直反跳在图像中的投影方向"约束，能滤掉
大量由轨迹跳变/跟错目标造成的伪折角（实测把 45 度 P 从 0.53 提到 0.78）。

触台坐标直接取该帧观测到的球心：真值即标注者在触台帧标下的球心位置，实测
模型在真值帧上的球心与真值中位差约 7px（远小于 50px 匹配容差），无需再外推。
"""

from __future__ import annotations

import math

import numpy as np


def fuse_obs(per_frame: dict, cluster_eps: float = 20.0
             ) -> dict[int, tuple[float, float, float]]:
    """把同一帧的多条观测合并成一条：贪心聚类后取「数量最多」那簇（多数票）。

    之前以「最高置信度」为种子聚类，会把多通道一致的真球（弱峰）当噪声丢——
    因为孤立干扰峰往往单通道、置信度反而更高。改成多数票：真球通常是多通道
    一致的那簇，孤立点只有一条通道。数量平手再比置信度。
    """
    out: dict[int, tuple[float, float, float]] = {}
    for fid, obs in per_frame.items():
        if not obs:
            continue
        if len(obs) == 1:
            out[fid] = (float(obs[0][0]), float(obs[0][1]), float(obs[0][2]))
            continue
        # 贪心聚类：按置信度降序取种子，把 eps 内的点并进同一簇。
        remaining = sorted(obs, key=lambda o: -o[2])
        clusters: list[tuple[float, float, float, int]] = []
        while remaining:
            seed = remaining.pop(0)
            grp = [seed]
            i = 0
            while i < len(remaining):
                if math.hypot(remaining[i][0] - seed[0],
                              remaining[i][1] - seed[1]) <= cluster_eps:
                    grp.append(remaining.pop(i))
                else:
                    i += 1
            clusters.append((sum(o[0] for o in grp) / len(grp),
                             sum(o[1] for o in grp) / len(grp),
                             max(o[2] for o in grp), len(grp)))
        best = max(clusters, key=lambda c: (c[3], c[2]))  # 数量优先，平手看置信度
        out[fid] = (float(best[0]), float(best[1]), float(best[2]))
    return out


def clean_track(points: list[tuple[float, float, float, float]],
                max_resid: float = 30.0, max_gap: int = 6,
                rounds: int = 2) -> list[tuple[float, float, float, float]]:
    """剔除离群点：与前后邻点线性插值的偏差超过 ``max_resid`` 且置信度不高者丢弃。

    ``points`` 必须是按帧号升序的 ``(f, x, y, conf)`` 列表。迭代 ``rounds`` 轮，
    这样"成对出现"的离群点也能被清理掉（单轮只看 i-1/i+1 时会互相掩护）。
    """
    pts = list(points)
    for _ in range(max(1, rounds)):
        if len(pts) < 3:
            break
        n = len(pts)
        keep: list[tuple[float, float, float, float]] = []
        for i in range(n):
            if i == 0 or i == n - 1:
                keep.append(pts[i])
                continue
            f0, x0, y0, _ = pts[i - 1]
            f1, x1, y1, _ = pts[i + 1]
            if (f1 - f0) <= 0 or (f1 - f0) > 2 * max_gap:
                keep.append(pts[i])
                continue
            f, x, y, c = pts[i]
            t = (f - f0) / (f1 - f0)
            ex = x0 + t * (x1 - x0)
            ey = y0 + t * (y1 - y0)
            if math.hypot(x - ex, y - ey) > max_resid and c < 0.9:
                continue
            keep.append(pts[i])
        if len(keep) == len(pts):
            break
        pts = keep
    return pts


def _line_slope(F: np.ndarray, V: np.ndarray, idx: np.ndarray, f0: float) -> float:
    """对 ``idx`` 选中的点拟合 ``V = p·(F - f0) + q``，返回斜率 p。"""
    t = F[idx] - f0
    if len(idx) == 2:
        dt = float(t[1] - t[0])
        return float((V[idx[1]] - V[idx[0]]) / dt) if dt else 0.0
    A = np.stack([t, np.ones_like(t)], axis=1)
    return float(np.linalg.lstsq(A, V[idx], rcond=None)[0][0])


def kink_candidates(points: list[tuple[float, float, float, float]],
                    k: int = 2, span: float = 6.0, max_step: float = 2.0
                    ) -> list[tuple[float, float, float, float, float, float]]:
    """逐帧计算"折角"原始信号，返回 ``[(f, x, y, Δv_y, Δv_x, conf), ...]``。

    每个候选帧用**前后各 k 个点**分别线性拟合 x/y，得到触台前后两个速度矢量，
    ``Δv = v_后 - v_前``。真触台的 ``Δv_y`` 是负尖峰；``Δv_x`` 用于方向一致性过滤。
    与筛选阈值无关，因此可被 dev/tune.py 缓存后网格搜索（只算一次）。

    ``max_step``：紧邻的前后观测必须在这么多帧内。轨迹断口（丢检/末端野点）会
    制造出 ±100px/帧² 量级的伪折角——实测这类伪折角是 FP 的主要来源之一。
    """
    n = len(points)
    if n < 3:
        return []
    F = np.array([p[0] for p in points], dtype=np.float64)
    X = np.array([p[1] for p in points], dtype=np.float64)
    Y = np.array([p[2] for p in points], dtype=np.float64)
    C = np.array([p[3] for p in points], dtype=np.float64)

    out: list[tuple[float, float, float, float, float, float]] = []
    for i in range(n):
        left = np.where((F < F[i]) & (F >= F[i] - span))[0][-k:]
        right = np.where((F > F[i]) & (F <= F[i] + span))[0][:k]
        if len(left) < 2 or len(right) < 2:
            continue                      # 两侧都要有观测量出前后速度
        if (F[i] - F[left[-1]] > max_step) or (F[right[0]] - F[i] > max_step):
            continue                      # 紧邻观测太远 → 断口，不可信
        dvy = _line_slope(F, Y, right, F[i]) - _line_slope(F, Y, left, F[i])
        dvx = _line_slope(F, X, right, F[i]) - _line_slope(F, X, left, F[i])
        out.append((float(F[i]), float(X[i]), float(Y[i]), dvy, dvx, float(C[i])))
    return out


def _nms_key(c: tuple[float, float, float, float, float, float], nms: str) -> float:
    """NMS 排序键：**键越小越优先**。"""
    if nms == "dv":
        return c[3] * c[5]                    # 折角最猛者优先（等价 -(-Δv_y·conf)）
    if nms == "combo":
        # 折角强度 × 方向一致性：Δv_y 负得越多、|Δv_x|/|Δv_y| 越小越优先。
        # 实测比 maxy 更稳：同一窗口里"最猛的一折"才是触台帧，而最低点可能落在
        # 触台后 1~2 帧（球继续远离相机，y 更低），导致整窗选错帧。
        return -((-c[3]) * c[5] / (1.0 + abs(c[4]) / max(abs(c[3]), 1e-9)))
    return -c[2]                              # maxy：最低点优先（历史行为）


def pick_landings(cands: list[tuple[float, float, float, float, float, float]],
                  dv_y: float = -10.0, max_dvx_ratio: float = 0.5,
                  min_gap: float = 6.0, nms: str = "maxy",
                  dv_strong: float = 0.0, ratio_strong: float = 0.0
                  ) -> list[tuple[int, float, float, float]]:
    """从折角信号里挑出落点（阈值 + 方向一致 + NMS）。

    ``dv_strong``/``ratio_strong``：强折角的宽松档。侧旋或斜视角下真触台的
    水平速度也可能明显变化（实测 45 度有真落点 ``|Δv_x|/|Δv_y|`` 高达 0.65~0.92
    被 0.5 的硬门槛误杀，而同窗口被保留下来的反而是偏差 50px+ 的假折角）。
    因此当 ``|Δv_y| >= dv_strong``（折角远超噪声，几乎不可能是轨迹跳变）时，
    放宽到 ``|Δv_x| <= ratio_strong·|Δv_y|``。设 0 关闭该档。

    ``nms``：``maxy`` 最低点优先 / ``dv`` 折角最猛者优先 / ``combo`` 折角强度×
    方向一致性优先。
    """
    strong = dv_strong > 0.0
    keep = []
    for c in cands:
        if c[3] > dv_y:
            continue
        if abs(c[4]) <= max_dvx_ratio * abs(c[3]):
            keep.append(c)
        elif strong and abs(c[3]) >= dv_strong and abs(c[4]) <= ratio_strong * abs(c[3]):
            keep.append(c)
    ordered = sorted(keep, key=lambda c: _nms_key(c, nms))
    picked: list[tuple[int, float, float, float]] = []
    for c in ordered:
        if all(abs(c[0] - p[0]) >= min_gap for p in picked):
            picked.append((int(c[0]), c[1], c[2], -c[3] * c[5]))
    picked.sort(key=lambda t: t[0])
    return picked


def table_roi(cfg_roi, width: float, height: float):
    """把配置里的归一化球台投影区 ``[x0, y0, x1, y1]``（0~1 比例）换成像素框。

    归一化而非绝对像素：视频分辨率变了照样能用（整套参数都是按公开验证集
    的机位标定的，属于机位常量，和 ``dv_y``/``max_step`` 一样是超参）。
    """
    if not cfg_roi:
        return None
    x0, y0, x1, y1 = (float(v) for v in cfg_roi)
    return (x0 * width, y0 * height, x1 * width, y1 * height)


def keep_in_roi(picked: list[tuple[int, float, float, float]], roi,
                min_keep: float = 0.5) -> list[tuple[int, float, float, float]]:
    """丢掉球台投影区之外的落点（触台只可能发生在台面上）。

    这是本方案里收益最大的单点改动：0/45/90 度各有相当一部分 FP 是**球已离台后
    在台外（地面/挡板）反弹**——那确实是一次真实的抛物线折角，Δv_y 信号和台内
    触台无法区分，只有"落点位置不在台面投影内"能判掉。

    fail-open 保护：若过滤后只剩不到 ``min_keep`` 比例，说明 ROI 先验与当前视频
    的机位不符（换了机位/裁剪），此时整体放弃过滤，宁可放过 FP 也不要丢掉真落点。
    """
    if not roi or len(picked) < 2:
        return picked
    x0, y0, x1, y1 = roi
    kept = [p for p in picked if x0 <= p[1] <= x1 and y0 <= p[2] <= y1]
    if len(kept) < min_keep * len(picked):
        return picked
    return kept


def find_landings(points: list[tuple[float, float, float, float]], cfg: dict
                  ) -> list[tuple[int, float, float, float]]:
    """返回 ``[(frame_id, x, y, score), ...]``（按帧号升序）。

    ``cfg`` 字段：

    * ``k``              每侧参与速度拟合的点数（默认 2）
    * ``span``           单侧最长回溯/前探帧距（默认 6）
    * ``max_step``       紧邻前后观测的最大帧距（默认 2；轨迹断口会伪造巨折角）
    * ``dv_y``           Δv_y 阈值，越小越严（默认 -10.0）
    * ``max_dvx_ratio``  |Δv_x| / |Δv_y| 上限（默认 0.5）
    * ``dv_strong``      强折角档：|Δv_y| 超过它时改用下面的 ratio（默认 0=关闭）
    * ``ratio_strong``   强折角档的 |Δv_x| / |Δv_y| 上限
    * ``min_gap``        两个落点之间的最小帧间隔（NMS，默认 6）
    * ``nms``            重叠候选保留谁：``maxy`` 最低点 / ``dv`` 折角最猛者 /
                         ``combo`` 折角强度×方向一致性
    """
    cands = kink_candidates(points, int(cfg.get("k", 2)), float(cfg.get("span", 6)),
                            float(cfg.get("max_step", 2.0)))
    return pick_landings(cands, float(cfg.get("dv_y", -10.0)),
                         float(cfg.get("max_dvx_ratio", 0.5)),
                         float(cfg.get("min_gap", 6)),
                         str(cfg.get("nms", "maxy")),
                         float(cfg.get("dv_strong", 0.0)),
                         float(cfg.get("ratio_strong", 0.0)))


def _depart_slope(points: list[tuple[float, float, float, float]],
                  f_land: float, k: int = 3, span: float = 6.0) -> float | None:
    """落点后 k 帧内的 y 斜率（负 = 球向上弹起，正 = 继续下落/贴台滚）。"""
    idx = [i for i, p in enumerate(points) if abs(p[0] - f_land) <= 1]
    if not idx:
        return None
    i = idx[0]
    after = [(points[j][0], points[j][2]) for j in range(i + 1, min(len(points), i + k + 1))
             if points[j][0] - points[i][0] <= span]
    if len(after) < 2:
        return None
    f0, y0 = after[0]
    f1, y1 = after[-1]
    return (y1 - y0) / (f1 - f0)


def rally_filter(points: list[tuple[float, float, float, float]],
                 raw_landings: list[tuple[int, float, float, float]],
                 roi=None, angle: str | None = None) -> list[tuple[int, float, float, float]]:
    """回合状态机：用轨迹 + 球台信息抑制干扰落点（不动 Δv_y 判据）。

    三类干扰（公开验证集实测，全部是真实强反弹但标注员不算）：

    1. **回合后弹跳**：球飞出/被收走后轨迹进入「静态阶段」——连续 >= 8 帧位置
       几乎不动（< 3px/帧）。球在回合中不可能停住 8 帧以上，出现静态阶段即球已
       消失，其后（重新活跃前）的落点属于新回合/场外，抑制。
    2. **桌边反弹**：球撞台边弹飞，离开速度极端（|y 斜率| > 60 px/帧），球不再
       回来。正常触台的向上弹起斜率最多 ~40，台边反弹会到 100+。
    3. **二次弹跳**：相邻两落点帧距 <= 8 且落点接近（< 200px）且后者弹起弱
       （|斜率| < 20）——球几乎没离开台面就再触一次，标注员只算第一次。
    4. **90 度同侧连续**：侧视下正常回合球每次触台必过网（左右交替），连续两次
       落在球台同侧 = 回合后/无回球，抑制后者。
    """
    if not raw_landings:
        return []

    # 1) 静态阶段：找第一个 >= 8 帧的「位置几乎不动」段
    static_end: float | None = None
    start = None
    for i in range(1, len(points)):
        dx = points[i][1] - points[i - 1][1]
        dy = points[i][2] - points[i - 1][2]
        if math.hypot(dx, dy) < 3.0:
            if start is None:
                start = i - 1
        else:
            if start is not None and (i - start) >= 8:
                static_end = points[i - 1][0]
                break
            start = None

    dep = {f: _depart_slope(points, f) for f, _x, _y, _s in raw_landings}

    # 4) 90 度：球台 x 中线
    mid_x = None
    if angle == "90度" and roi and len(roi) == 4:
        mid_x = (roi[0] + roi[2]) / 2.0

    kept: list[tuple[int, float, float, float]] = []
    prev = None
    prev_side = None
    for f, x, y, _s in raw_landings:
        if static_end is not None and f > static_end:
            continue
        s = dep.get(f)
        # 2) 桌边反弹：极端离开速度
        if s is not None and abs(s) > 60.0:
            continue
        # 3) 二次弹跳
        if prev is not None:
            pf, px, py = prev
            d = math.hypot(x - px, y - py)
            if (f - pf) <= 8 and d < 200.0 and s is not None and abs(s) < 20.0:
                continue
        # 4) 90 度同侧连续
        if mid_x is not None and prev_side is not None:
            side = "L" if x < mid_x else "R"
            if side == prev_side:
                continue
        kept.append((f, x, y, _s))
        prev = (f, x, y)
        if mid_x is not None:
            prev_side = "L" if x < mid_x else "R"
    return kept
