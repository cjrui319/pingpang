"""落点判定：轨迹清洗 + 触台"速度折角"检测（纯 numpy，无框架依赖）。

数据流：``{frame_id: [(x, y, conf), ...]}``（多通道冗余观测）
        → ``fuse_obs`` 合并成 ``{frame_id: (x, y, conf)}``
        → ``clean_track`` 剔除离群点得到按帧排序的 ``[(f, x, y, conf)]``
        → ``find_landings`` 用"球心速度矢量的剧烈折转"定位触台帧。

物理依据（与 SPIN 数据集论文的 bounce-inflection 方法一致：滑动窗口拟合 →
竖直速度符号反转 → NMS）：
  触台瞬间只有**竖直速度反向**（水平速度基本不变）。该突变在图像上表现为
  球心速度矢量的一次折转，且与视角无关：

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
    """把同一帧的多条观测合并成一条：以最高置信度为种子做邻域聚类后取均值。

    若模型三通道确实各对应一帧，则三条观测几乎重合 → 求平均（降噪）；
    若只有部分通道可用，则离群观测被 ``cluster_eps`` 过滤掉。
    """
    out: dict[int, tuple[float, float, float]] = {}
    for fid, obs in per_frame.items():
        if not obs:
            continue
        if len(obs) == 1:
            out[fid] = (float(obs[0][0]), float(obs[0][1]), float(obs[0][2]))
            continue
        seed = max(obs, key=lambda o: o[2])
        grp = [o for o in obs
               if math.hypot(o[0] - seed[0], o[1] - seed[1]) <= cluster_eps]
        xs = [o[0] for o in grp]
        ys = [o[1] for o in grp]
        cs = [o[2] for o in grp]
        out[fid] = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)),
                    float(max(cs)))
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
    与筛选阈值无关，因此可被离线网格搜索缓存（只算一次）。

    ``max_step``：紧邻的前后观测必须在这么多帧内。轨迹断口（丢检/末端野点）会
    制造出 ±100px/帧² 量级的伪折角——实测这类伪折角是 FP 的主要来源之一。

    注：曾试过额外要求「速度符号真正反转」（v前>0 且 v后<0），实测反而把 F1
    从 0.816 拉到 0.651（0° 俯视角下球接近球台的方向不恒定，符号约束误杀真触台），
    故不采用。
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


def pick_landings(cands: list[tuple[float, float, float, float, float, float]],
                  dv_y: float = -10.0, max_dvx_ratio: float = 0.5,
                  min_gap: float = 6.0, nms: str = "maxy"
                  ) -> list[tuple[int, float, float, float]]:
    """从折角信号里挑出落点（阈值 + 方向一致 + NMS）。"""
    keep = [c for c in cands
            if c[3] <= dv_y and abs(c[4]) <= max_dvx_ratio * abs(c[3])]
    if nms == "dv":
        ordered = sorted(keep, key=lambda c: -c[3] * c[5])   # 折角最猛者优先
    else:
        ordered = sorted(keep, key=lambda c: -c[2])          # 最低点优先
    picked: list[tuple[int, float, float, float]] = []
    for c in ordered:
        if all(abs(c[0] - p[0]) >= min_gap for p in picked):
            picked.append((int(c[0]), c[1], c[2], -c[3] * c[5]))
    picked.sort(key=lambda t: t[0])
    return picked


def find_landings(points: list[tuple[float, float, float, float]], cfg: dict
                  ) -> list[tuple[int, float, float, float]]:
    """返回 ``[(frame_id, x, y, score), ...]``（按帧号升序）。

    ``cfg`` 字段：

    * ``k``              每侧参与速度拟合的点数（默认 2）
    * ``span``           单侧最长回溯/前探帧距（默认 6）
    * ``max_step``       紧邻前后观测的最大帧距（默认 2；轨迹断口会伪造巨折角）
    * ``dv_y``           Δv_y 阈值，越小越严（默认 -10.0）
    * ``max_dvx_ratio``  |Δv_x| / |Δv_y| 上限（默认 0.5）
    * ``min_gap``        两个落点之间的最小帧间隔（NMS，默认 6）
    * ``nms``            重叠候选保留谁：``maxy`` 取最低点 / ``dv`` 取折角最猛者
    """
    cands = kink_candidates(points, int(cfg.get("k", 2)), float(cfg.get("span", 6)),
                            float(cfg.get("max_step", 2.0)))
    return pick_landings(cands, float(cfg.get("dv_y", -10.0)),
                         float(cfg.get("max_dvx_ratio", 0.5)),
                         float(cfg.get("min_gap", 6)),
                         str(cfg.get("nms", "maxy")))
