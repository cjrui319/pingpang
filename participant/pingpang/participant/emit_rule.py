"""落点判定规则：球心 y 的局部极大 + 下落/反弹幅度门控（纯 Python，不依赖 cv2/numpy 之外的东西）。

为什么单独成模块（而不是留在 solution.py 里）：
    **调参与提交必须共用同一份实现。** 离线扫参工具 `_tune_eval.py`、等价性测试
    `_check_emit_equiv.py`、正式提交 `solution.py` 全部 import 这里的 `EmitRule`；
    否则「扫出来的最优参数」和「线上真正跑的判定逻辑」会各自漂移——典型事故是
    测试全绿但线上参数根本没生效。

语义（与官方参考实现逐字一致，含固有 1 帧滞后）：
    * 轨迹点：仅当该帧有检测且 conf >= conf_th 时进入 traj；
    * 判定：对 traj 的 j 检查 y 的局部极大与下落/反弹幅度；
    * 门控：`fid_j - last_emit >= min_gap`（last_emit 单调不减 → 可增量扫描）；
    * 帧对齐：ch1 解码的是 3 帧栈的中间帧（滞后 1），上报 max(0, fid-1)；
    * 尾部：视频结束时允许只依赖 1 个未来点（need_future=1）再判一次，避免丢末尾落点。

实测结论（public_data 15 条视频 2109 帧；用 participant/_tune_eval.py 本地复现）：
    基线 min_gap=10 conf_th=0.75 fall_amp=2.0 rise_amp=1.0 → 折分 45.00/75
    （0° TP10/FP16→F1 0.426、45° 28/18→0.667、90° 23/9→0.708）。
    **失分主要在精度（FP）而非漏检**。7 个方向逐一实测，参考实现原值全部是局部最优，
    所以下面这些开关默认全部关闭（它们只是为了「可复测/可再验证」而保留）：
      * min_gap 放松到 8 / 6：天花板确实更高（理想检测器 92/92 命中 vs 85/92），但真实
        检测器的噪声峰随之被放进来，FP 增加 → 42.34 / 43.18 分，反而低于 45.00。
        ⇒「放宽 min_gap 提召回」在本数据集上被证伪。
      * peak_win = 2 / 3 / 4（±w 帧窗口取极值）→ 42.43 / 39.47 / 36.15（真峰很窄，先丢 TP）。
      * peak_conf = 0.8 / 0.85 / 0.9（峰值帧置信下限）→ 44.16 / 42.07 / 36.02。
      * smooth = 1 / 2 / 3（判定前对 y 做 ±s 帧均值）→ 24.86 / 6.20 / 1.64（最差：平滑直接
        抹掉要判的那个尖峰）。
    结论：这组启发式参数已到局部最优，继续提分要靠更好的检测/跟踪（模型侧），而不是在这
    7 个阈值上继续拧。任何改动前请先用
    `python3 participant/_tune_eval.py score --traj <dump 目录> --one-way --per-video` 复测。

参数默认值 = 官方参考实现原值；同时支持用环境变量覆盖，**只为 A/B 测量**（不设变量时行为与
参考实现完全一致）。若将来在评测机的 TRT 轨迹上确证某组参数更优，把数值写死在 DEFAULT_*。
"""

from __future__ import annotations

import os
from typing import Optional

from core.types import Landing

__all__ = ["EmitRule", "PARAMS", "PARAM_NAMES", "params_from_env", "describe",
           "MIN_GAP", "CONF_TH", "FALL_AMP", "RISE_AMP"]


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEFAULT_MIN_GAP = 10     # 两次落点最小帧间隔
DEFAULT_CONF_TH = 0.75   # 轨迹点置信门槛
DEFAULT_FALL_AMP = 2.0   # 判定前的最小下落幅度（px，图像 y）
DEFAULT_RISE_AMP = 1.0   # 判定后的最小反弹幅度（px，图像 y）
DEFAULT_PEAK_WIN = 1     # 局部极大比较窗口（±win 帧）；1 = 只看相邻两点（参考实现）
DEFAULT_PEAK_CONF = 0.0  # 峰值帧自身的置信下限（0 = 不额外要求，与参考实现一致）
DEFAULT_SMOOTH = 0       # 判定用的 y 时域平滑半窗（0 = 不平滑，与参考实现一致）

# 兼容别名 = **参考实现原值**（固定值，不受 PP_* 环境影响）。
# 需要「当前生效值」请用 params_from_env()，或直接读 EmitRule 实例的同名属性。
MIN_GAP, CONF_TH, FALL_AMP, RISE_AMP = (DEFAULT_MIN_GAP, DEFAULT_CONF_TH,
                                        DEFAULT_FALL_AMP, DEFAULT_RISE_AMP)

# 环境变量名 → (默认值)。A/B 时用 PP_MIN_GAP=6 之类即可，无需改代码。
PARAMS = {
    "min_gap": ("PP_MIN_GAP", DEFAULT_MIN_GAP),
    "conf_th": ("PP_CONF_TH", DEFAULT_CONF_TH),
    "fall_amp": ("PP_FALL_AMP", DEFAULT_FALL_AMP),
    "rise_amp": ("PP_RISE_AMP", DEFAULT_RISE_AMP),
    "peak_win": ("PP_PEAK_WIN", DEFAULT_PEAK_WIN),
    "peak_conf": ("PP_PEAK_CONF", DEFAULT_PEAK_CONF),
    "smooth": ("PP_SMOOTH", DEFAULT_SMOOTH),
}
PARAM_NAMES = tuple(PARAMS)


def params_from_env() -> dict:
    """当前生效的参数（含环境变量覆盖）。"""
    out = {}
    for name, (env, default) in PARAMS.items():
        if isinstance(default, int):
            out[name] = _env_int(env, default)
        else:
            out[name] = _env_float(env, default)
    return out


def describe(params: Optional[dict] = None) -> str:
    """参数串。不传则描述**当前生效**的参数（含 PP_* 覆盖），传 dict 则描述该组参数。

    为什么不默认用 DEFAULT_*：solution.py 会把这一串写进 run.log 的 [pp_profile] 行，
    而 A/B 时真正跑的是环境变量覆盖后的值——记录参考默认值会误导排查。
    """
    p = params_from_env() if params is None else params
    return " ".join(f"{k}={p[k]}" for k in PARAM_NAMES)


class EmitRule:
    """轨迹流式喂入 → 滚出落点。状态：traj / last_emit / 两个增量游标。

    增量游标（相对参考实现「每帧全量重扫」的等价改写，已由 648 组用例验证）：
    游标只前进不回退；被门控挡掉的 j 之后只会更被挡掉，局部极大只依赖 j±1，
    故已扫描过的 j 不会改变结论，总复杂度 O(n)。
    """

    __slots__ = ("min_gap", "conf_th", "fall_amp", "rise_amp",
                 "peak_win", "peak_conf", "smooth", "traj",
                 "last_emit", "_cur2", "_cur1", "_w", "_h", "n_emitted",
                 "n_ticks")

    def __init__(self, min_gap: int = None, conf_th: float = None,
                 fall_amp: float = None, rise_amp: float = None,
                 peak_win: int = None, peak_conf: float = None,
                 smooth: int = None):
        env = params_from_env()
        self.min_gap = env["min_gap"] if min_gap is None else min_gap
        self.conf_th = env["conf_th"] if conf_th is None else conf_th
        self.fall_amp = env["fall_amp"] if fall_amp is None else fall_amp
        self.rise_amp = env["rise_amp"] if rise_amp is None else rise_amp
        self.peak_win = env["peak_win"] if peak_win is None else peak_win
        self.peak_conf = env["peak_conf"] if peak_conf is None else peak_conf
        self.smooth = env["smooth"] if smooth is None else smooth
        self.traj: list[tuple[int, float, float, float]] = []
        self.last_emit = -10_000
        self._cur2 = 1
        self._cur1 = 1
        self._w = 1920.0
        self._h = 1080.0
        self.n_emitted = 0
        self.n_ticks = 0

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def reset(self, width: float = 1920, height: float = 1080) -> None:
        self.traj.clear()
        self.last_emit = -10_000
        self._cur2 = 1
        self._cur1 = 1
        self._w = float(width)
        self._h = float(height)
        self.n_emitted = 0
        self.n_ticks = 0

    @property
    def n_traj(self) -> int:
        return len(self.traj)

    # ── 每帧 ────────────────────────────────────────────────────────────────
    def tick(self, fid: int, det: Optional[tuple]) -> list:
        """喂入一帧。det = (x, y, conf) 或 None。返回本帧确认出的落点列表。"""
        if det is not None and det[2] >= self.conf_th:
            self.traj.append((fid, det[0], det[1], det[2]))
        self.n_ticks += 1
        # 需要 w 个未来点才能判完 ±w 窗口；w=1 时 == 参考实现的 need_future=2
        return self._scan(max(2, self.peak_win + 1))

    def flush(self) -> list:
        """视频结束：尾部降级为 need_future=1 再判一次。"""
        return self._scan(1)

    # ── 判定核心（peak_win=1 且 peak_conf=0 时逐字对齐参考实现）─────────────
    def _scan(self, need_future: int) -> list:
        out: list = []
        traj = self.traj
        n = len(traj)
        max_j = n - need_future - 1
        j = self._cur2 if need_future == 2 else self._cur1
        if j < 1:
            j = 1
        last_emit = self.last_emit
        w, h = self._w, self._h
        win = self.peak_win
        sm = self.smooth
        fast = win <= 1 and sm <= 0        # 参考实现路径：与原来逐字相同、零额外开销
        while j <= max_j:
            fid, x, y, conf = traj[j]
            if fid - last_emit >= self.min_gap and conf >= self.peak_conf:
                if fast:
                    left_max, right_max, y_sig = traj[j - 1][2], traj[j + 1][2], y
                else:
                    # 判定信号：可选时域平滑（抑制抖动造成的假峰）+ 可选 ±win 窗口取极值
                    lo = j - win if j - win > 0 else 0
                    hi = j + win if j + win < n else n - 1
                    y_sig = self._sig(j, sm, n)
                    left_max = max(self._sig(k, sm, n) for k in range(lo, j))
                    right_max = max(self._sig(k, sm, n) for k in range(j + 1, hi + 1))
                if (y_sig > left_max and y_sig >= right_max      # 局部极大
                        and (y_sig - left_max) >= self.fall_amp   # 之前在下落
                        and (y_sig - right_max) >= self.rise_amp):  # 之后在反弹
                    out.append(Landing(
                        frame_id=max(0, fid - 1),            # ch1 固有滞后 1 帧
                        x=min(max(x, 0.0), w - 1.0),         # 规格要求：坐标须落在画面内
                        y=min(max(y, 0.0), h - 1.0),         # 仍上报**原始** y（不平滑）
                    ))
                    last_emit = fid
                    self.n_emitted += 1
            j += 1
        self.last_emit = last_emit
        if need_future == 2:
            self._cur2 = j
        else:
            self._cur1 = j
        return out

    def _sig(self, k: int, sm: int, n: int) -> float:
        """判定用的 y 信号：sm<=0 时就是原始 y（参考实现等价）；sm>0 时为 ±sm 帧均值。

        只影响「是否判为峰值」，上报坐标仍是原始 x/y（坐标语义与参考实现保持一致）。
        """
        traj = self.traj
        if sm <= 0:
            return traj[k][2]
        lo = k - sm if k - sm > 0 else 0
        hi = k + sm if k + sm < n else n - 1
        tot = 0.0
        for t in range(lo, hi + 1):
            tot += traj[t][2]
        return tot / (hi - lo + 1)
