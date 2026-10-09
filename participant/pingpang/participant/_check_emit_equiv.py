"""差分测试：真实判定器 emit_rule.EmitRule 的「增量游标扫描」是否与参考实现的
「每帧全量重扫」严格同结果。

不依赖 cv2 / torch / 模型——只重建两条判定路径，用固定轨迹喂进去逐步比对。

    * 参考路径 RefEmit：照抄原 solution.py 的写法
          for j in range(1, n - need_future): ... 命中则 last_emit = fid
      （含「conf >= CONF_TH 才进 traj」的过滤，与参考实现一致）
    * 被测路径 NewEmit：**直接调用真实现** participant/emit_rule.py 的 EmitRule。

为什么必须 import 真实现：早先这个测试自己复制了一份判定代码，后果是——改了
solution.py 的参数，测试照样全绿。现在参数也从 emit_rule 取，测试才真的在保护它。

关于环境变量：本测试会**先清空所有 PP_ 前缀的环境变量**，再读参数，
强制按默认参数比较。理由是这个测试要证明的是「默认参数下与参考实现逐字等价」，
若允许 PP_* 覆盖，参考侧的基准也会跟着变，测试就失去意义了。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # participant/
ROOT = HERE.parent                              # pingpang/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# Windows 控制台默认 GBK：结论行里的 ✅/❌ 会抛 UnicodeEncodeError，把「逻辑全过」
# 变成「退出码 1」的假失败（实测踩到）。强制 UTF-8 + 替换字符，保证退出码只反映判定。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import os

# 任何 PP_* 环境变量都会改变"参考语义"的比较基准，让这个测试失去意义
# （它要证明的是「默认参数下与参考实现逐字等价」）。
# 本测试**强制按默认参数**运行：先清掉 PP_*，再读参数。
_CLEARED = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("PP_")}

from participant.emit_rule import EmitRule, describe, params_from_env  # noqa: E402

P = params_from_env()                # 现在必然等于 DEFAULT_*（PP_* 已清空）
MIN_GAP, CONF_TH, FALL_AMP, RISE_AMP = (P["min_gap"], P["conf_th"],
                                        P["fall_amp"], P["rise_amp"])

# peak_win / peak_conf 是「默认关闭的加强判据」，官方参考实现里没有对应物——
# 本测试只证明「默认关闭状态下逐字等价」，故这里强制按关闭态比较。
_DEV = {"peak_win": P["peak_win"], "peak_conf": P["peak_conf"], "smooth": P["smooth"]}
print(f"参数: {describe(P)}")
if _CLEARED:
    print(f"  已清空环境变量 {sorted(_CLEARED)}：本测试只验证默认参数（清空前 {_CLEARED}）")
if _DEV["peak_win"] != 1 or _DEV["peak_conf"] != 0.0 or _DEV["smooth"] != 0:
    print(f"  ⚠️ 非默认的加强判据生效中 {_DEV}——参考实现无对应物，"
          "本测试只覆盖默认关闭态的等价性。")


# ── 参考：每帧全量重扫 ──────────────────────────────────────────────────────
class RefEmit:
    def __init__(self):
        self.traj = []
        self.last_emit = -10_000

    def append(self, fid, x, y, conf):
        if conf >= CONF_TH:                  # 参考实现就是在这里过滤的
            self.traj.append((fid, x, y, conf))

    def emit(self, need_future):
        out = []
        n = len(self.traj)
        for j in range(1, n - need_future):
            fid, x, y, conf = self.traj[j]
            if fid - self.last_emit < MIN_GAP:
                continue
            prev_y = self.traj[j - 1][2]
            next_y = self.traj[j + 1][2]
            if (y > prev_y and y >= next_y
                    and (y - prev_y) >= FALL_AMP
                    and (y - next_y) >= RISE_AMP):
                out.append((max(0, fid - 1), x, y))
                self.last_emit = fid
        return out


# ── 当前：增量游标 ──────────────────────────────────────────────────────────
class NewEmit:
    """适配器：直接使用 participant/emit_rule.py 的**真实现**。

    语义映射：EmitRule.tick() 一次完成「按 conf_th 进 traj + need_future=2 扫描」，
    恰好对应参考实现的 append(...) + emit(2) 两步；尾部的 emit(1) 对应 flush()。
    """

    def __init__(self):
        self.rule = EmitRule(min_gap=MIN_GAP, conf_th=CONF_TH,
                             fall_amp=FALL_AMP, rise_amp=RISE_AMP,
                             peak_win=1, peak_conf=0.0)   # 强制关闭态（见文件头说明）
        self.rule.reset(1920, 1080)
        self._out = []

    @property
    def traj(self):
        return self.rule.traj

    def append(self, fid, x, y, conf):
        self._out = self.rule.tick(fid, (x, y, conf))

    def emit(self, need_future):
        out = self._out if need_future == 2 else self.rule.flush()
        # 归一成 (frame_id, x, y) 元组 —— 参考实现的输出就是元组。
        # （真实现返回 core.types.Landing，值相同、类型不同，直接比会假报不等）
        return [(lp.frame_id, lp.x, lp.y) for lp in out]


def make_traj(rng: random.Random, n: int, pattern: str):
    """构造一条确定性轨迹。pattern 决定「球怎么动」，以制造足够多的局部极大。"""
    out = []
    fid = 0
    while len(out) < n:
        if pattern == "sawtooth":       # 锯齿：稳定的下落-反弹，落点密集
            for k in range(1, 8):
                out.append((fid, 100.0 + k, 200.0 + k * 3.0, rng.uniform(0.6, 0.99)))
                fid += 1
                if len(out) >= n:
                    break
            for k in range(1, 4):
                out.append((fid, 100.0, 221.0 - k * 8.0, rng.uniform(0.6, 0.99)))
                fid += 1
                if len(out) >= n:
                    break
        elif pattern == "flat":         # 几乎不动：不应触发任何落点
            out.append((fid, 500.0, 300.0 + rng.uniform(-0.5, 0.5), rng.uniform(0.6, 0.99)))
            fid += 1
        elif pattern == "noise":        # 纯随机游走
            out.append((fid, rng.uniform(0, 1919), rng.uniform(0, 1079),
                        rng.uniform(0.5, 1.0)))
            fid += 1
        else:
            raise ValueError(pattern)
    return out[:n]


def run_case(seed: int, n: int, pattern: str, tail: bool):
    rng = random.Random(seed)
    traj = make_traj(rng, n, pattern)

    ref, new = RefEmit(), NewEmit()
    diffs = []
    for i, (fid, x, y, c) in enumerate(traj):
        ref.append(fid, x, y, c)
        new.append(fid, x, y, c)
        r = ref.emit(2)
        v = new.emit(2)
        if r != v:
            diffs.append((i, r, v))
    if tail:                            # 视频结束时的尾帧降级
        r = ref.emit(1)
        v = new.emit(1)
        if r != v:
            diffs.append((n, r, v))
    return diffs, ref, new


def main() -> int:
    total_cases = 0
    bad = []
    for pattern in ("sawtooth", "flat", "noise"):
        for n in (1, 2, 3, 4, 5, 10, 25, 60, 200):
            for seed in range(12):
                for tail in (False, True):
                    total_cases += 1
                    d, ref, new = run_case(seed, n, pattern, tail)
                    if d:
                        bad.append((pattern, n, seed, tail, d[0]))
    print(f"跑了 {total_cases} 组用例（pattern × 帧数 × 随机种子 × 是否尾部 flush）")
    print(f"被测实现：participant/emit_rule.py EmitRule；参数：{describe(P)}")
    if not bad:
        print("结果：真实现的增量游标扫描与参考全量重扫 **逐组完全一致** ✅")
        print("      → solution.py 中「结果与全量重扫严格相同」的声明成立")
        return 0
    print(f"结果：❌ {len(bad)} 组不一致，前 3 例：")
    for pattern, n, seed, tail, first in bad[:3]:
        print(f"  pattern={pattern} n={n} seed={seed} tail={tail}")
        print(f"    首个分歧在第 {first[0]} 步: 参考={first[1]} 当前={first[2]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
