"""验证 FrameTiler 的滑窗通道顺序是否与参考实现 BallDetector 一致。

不依赖 cv2：用 stub 顶替 cv2.resize（只做形状变换，数值用 numpy 复现），
本测试只关心「哪一帧落在哪个通道组」这个结构性质。
"""
import sys
import types
from pathlib import Path

import numpy as np

# ---- 目录布局（pp_fast/ball_detector 在 participant/，core 在其上一级）----
HERE = Path(__file__).resolve().parent          # participant/
ROOT = HERE.parent                              # pingpang/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# Windows 控制台默认 GBK：结论行里的 ✅/❌ 会抛 UnicodeEncodeError（会把「通过」变成
# 退出码 1 的假失败）。强制 UTF-8 + 替换字符，保证退出码只反映判定结果。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ---- cv2 stub：只实现 resize(frame, (w,h), dst=..., interpolation=...) ----
cv2 = types.ModuleType("cv2")
cv2.INTER_LINEAR = 1


def _resize(src, dsize, dst=None, interpolation=None):
    w, h = dsize
    out = np.empty((h, w, src.shape[2]), src.dtype)
    for c in range(src.shape[2]):
        out[:, :, c] = src[:h, :w, c]
    if dst is not None:
        dst[...] = out
        return dst
    return out


cv2.resize = _resize
sys.modules["cv2"] = cv2

from ball_detector import BallDetector          # noqa: E402
import importlib                                # noqa: E402

# pp_fast 用的是相对导入（from .ball_detector import ...），必须从包内导入
sys.path.insert(0, str(ROOT))
import participant as _pkg                      # noqa: E402
sys.modules.setdefault("participant", _pkg)
FrameTiler = importlib.import_module("participant.pp_fast").FrameTiler

H, W = 288, 512          # 直接给出已 resize 的尺寸，使 stub 的裁剪成为恒等映射
NFRAMES = 6

# 每帧一个可识别的常量值，便于追踪它落在哪个通道组
frames = [np.full((H, W, 3), v, np.uint8) for v in (10, 20, 30, 40, 50, 60)]

print("=" * 72)
print("参考实现 BallDetector: concat(buf[-3:])，通道组 = [0:3]/[3:6]/[6:9]")
det = BallDetector(lambda x: np.zeros((1, 3, H, W), np.float32), "dummy.onnx")
ref_trace = []
for i, f in enumerate(frames):
    det._buf.append(det._normalize(f))
    det._buf = det._buf[-3:]
    inp = np.concatenate(det._buf, axis=0)[None]
    groups = tuple(round(float(inp[0, 3 * k].mean()), 4) for k in range(3))
    ref_trace.append(groups)
    print(f"  帧{i}: 通道组[0:3]={groups[0]:>8}  [3:6]={groups[1]:>8}  [6:9]={groups[2]:>8}")

print()
print("=" * 72)
print("当前实现 FrameTiler: 参考语义拷贝（本帧恒落组[6:9]）")
tiler = FrameTiler(in_h=H, in_w=W)
cur_trace = []
for i, f in enumerate(frames):
    inp = tiler.push(f)
    groups = tuple(round(float(inp[0, 3 * k].mean()), 4) for k in range(3))
    cur_trace.append(groups)
    print(f"  帧{i}: 通道组[0:3]={groups[0]:>8}  [3:6]={groups[1]:>8}  [6:9]={groups[2]:>8}")

print()
print("=" * 72)
print("逐帧比对结果：")
mismatch = 0
for i, (r, c) in enumerate(zip(ref_trace, cur_trace)):
    ok = (r == c)
    if not ok:
        mismatch += 1
    print(f"  帧{i}: {'一致 ✅' if ok else '不一致 ❌'}"
          f"{'' if ok else f'   参考={r}  当前={c}'}")

print()
if mismatch == 0:
    print("结论：通道顺序与参考实现完全一致，逐字节等价的前提成立。")
else:
    print(f"结论：❌ 有 {mismatch}/{len(frames)} 帧通道顺序不一致！")
    print("      → 模型输入的时序语义被改变，精度保证（逐字节等价）不成立。")
    sys.exit(1)
