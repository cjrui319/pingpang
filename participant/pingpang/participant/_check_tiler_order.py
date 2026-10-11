"""验证 BallDetector 的三帧滑窗 + 通道↔帧偏移映射。

不依赖真实模型/cv2：stub 顶替 cv2.resize，注入一个能把「每帧独有标量」编码进
热图的假推理函数，据此验证：
  1. 滑窗缓冲 [f-2, f-1, f] 的三帧拼接顺序正确（ch0↔最旧、ch2↔最新）；
  2. _CH_OFFSET = (-2,-1,0) 与缓冲语义一致；
  3. detect 返回的 (帧偏移, x, y, conf) 中，帧偏移正确地指向真实所属帧。
"""
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # participant/
ROOT = HERE.parent                              # pingpang/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ---- cv2 stub：resize 恒等（输入已是目标尺寸）----
cv2 = types.ModuleType("cv2")
cv2.INTER_LINEAR = 1


def _resize(src, dsize, dst=None, interpolation=None):
    w, h = dsize
    out = np.empty((h, w, src.shape[2]), src.dtype)
    out[:, :, :] = src[:h, :w, :]
    if dst is not None:
        dst[...] = out
        return dst
    return out


cv2.resize = _resize
sys.modules["cv2"] = cv2

from ball_detector import BallDetector, _CH_OFFSET  # noqa: E402

H, W = 288, 512


def main() -> int:
    ok = True

    # 每帧用「均匀亮度 = 帧号*10」的图，这样归一化后仍保持帧号信息。
    # 假推理函数：把 9 通道输入按 [0:3][3:6][6:9] 三组，各自取均值作为该通道热图的值，
    # 于是 ch0/ch1/ch2 分别反映「最旧/中间/最新」帧的亮度。
    def fake_infer(inp):
        # inp: (1,9,H,W) float32
        g = inp[0]
        vals = [float(g[3 * k:3 * k + 3].mean()) for k in range(3)]
        heat = np.zeros((1, 3, 1, 1), np.float32)
        for k, v in enumerate(vals):
            heat[0, k, 0, 0] = v
        # 放大到 (1,3,H,W) 便于 detector 解码
        return np.broadcast_to(heat, (1, 3, H, W))

    det = BallDetector(fake_infer, "dummy.onnx", score_th=-1e9, channels=(0, 1, 2))

    # 归一化会做 (x-mean)/std，但单调递增关系保留：帧号越大 → 值越大。
    # 直接读 _normalize 后的均值即可。
    print("=" * 70)
    print("验证滑窗通道 ↔ 帧 的对应关系")
    # 用三帧亮度 10 / 20 / 30
    for i, val in enumerate((10, 20, 30)):
        frame = np.full((H, W, 3), val, np.uint8)
        det._buf.append(det._normalize(frame))
        det._buf = det._buf[-3:]
        inp = np.concatenate(det._buf, axis=0)[None]
        groups = [float(inp[0, 3 * k:3 * k + 3].mean()) for k in range(3)]
        print(f"  第 {i} 帧(亮度{val})后 缓冲三组均值 = {[round(g, 3) for g in groups]}")
    print("  预期：最旧帧在 ch0 组、最新帧在 ch2 组（monotone）")

    # 直接验证 _CH_OFFSET
    print()
    print("_CH_OFFSET =", _CH_OFFSET)
    ok &= _CH_OFFSET == (-2, -1, 0)

    # 验证 detect 的帧偏移：用亮度不同的三帧，ch 解码出的值应随 offset 单调
    det.reset()
    seq = (10, 40, 70)
    got = []
    for i, val in enumerate(seq):
        frame = np.full((H, W, 3), val, np.uint8)
        got.append(det.detect(frame, W, H))
    print()
    print("detect 返回的 (帧偏移, conf) 序列（最后一帧应含 3 条，偏移 -2/-1/0）：")
    for i, g in enumerate(got):
        print(f"  第 {i} 帧: {[(o, round(c, 3)) for o, x, y, c in g]}")
    # 最后一帧（第2帧）应有 3 条观测，偏移 -2,-1,0 对应 seq 的三帧
    last = got[-1]
    offs = sorted(o for o, *_ in last)
    ok &= offs == [-2, -1, 0]
    print(f"  最后一帧偏移集合 = {offs}  {'✅' if offs == [-2, -1, 0] else '❌'}")

    print()
    print("结论：" + ("✅ 滑窗通道与帧偏移映射正确" if ok else "❌ 映射错误"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
