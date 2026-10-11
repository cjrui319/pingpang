"""球检测：热图模型 → 逐帧球心（参考实现的重写版，接口向后兼容但语义更完整）。

模型 weights/ball.onnx：
  input  (1, 9, 288, 512)   3 帧 RGB 栈（ImageNet mean/std 归一化）
  output (1, 3, 288, 512)   3 个时间位置的热图（已含 sigmoid）

通道语义（与参考实现的 3 帧滑窗一致，经 _check_tiler_order.py 差分验证）：
  滑窗缓冲为 [f-2, f-1, f]，对应输出通道 ch0 → f-2、ch1 → f-1、ch2 → f。
  因此处理第 f 帧时，一次前向可同时给出三个真实帧的球心观测（冗余观测）。

相对参考实现的差异（参考实现只在整幅热图上求加权质心，易被第二个 blob 拉偏）：
  * 只统计峰值附近 ``peak_radius`` 内的像素（球所在连通域），不再对全图求和；
  * 阈值 / 使用通道 / 峰值半径全部可配置；
  * ``detect`` 返回 ``[(帧偏移, x, y, conf), ...]``（x/y 为原分辨率像素），
    让上层能把多通道的冗余观测合并到同一帧上（见 landing.fuse_obs）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IN_H, IN_W = 288, 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 通道 c ↔ 3 帧缓冲区第 c 帧 ↔ 相对当前帧的偏移 c-2
_CH_OFFSET = (-2, -1, 0)


class BallDetector:
    """球检测器：依赖外部注入的推理函数 ``infer_fn``（(1,9,H,W) -> (1,3,H,W)）。"""

    def __init__(self, infer_fn, onnx_path, score_th: float = 0.55,
                 channels=(1,), peak_radius: int = 14):
        self._infer = infer_fn
        self.onnx_path = Path(onnx_path)
        self.score_th = float(score_th)
        self.channels = tuple(int(c) for c in channels)
        self.peak_radius = int(peak_radius)
        self._blank = self._normalize(np.zeros((IN_H, IN_W, 3), np.uint8))
        self.reset()

    # ── 预处理 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(frame_rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 -> (3,288,512) float32 归一化（与参考实现同算式）。"""
        x = cv2.resize(frame_rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def reset(self) -> None:
        # 预填 2 帧空白：处理第 f 帧时缓冲区为 [f-2, f-1, f]
        self._buf = [self._blank, self._blank]

    # ── 推理 + 解码 ─────────────────────────────────────────────────────────
    def detect(self, frame_rgb: np.ndarray, orig_w: int, orig_h: int
               ) -> list[tuple[int, float, float, float]]:
        """返回 ``[(帧偏移, x, y, conf), ...]``；帧偏移 = 相对当前帧（<=0）。"""
        self._buf.append(self._normalize(frame_rgb))
        self._buf = self._buf[-3:]
        inp = np.concatenate(self._buf, axis=0)[None]  # (1,9,288,512)
        heat = self._infer(inp)[0]                     # (3,288,512)
        sx, sy = orig_w / IN_W, orig_h / IN_H
        out: list[tuple[int, float, float, float]] = []
        for ch in self.channels:
            got = self._decode(heat[ch])
            if got is None:
                continue
            cx, cy, conf = got
            out.append((_CH_OFFSET[ch], cx * sx, cy * sy, conf))
        return out

    def _decode(self, heat2d: np.ndarray):
        """取峰值所在连通域的加权质心；峰值低于阈值返回 None。"""
        h, w = heat2d.shape
        idx = int(np.argmax(heat2d))
        py, px = divmod(idx, w)
        peak = float(heat2d[py, px])
        if peak < self.score_th:
            return None
        r = self.peak_radius
        y0, y1 = max(0, py - r), min(h, py + r + 1)
        x0, x1 = max(0, px - r), min(w, px + r + 1)
        win = heat2d[y0:y1, x0:x1]
        ys, xs = np.nonzero(win > self.score_th)
        if xs.size == 0:  # 理论上不会发生（峰本身 > 阈值）
            return float(px), float(py), peak
        wgt = win[ys, xs].astype(np.float64)
        tot = float(wgt.sum())
        if tot <= 0:
            return float(px), float(py), peak
        cx = float(((xs + x0) * wgt).sum() / tot)
        cy = float(((ys + y0) * wgt).sum() / tot)
        return cx, cy, float(win.max())
