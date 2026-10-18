"""球检测：热图模型 → 逐帧球心（官方示例的重写版，接口向后兼容）。

模型 weights/ball.onnx：
  input  (1, 9, 288, 512)   3 帧 RGB 栈（ImageNet mean/std 归一化）
  output (1, 3, 288, 512)   3 个时间位置的热图（已含 sigmoid）；
                            第 c 通道对应输入栈第 c 帧，栈按 [旧, 中, 新] 排布，
                            即处理第 f 帧时 ch0→f-2、ch1→f-1、ch2→f。

与旧实现的差异（旧实现只在整幅热图上求和，容易被第二个 blob 拉偏）：
  * 解码只统计峰值附近 ``peak_radius`` 内的像素（即球所在的那个连通域），
    不再对全图求和；
  * 阈值 / 使用通道 / 峰值半径全部可配置；
  * ``detect`` 返回 ``[(帧偏移, x, y, conf), ...]``（x/y 为原分辨率像素），
    让上层能把多通道的冗余观测合并到同一帧上。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IN_H, IN_W = 288, 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# blobFromImage 的减均值单位是 0~255，故把 ImageNet 均值乘回 255；
# 之后逐通道乘 1/(255·std) 即等价于 ((x/255 - mean) / std)（实测最大差 5e-7）。
_MEAN255 = tuple(float(v) * 255.0 for v in _MEAN)
_STD_SCALE = (1.0 / (255.0 * _STD)).astype(np.float32).reshape(1, 3, 1, 1)

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
        # 流水线：infer_fn 若带 submit/finish（异步 TRT），push/flush 走重叠路径
        self._async = hasattr(infer_fn, "submit") and hasattr(infer_fn, "finish")
        # 异步实现提供 pinned 输入缓冲时直接往里写，省掉每帧一次 5.5MB 拷贝
        self._inp = getattr(infer_fn, "input_buffer", None)
        if self._inp is None:
            self._inp = np.empty((1, 9, IN_H, IN_W), np.float32)
        self._sx = self._sy = 1.0
        self._pending_fid = None
        self.reset()

    # ── 预处理 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(frame_rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 -> (3,288,512) float32 归一化。

        用 ``cv2.dnn.blobFromImage`` 一次完成 resize + 转 float32 + 减均值，
        再逐通道原地乘 ``1/(255·std)``。与 ``(x/255 - mean)/std`` 数学等价
        （实测最大差 5e-7），但省掉 4 个全图临时数组：4.7 → 1.2 ms/帧。
        归一化曾是单帧最大开销（占方案侧 56%），是全流程 fps 的关键一环。
        """
        blob = cv2.dnn.blobFromImage(
            frame_rgb, 1.0, (IN_W, IN_H), _MEAN255,
            swapRB=False, crop=False, ddepth=cv2.CV_32F)
        blob *= _STD_SCALE                      # (1,3,1,1) 广播，原地
        return blob[0]                          # 已是连续内存，无需再拷贝

    def reset(self) -> None:
        # 预填 2 帧空白：处理第 f 帧时缓冲区为 [f-2, f-1, f]
        self._buf = [self._blank, self._blank]
        if getattr(self, "_async", False) and self._pending_fid is not None:
            try:                      # 丢弃在飞的结果，避免下一段视频复用缓冲时踩内存
                self._infer.finish()
            except Exception:
                pass
        self._pending_fid = None

    # ── 推理 + 解码 ─────────────────────────────────────────────────────────
    def _prep(self, frame_rgb: np.ndarray) -> np.ndarray:
        """3 帧栈 → (1,9,288,512)，写入预分配缓冲（不每帧新建 5MB 数组）。

        返回缓冲本体而不是切片视图：异步后端把 ``input_buffer`` 交给这里时，
        ``submit`` 就能靠对象同一性跳过那一份 5.5MB 的拷贝。
        """
        self._buf.append(self._normalize(frame_rgb))
        self._buf = self._buf[-3:]
        try:
            np.concatenate(self._buf, axis=0, out=self._inp[0])
            return self._inp
        except (TypeError, ValueError):          # 老 numpy 不支持 out=
            self._inp = np.concatenate(self._buf, axis=0)[None]
            return self._inp

    def _collect(self, fid: int) -> list[tuple[int, float, float, float]]:
        """取回一次已提交推理，解码成 (绝对帧号, x, y, conf)。"""
        heat = self._infer.finish()[0]              # (3,288,512)，pinned 主机缓冲
        out: list[tuple[int, float, float, float]] = []
        for ch in self.channels:
            got = self._decode_heat(heat[ch])
            if got is None:
                continue
            cx, cy, conf = got
            out.append((int(fid) + _CH_OFFSET[ch],
                        cx * self._sx, cy * self._sy, conf))
        return out

    def push(self, frame_rgb: np.ndarray, fid: int, orig_w: int, orig_h: int
             ) -> list[tuple[int, float, float, float]]:
        """流水线版 detect：提交本帧推理，返回**上一帧**的观测（绝对帧号）。

        上一帧的 kernel 在框架解码本帧（≈15ms）时已经跑完，因此 finish()
        几乎不阻塞；这正是把 2.5ms 前向从 solution_sec 里省掉的来源。
        """
        if not self._async:
            raise RuntimeError("push() 需要带 submit/finish 的异步推理器")
        self._sx, self._sy = orig_w / IN_W, orig_h / IN_H
        out = self._collect(self._pending_fid) if self._pending_fid is not None else []
        self._infer.submit(self._prep(frame_rgb))
        self._pending_fid = int(fid)
        return out

    def flush(self) -> list[tuple[int, float, float, float]]:
        """取回最后一帧（finish_video 调用）。"""
        out = []
        if self._async and self._pending_fid is not None:
            out = self._collect(self._pending_fid)
            self._pending_fid = None
        return out

    def detect(self, frame_rgb: np.ndarray, orig_w: int, orig_h: int
               ) -> list[tuple[int, float, float, float]]:
        """同步版：返回 ``[(帧偏移, x, y, conf), ...]``；帧偏移 = 相对当前帧（<=0）。"""
        self._sx, self._sy = orig_w / IN_W, orig_h / IN_H
        heat = self._infer(self._prep(frame_rgb))[0]   # (3,288,512)
        out: list[tuple[int, float, float, float]] = []
        for ch in self.channels:
            got = self._decode_heat(heat[ch])
            if got is None:
                continue
            cx, cy, conf = got
            out.append((_CH_OFFSET[ch], cx * self._sx, cy * self._sy, conf))
        return out

    def _decode_heat(self, heat2d: np.ndarray):
        """取峰值所在连通域的加权质心；峰值低于阈值返回 None。"""
        h, w = heat2d.shape
        idx = int(np.argmax(heat2d))
        py, px = divmod(idx, w)
        r = self.peak_radius
        y0, y1 = max(0, py - r), min(h, py + r + 1)
        x0, x1 = max(0, px - r), min(w, px + r + 1)
        return self._decode_win((y0, x0, heat2d[y0:y1, x0:x1]))

    def _decode_win(self, item):
        """``(y0, x0, 窗口)`` → (cx, cy, conf)；窗口必须包含该通道的全局峰值。

        阈值判定用窗口内最大值（= 全局峰值，两者同值）；质心只统计窗口内
        ``> score_th`` 的像素——与旧实现「整图上取 argmax 再取窗口」等价。
        """
        y0, x0, win = item
        peak = float(win.max())
        if peak < self.score_th:
            return None
        ys, xs = np.nonzero(win > self.score_th)
        if xs.size == 0:  # 理论上不会发生（峰本身 > 阈值）→ 退回峰值坐标
            iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
            return float(x0 + ix), float(y0 + iy), peak
        wgt = win[ys, xs].astype(np.float64)
        tot = float(wgt.sum())
        if tot <= 0:
            iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
            return float(x0 + ix), float(y0 + iy), peak
        cx = float(((xs + x0) * wgt).sum() / tot)
        cy = float(((ys + y0) * wgt).sum() / tot)
        return cx, cy, peak
