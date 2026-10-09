"""快速推理管线（仅属于 participant/ 可改区；不依赖也不改动 core/ 与参考实现语义）。

目的：在**与参考实现逐字节等价**的前提下，消除 process_frame 的每帧重复开销，
并输出各阶段耗时画像，用于定位「推理耗时分（25 分）」的真实瓶颈。

精度不改的四个对齐点（改代码时勿破坏）：
  1. cv2.resize(frame_rgb, (512, 288), INTER_LINEAR) —— 与 BallDetector._normalize 完全同一调用
  2. 归一化 /255 -> -ImageNet mean -> /ImageNet std，全程 float32（就地运算，逐元素与参考同算式）
  3. 3 帧滑窗顺序 [old, mid, new] 对应通道组 [0:3] / [3:6] / [6:9]，等价于参考的 concat(buf[-3:])
  4. 解码取 ch1（中间帧，固有滞后 1 帧），阈值 0.7 blob 加权质心，sx=W/512, sy=H/288

速度影响（正收益，量级取决于环境，见 run.log 的 [pp_profile] 行）：
  * 预处理零分配：resize 直接写入持久 dst，归一化就地，滑窗用 3 个持久 slot 组装，
    不再每帧 np.concatenate / astype / ascontiguousarray 各分配一块大数组
  * 推理零设备分配：输入/输出 CUDA 缓冲持久化，只把需要的 ch1 热图 (288x512) D2H 回主机，
    而不是每帧新建 device 张量并拷回整块 (1,3,288,512)
  * 结果：单帧少 4~5 次大数组分配 + 少拷 1.18MB/帧；纯 CPU/内存开销，
    不改变 GPU 上实际算的算子，因此精度与参考完全一致
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .ball_detector import IN_H, IN_W, OUT_CHANNEL, SCORE_TH
from .ball_detector import _MEAN as MEAN
from .ball_detector import _STD as STD

__all__ = ["StageTimer", "FrameTiler", "FastTrt", "RefTrt",
           "gpu_index_from_env", "select_device", "decode_heat"]

# 归因用途的开关（默认全部走最快路径，不设即最优）：
#   PP_NO_DST=1     强制不用 cv2.resize(dst=)，退回安全写法（自检失败时也会自动退回）
FORCE_NO_DST = os.environ.get("PP_NO_DST", "").strip() not in ("", "0")


# ── 耗时画像 ────────────────────────────────────────────────────────────────
class StageTimer:
    """累计各阶段耗时；report() 同时 print 与追加写 <output_dir>/run.log。

    run.sh 不会把 stdout 重定向进 run.log，所以这里主动追加到 run.log，
    保证「调试时优先看 run.log」这条路径能看到分阶段数据。
    """

    __slots__ = ("acc", "n", "log_path", "video_id")

    def __init__(self, output_dir, enabled: bool = True):
        self.acc: dict[str, list] = {}
        self.n = 0
        self.video_id = ""
        self.log_path = None
        if enabled and output_dir:
            try:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                self.log_path = out / "run.log"
            except OSError:
                self.log_path = None

    def reset(self, video_id: str = "") -> None:
        self.acc.clear()
        self.n = 0
        self.video_id = video_id

    def add(self, stage: str, dt: float) -> None:
        a = self.acc.get(stage)
        if a is None:
            self.acc[stage] = [dt, 1]
        else:
            a[0] += dt
            a[1] += 1

    def report(self, extra: str = "") -> None:
        if self.n <= 0 or not self.acc:
            return
        parts = []
        for stage, (tot, cnt) in sorted(self.acc.items(), key=lambda kv: -kv[1][0]):
            parts.append(f"{stage}={tot * 1e3:.1f}ms(total)/"
                         f"{tot / max(cnt, 1) * 1e3:.3f}ms(avg)")
        line = (f"[pp_profile] {self.video_id} frames={self.n} "
                f"{' '.join(parts)}{(' ' + extra) if extra else ''}")
        print(line, flush=True)
        if self.log_path is not None:
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


# ── 预处理：3 帧滑窗 -> (1,9,288,512) ───────────────────────────────────────
class FrameTiler:
    """预分配 + 就地运算的 3 帧滑窗输入构造器。语义与 BallDetector 完全一致。"""

    def __init__(self, in_h: int = IN_H, in_w: int = IN_W):
        self.in_h = int(in_h)
        self.in_w = int(in_w)
        self._u8 = np.empty((self.in_h, self.in_w, 3), np.uint8)       # resize dst
        self._f32 = np.empty((self.in_h, self.in_w, 3), np.float32)    # 归一化工作区
        self.inp = np.empty((1, 9, self.in_h, self.in_w), np.float32)  # 模型输入
        self._slots = [self.inp[0, 3 * i:3 * i + 3] for i in range(3)]
        # cv2.resize(dst=...) 复用缓冲：启动期自检一次，异常/版本差异则自动退回安全写法
        # （安全写法仍比参考少分配——只多一个 resize 输出缓冲）
        self._dst_ok = (not FORCE_NO_DST) and self._check_dst()
        # 与参考实现一致：blank 是对「288x512 的全零图」做归一化（resize 为恒等）
        self._blank = self._normalize_np(np.zeros((self.in_h, self.in_w, 3), np.uint8))
        self._n = 0                    # 已 push 的真实帧数
        self.reset()

    def _check_dst(self) -> bool:
        """验证 cv2.resize 传 dst= 与不传时结果逐位相同（防 OpenCV 版本/绑定差异）。

        自检必须永不抛异常：任何异常都只意味着「放弃 dst= 快路径」，不能影响 prepare。
        """
        rng = np.random.default_rng(0)
        src = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)
        dst = np.empty((5, 13, 3), np.uint8)
        try:
            want = cv2.resize(src, (13, 5), interpolation=cv2.INTER_LINEAR)
            cv2.resize(src, (13, 5), dst=dst, interpolation=cv2.INTER_LINEAR)
        except Exception as exc:  # noqa: BLE001
            print(f"[pp_fast] cv2.resize(dst=) 自检失败，改用安全写法: {exc}", flush=True)
            return False
        if not np.array_equal(np.asarray(want), dst):
            print("[pp_fast] cv2.resize(dst=) 结果与不传 dst 时不同，改用安全写法",
                  flush=True)
            return False
        return True

    def _resize_into(self, frame_rgb: np.ndarray) -> None:
        """resize 进持久缓冲 self._u8（数值与参考实现的 cv2.resize 调用完全相同）。"""
        if self._dst_ok:
            cv2.resize(frame_rgb, (self.in_w, self.in_h), dst=self._u8,
                       interpolation=cv2.INTER_LINEAR)
        else:
            np.copyto(self._u8, cv2.resize(frame_rgb, (self.in_w, self.in_h),
                                           interpolation=cv2.INTER_LINEAR))

    def _normalize_np(self, frame_rgb: np.ndarray) -> np.ndarray:
        """返回 (3,H,W) 连续 float32（仅用于构造 blank 常量）。"""
        self._resize_into(frame_rgb)
        np.copyto(self._f32, self._u8)
        self._f32 /= 255.0
        self._f32 -= MEAN
        self._f32 /= STD
        return np.ascontiguousarray(self._f32.transpose(2, 0, 1)).copy()

    def reset(self) -> None:
        for s in self._slots:
            s[:] = self._blank
        self._n = 0

    def push(self, frame_rgb: np.ndarray) -> np.ndarray:
        """写入当前帧，返回可直接喂模型的 (1,9,288,512) 缓冲（内部复用，勿保存）。

        通道语义严格对齐参考实现 BallDetector 的 `concat(buf[-3:])`：
          buf 初始为 [blank, blank]（预填 2 帧），每帧 append 当前帧。
          设 n = 已 push 的真实帧数（本帧计入后），则
            buf[-3] = 帧(n-3)（n<3 时仍是 blank） → 通道组 [0:3]
            buf[-2] = 帧(n-2)（n<2 时仍是 blank） → 通道组 [3:6]
            buf[-1] = 帧(n-1) = 当前帧           → 通道组 [6:9]
          即：本帧永远落在 [6:9]，历史帧依次落在 [3:6] / [0:3]。
        说明：用「轮转指针 + 按序组装」写也能得到同样的通道布局（两条路都已通过
        等价性测试）；这里保留显式搬移是因为少一次取模、时序更直观。无论怎么改，
        动完**必须重跑 `_check_tiler_order.py`**（通道顺序差分测试），
        否则精度等价性无保障。
        """
        self._resize_into(frame_rgb)
        np.copyto(self._f32, self._u8)
        self._f32 /= 255.0
        self._f32 -= MEAN
        self._f32 /= STD

        if self._n >= 2:            # 帧(n-3) = 原 slots[1] 移到组[0:3]
            np.copyto(self._slots[0], self._slots[1])
        if self._n >= 1:            # 帧(n-2) = 原 slots[2] 移到组[3:6]
            np.copyto(self._slots[1], self._slots[2])
        np.copyto(self._slots[2], self._f32.transpose(2, 0, 1))
        self._n += 1
        return self.inp


# ── 推理：持久化缓冲的 TensorRT 封装 ────────────────────────────────────────
class FastTrt:
    """在 core.trt.load_or_build 之上做零分配 + 锁页搬运推理，只回传所需通道的热图。

    与 core.trt.TrtModel 的差异（都是「怎么搬」，不涉及「算什么」）：
      * 输入/输出 device 缓冲持久化（不再每帧 torch.empty）
      * 主机侧用**锁页内存**（pinned）：可分页内存的 H2D/D2H 在 CUDA 里要走一段
        驱动内部 staging 缓冲（页错误 + 额外 memcpy），本模型每帧 H2D 达 5.3MB
        （1x9x288x512 fp32），这段开销不可忽略
      * 只把 ch=out_channel 的 (288,512) 热图 D2H 到持久锁页缓冲（少拷 1.18MB/帧）
      * copy_ 加 non_blocking=True + 显式 stream.synchronize()，语义与阻塞拷贝相同
        （同样在返回前确保数据可用），但去掉了隐式同步

    数值等价性：以上只改变主机缓冲的驻留方式与拷贝时机，不改变任何被计算的数值，
    因此与参考实现仍逐位等价。锁页分配失败（受限环境）时自动退回可分页路径。
    """

    def __init__(self, onnx_path, trt_cache, input_name: str = "input",
                 shape=(1, 9, IN_H, IN_W), fp16: bool = True,
                 out_channel: int = OUT_CHANNEL, device=None, log=print):
        import torch
        from core.trt import load_or_build

        self._torch = torch
        if device is not None:
            torch.cuda.set_device(select_device(torch, device))
        self.device = int(torch.cuda.current_device())

        self.model, self.how = load_or_build(
            onnx_path, trt_cache, input_name=input_name, shape=tuple(shape),
            fp16=fp16, log=log)

        eng = self.model.engine
        in_idx = out_idx = None
        for i in range(eng.num_bindings):
            if eng.binding_is_input(i):
                in_idx = i
            elif out_idx is None:
                out_idx = i
        if in_idx is None or out_idx is None:
            raise RuntimeError("engine 找不到输入/输出 binding")

        self._ctx = self.model.context
        self._in_idx = in_idx
        self._out_idx = out_idx
        self.input_shape = tuple(eng.get_binding_shape(in_idx))
        self.output_shape = tuple(eng.get_binding_shape(out_idx))
        self.out_channel = int(out_channel)
        if self.input_shape != tuple(shape):
            # 动态/非预期 shape：交给 RefTrt（它会 set_input_shape），避免逐帧崩溃
            raise RuntimeError(f"engine 输入 shape {self.input_shape} != 期望 {tuple(shape)}")
        if len(self.output_shape) != 4:
            raise RuntimeError(f"仅支持 (N,C,H,W) 输出，实际 {self.output_shape}")

        # 按 engine 的真实精度建缓冲。参考实现硬编码 float32，若 ONNX 输出 fp16
        # 就会在 copy_ 上直接炸；这里读一次 binding dtype，不一致则退回 RefTrt
        # （比逐帧报错好，且不改变任何数值）。
        dtypes = self._binding_dtypes(eng)
        if dtypes != (torch.float32, torch.float32):
            raise RuntimeError(
                f"engine binding dtype {dtypes} 非 (float32, float32)，改用 RefTrt")

        self._x = torch.empty(self.input_shape, dtype=torch.float32, device="cuda")
        self._y = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")
        self._bindings = [0] * eng.num_bindings
        self._bindings[in_idx] = int(self._x.data_ptr())
        self._bindings[out_idx] = int(self._y.data_ptr())
        self._stream = torch.cuda.Stream()

        # 热图宿主缓冲（ch=out_channel，形状 (H,W)）
        self._heat_host = np.empty(self.output_shape[2:], dtype=np.float32)
        # 锁页缓冲区：分配失败只是一次异常，退回可分页路径即可，不能让 prepare 挂掉
        self._pinned = False
        try:
            self._x_host_t = torch.empty(self.input_shape, dtype=torch.float32,
                                         pin_memory=True)
            self._heat_host_t = torch.empty(tuple(self.output_shape[2:]), dtype=torch.float32,
                                            pin_memory=True)
            self._heat_host = self._heat_host_t.numpy()
            self._pinned = True
        except Exception as exc:  # noqa: BLE001
            log(f"[pp_fast] 锁页内存不可用，退回可分页拷贝（不影响精度）: {exc}")
            self._heat_host_t = torch.from_numpy(self._heat_host)

    @staticmethod
    def _binding_dtypes(eng):
        """读 (输入, 输出) 的 torch dtype。读不到就保守按 float32 返回。

        TRT 各版本 API 不同：新版 get_tensor_dtype + TensorIOMode，
        老版 get_binding_dtype + binding_is_input。两边都试，失败即放弃判断。
        """
        import torch
        trt2torch = {
            "FLOAT": torch.float32, "HALF": torch.float16,
            "BF16": torch.bfloat16, "INT32": torch.int32, "INT8": torch.int8,
        }

        def _name(dt):
            v = getattr(dt, "name", None)
            return v if isinstance(v, str) else str(dt).split(".")[-1]

        try:
            names = []
            for i in range(eng.num_bindings):
                dt = (eng.get_binding_dtype(i) if hasattr(eng, "get_binding_dtype")
                      else eng.get_tensor_dtype(eng.get_binding_name(i)))
                names.append(_name(dt))
            if len(names) >= 2:
                return (trt2torch.get(names[0].upper(), torch.float32),
                        trt2torch.get(names[1].upper(), torch.float32))
        except Exception:  # noqa: BLE001
            pass
        return (torch.float32, torch.float32)

    def infer(self, x_np: np.ndarray) -> np.ndarray:
        """x_np: (1,9,288,512) float32 C 连续 -> ch=out_channel 的 (288,512) 热图。"""
        torch = self._torch
        nb = self._pinned          # 只有锁页内存才值得开 non_blocking
        x_t = torch.from_numpy(x_np)
        if self._pinned:
            # 先落到锁页缓冲，再让 H2D 成为真·异步 DMA（避免从可分页内存发起）
            self._x_host_t.copy_(x_t)
            self._x.copy_(self._x_host_t, non_blocking=nb)
        else:
            self._x.copy_(x_t)
        ok = self._ctx.execute_async_v2(self._bindings, self._stream.cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v2 失败")
        self._stream.synchronize()          # 等 GPU 算完，再做 D2H（跨 stream 安全）
        self._heat_host_t.copy_(self._y[0, self.out_channel], non_blocking=nb)
        if nb:
            self._stream.synchronize()      # 确保 D2H 落地后再把缓冲交出去
        return self._heat_host


class RefTrt:
    """回退通道：直接走 core.trt.TrtModel（与参考实现同路径，仅做薄适配）。"""

    def __init__(self, onnx_path, trt_cache, input_name: str = "input",
                 shape=(1, 9, IN_H, IN_W), fp16: bool = True,
                 out_channel: int = OUT_CHANNEL, device=None, log=print):
        import torch
        from core.trt import load_or_build

        if device is not None:
            torch.cuda.set_device(select_device(torch, device))
        self.model, self.how = load_or_build(
            onnx_path, trt_cache, input_name=input_name, shape=tuple(shape),
            fp16=fp16, log=log)
        self.out_channel = int(out_channel)
        self.device = int(torch.cuda.current_device())

    def infer(self, x_np: np.ndarray) -> np.ndarray:
        return self.model.infer(x_np)[0, self.out_channel]


def gpu_index_from_env() -> int | None:
    """PP_GPU_INDEX 指定推理用的物理卡（避免与 NVDEC 抢同一张卡）。未设置则 None。

    只做"能否解析成整数"的判断；**范围校验放在 select_device()**，因为这里拿不到 torch。
    """
    v = os.environ.get("PP_GPU_INDEX", "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def select_device(torch, device) -> int:
    """解析并**校验**要用的 GPU 序号，返回实际生效的序号。

    为什么必须校验（实测踩到）：
      * `torch.cuda.set_device(99)` 在 2 卡机上抛
        `AcceleratorError: CUDA error: invalid device ordinal`；
        这个异常会被 solution.py 的兜底捕获并回退 RefTrt，而 RefTrt 用同一个非法
        序号**再抛一次** -> 最终 prepare 失败 -> 整个提交 0 分。
      * `torch.cuda.set_device(-1)` **不会报错**，会静默绑到别的卡，
        让 PP_GPU_INDEX 的"卡隔离"意图失效。
    两者都不是我们想要的行为，所以越界一律回退到当前设备，并且不抛异常。
    """
    try:
        n = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 —— 取不到卡数就干脆不切卡
        n = 0
    cur = int(torch.cuda.current_device()) if n else 0
    if device is None:
        return cur
    try:
        idx = int(device)
    except (TypeError, ValueError):
        print(f"[pp_fast] PP_GPU_INDEX={device!r} 非法，沿用当前设备 {cur}", flush=True)
        return cur
    if n and 0 <= idx < n:
        return idx
    print(f"[pp_fast] PP_GPU_INDEX={idx} 越界（本机 {n} 张卡），回退设备 {cur}", flush=True)
    return cur


def decode_heat(heat: np.ndarray, orig_w: int, orig_h: int):
    """(288,512) 热图 -> (x, y, conf) 像素坐标；无响应返回 None。与参考同算式。

    相对参考的一处纯开销削减：不再先做 mask.any() 再 nonzero——那会把布尔数组
    完整扫两遍。nonzero 返回空索引即等价于「无响应」，判定结果完全相同。
    """
    ys, xs = np.nonzero(heat > SCORE_TH)
    if ys.size == 0:
        return None
    vals = heat[ys, xs]
    total = float(vals.astype(np.float64).sum())
    if total <= 0.0:
        return None
    cx = float((xs * vals).sum() / total)
    cy = float((ys * vals).sum() / total)
    return (cx * orig_w / float(IN_W), cy * orig_h / float(IN_H), float(vals.max()))
