"""乒乓球落点检测：冗余观测融合 + 轨迹清洗 + 速度折角触台判定（官方参考实现的重写版）。

**参赛者代码区**——本文件实现 run.py 调用的四个生命周期方法
（``prepare / reset / process_frame / finish_video``）。

流水线（相对参考实现的差异）：
  1. ``prepare``：TensorRT 从 ``weights/ball.onnx`` 现场构建（缓存到
     ``<output>/trt_cache``）；若 TRT 不可用则自动回退 onnxruntime（CUDA→CPU），
     避免 engine 构建失败直接导致整场 0 分。
  2. ``process_frame``：逐帧检测球心，按"真实所属帧"存进缓冲，本帧先不输出。
  3. ``finish_video``：对整段轨迹做一次性分析——融合同帧的冗余通道观测 →
     剔除离群点 → 用前后速度矢量的**折角**（Δv_y 负尖峰 = 竖直速度被瞬间反向）
     判定触台帧 → 按视角参数 NMS 去重。框架明确允许延迟输出，用完整轨迹信息
     判定落点远比逐帧比较 y 稳（俯视角下触台后 y 会继续增大，极值法失效）。

参数在 ``configs/landing.json``（按 0/45/90 度分别配置），缺失则用内置默认值。
本地评分：``python3 participant/dev/eval_f1.py --help``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from core.types import Landing, RunContext, VideoMeta

from .ball_detector import IN_H, IN_W, BallDetector
from .base_solution import BaseSolution
from .landing import (clean_track, find_landings, fuse_obs, keep_in_roi,
                      rally_filter, table_roi)

_HERE = Path(__file__).resolve().parent

# ── 内置默认参数（configs/landing.json 会覆盖其中同名字段）─────────────────
_DEFAULT_CONFIG = {
    "detector": {
        "score_th": 0.55,      # 热图峰值阈值（低于此值不产出观测）
        "conf_gate": 0.60,     # 轨迹点置信度门槛（低于此值不参与拟合）
        "channels": [1],       # 使用哪些输出通道（1=中间帧，与参考实现一致；
                               # 模型三通道若各对应一帧，可改 [0,1,2] 提高召回）
        "peak_radius": 14,     # 峰值邻域半径（288×512 尺度）
    },
    "landing": {
        # 触台 = 球心速度矢量的折角：Δv_y 为负（下落被瞬间抬升）+ Δv_x 变化受限。
        # k/span/max_step 见 landing.kink_candidates 的说明。
        "default": {"dv_y": -10.0, "max_dvx_ratio": 0.5, "min_gap": 5.0,
                    "nms": "maxy", "k": 2, "span": 4.0, "max_step": 1.0,
                    "max_resid": 30.0, "max_gap": 6},
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    """读取 configs/landing.json 并覆盖内置默认值；任何异常都退回默认值。"""
    cfg = _DEFAULT_CONFIG
    path = _HERE / "configs" / "landing.json"
    try:
        if path.exists():
            cfg = _deep_merge(cfg, json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:  # 配置坏了不能拖垮整场评测
        print(f"[solution] 读取 {path} 失败，使用内置默认参数：{type(e).__name__}: {e}")
    return cfg


def _onnx_session(onnx_path: Path, decoder: str):
    """建 onnxruntime session：既用于读 ONNX 输入签名，也作为 TRT 不可用时的回退。"""
    import onnxruntime as ort

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if decoder == "gpu" else ["CPUExecutionProvider"])
    return ort.InferenceSession(str(onnx_path), providers=providers)


def _build_infer(onnx_path: Path, ctx: RunContext):
    """返回 (infer_fn, backend_name)。优先 TensorRT，失败回退 onnxruntime。"""
    try:
        # 多卡机：PP_GPU_INDEX 指定推理卡（与 NVDEC 抢同一张卡时可换卡）；越界自动回退
        from .pp_fast import select_device

        import torch
        select_device(torch, os.environ.get("PP_GPU_INDEX", "").strip() or None)
    except Exception as e:  # 选卡失败不能影响推理
        print(f"[solution] PP_GPU_INDEX 处理跳过（{type(e).__name__}: {e}）")

    sess = None
    in_name, shape = "input", (1, 9, IN_H, IN_W)
    try:
        # TRT 的 optimization profile 必须用 ONNX 里的真实输入名（见 core/trt.py
        # 的 profile.set_shape），写错会直接构建失败，所以从模型签名里读。
        sess = _onnx_session(onnx_path, ctx.decoder)
        sig = sess.get_inputs()[0]
        in_name = sig.name
        dims = [int(d) if isinstance(d, int) else 0 for d in sig.shape]
        if all(d > 0 for d in dims):
            shape = tuple(dims)
        print(f"[solution] ONNX 输入 {in_name}{shape}")
    except Exception as e:
        print(f"[solution] 读取 ONNX 输入签名失败（{type(e).__name__}: {e}），"
              f"按 {in_name}{shape} 处理")

    try:
        from core.trt import load_or_build

        model, how = load_or_build(onnx_path, Path(ctx.output_dir) / "trt_cache",
                                   input_name=in_name, shape=shape, log=print)
        try:
            return _AsyncTrt(model, shape), f"tensorrt_{how}_async"
        except Exception as e:  # 封装失败不能影响正确性，退回同步
            print(f"[solution] 异步推理不可用（{type(e).__name__}: {e}），退回同步")
            return model.infer, f"tensorrt_{how}"
    except Exception as e:
        print(f"[solution] TensorRT 不可用（{type(e).__name__}: {e}），回退 onnxruntime")

    if sess is None:
        import onnxruntime as ort

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if ctx.decoder == "gpu" else ["CPUExecutionProvider"])
        sess = ort.InferenceSession(str(onnx_path), providers=providers)

    def infer(x):
        return sess.run(None, {in_name: np.ascontiguousarray(x)})[0]

    return infer, f"onnxruntime_{sess.get_providers()[0]}"


class _AsyncTrt:
    """``core.trt.TrtModel`` 的异步封装：submit 只入队，finish 再取结果。

    框架的循环是 ``decode → process_frame``；单帧前向（实测 ≈1.6ms GPU）原本整段
    串行在 ``process_frame`` 里。拆成「本帧提交、下帧取回」后，前向与下一帧的
    NVDEC 解码重叠，``solution_sec`` 里的等待时间被解码时间吃掉。

    单帧里真正贵的不是 GPU，而是**入队**（评测在 proot 下跑，CUDA ioctl 被 ptrace
    拦截，TRT 逐 kernel 入队要 1.6ms；``out.cpu()`` 走 pageable 中转要 1.1ms）。
    因此这里把整帧做成一张 CUDA Graph 一次回放：

        H2D(pinned → GPU) → TRT kernel → D2H(GPU → pinned)

    * 回放实测 0.144ms（对比逐帧入队 1.765ms），且 D2H 落在框架解码的空档里；
    * 取回时 ``finish`` 只需要一次 stream 同步 + 主机端 numpy 解码，
      解码逻辑与同步路径**完全同一份代码**，结果逐位相同（捕获后会自动校验一次，
      校验会换一份输入再回放，确认图读的是当前 pinned 缓冲而不是捕获时的数据）；
    * 回放统一在 ``_stream`` 上发起，``finish`` 的同步范围才覆盖得到它；
    * 输入 / 输出 / 绑定表全部预分配，BallDetector 直接写 pinned 缓冲，
      省掉每帧一次 5.5MB 的中间拷贝。

    Graph 捕获失败（驱动不支持、引擎里有不可捕获的节点等）时自动退回逐帧入队，
    只是慢一点，行为不变。
    """

    def __init__(self, model, shape):
        self._m = model
        self._torch = torch = model._torch
        self._stream = model._stream
        self._shape = tuple(int(v) for v in shape)
        self._out = torch.empty(model.output_shape, dtype=torch.float32, device="cuda")
        self._in = torch.empty(self._shape, dtype=torch.float32, device="cuda")
        self._pin = torch.empty(self._shape, dtype=torch.float32, pin_memory=True)
        self._pin_np = self._pin.numpy()
        self._host = torch.empty(model.output_shape, dtype=torch.float32,
                                 pin_memory=True)
        self._host_np = self._host.numpy()
        self._bindings = [0] * model.engine.num_bindings
        self._bindings[model._input_idx] = int(self._in.data_ptr())
        self._bindings[model._output_idx] = int(self._out.data_ptr())
        self._graph = None
        self._busy = False

    @property
    def input_buffer(self):
        """预处理可直接写入的 pinned 缓冲（BallDetector 复用它，省一次拷贝）。"""
        return self._pin_np

    def _body(self):
        """一帧全部 GPU 操作（H2D → kernel → D2H），捕获与回放共用。"""
        with self._torch.cuda.stream(self._stream):
            self._in.copy_(self._pin, non_blocking=True)
            ok = self._m.context.execute_async_v2(self._bindings,
                                                  self._stream.cuda_stream)
            self._host.copy_(self._out, non_blocking=True)
        if not ok:
            raise RuntimeError("execute_async_v2 失败")

    def capture(self, warm: int = 3) -> bool:
        """把 ``_body`` 捕获成 CUDA Graph；成功返回 True，失败保持原路径。"""
        if self._graph is not None:
            return True
        try:
            for _ in range(warm):
                self._body()
            self._stream.synchronize()
            ref = self._host_np.copy()          # 逐帧入队的结果，用于校验
            graph = self._torch.cuda.CUDAGraph()
            with self._torch.cuda.graph(graph, stream=self._stream):
                self._body()
            self._stream.synchronize()

            def replay():
                with self._torch.cuda.stream(self._stream):
                    graph.replay()
                self._stream.synchronize()

            replay()
            diff = float(np.abs(self._host_np - ref).max())
            # 换一份输入再比一次：确认回放读的是当前 pinned 缓冲（而不是把捕获
            # 时的输入写死在图里），顺带验证 finish 的同步确实覆盖到回放。
            keep = self._pin_np.copy()
            self._pin_np += 1.0
            self._body()
            self._stream.synchronize()
            ref2 = self._host_np.copy()
            replay()
            diff2 = float(np.abs(self._host_np - ref2).max())
            self._pin_np[...] = keep
            self._body()
            self._stream.synchronize()
            if not (diff < 1e-5 and diff2 < 1e-5):
                print(f"[solution] CUDA Graph 输出不一致（max|Δ|={diff:.3e}，"
                      f"换输入后 {diff2:.3e}），不用它")
                return False
            self._graph = graph
            return True
        except Exception as e:
            print(f"[solution] CUDA Graph 不可用（{type(e).__name__}: {e}），逐帧入队")
            self._graph = None
            return False

    def submit(self, x_np):
        if x_np is not self._pin_np and not np.shares_memory(x_np, self._pin_np):
            np.copyto(self._pin_np, x_np)
        # 回放必须落在 self._stream 上：``finish`` 只同步这条流。若让 replay 走
        # 调用者的当前流（默认流），finish 会在 D2H 落地前就返回上一帧的结果——
        # 同步路径（submit 后立刻 finish）实测会整体晚 1 帧，这里不能省。
        with self._torch.cuda.stream(self._stream):
            if self._graph is not None:
                self._graph.replay()
            else:
                self._body()
        self._busy = True

    def finish(self):
        """同步并返回完整热图（1,3,H,W）——pinned 主机内存的视图，不额外拷贝。"""
        self._stream.synchronize()
        self._busy = False
        return self._host_np

    def __call__(self, x_np):
        """同步语义（warmup / 回退路径）：返回完整热图，与原 TrtModel.infer 一致。"""
        self.submit(x_np)
        return self.finish()


class Solution(BaseSolution):
    """轨迹建模 + 抛物线顶点拟合的落点判定。"""

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def prepare(self, context: RunContext) -> None:
        onnx_path = _HERE / "weights" / "ball.onnx"
        assert onnx_path.exists(), f"缺少权重: {onnx_path}"

        infer_fn, backend = _build_infer(onnx_path, context)
        self.cfg = load_config()
        self._pipe = (hasattr(infer_fn, "submit") and hasattr(infer_fn, "finish")
                      and os.environ.get("PP_ASYNC", "1") != "0")
        # 只把 BallDetector 认识的参数透传给它（conf_gate 是上层轨迹参数）
        det_cfg = {k: v for k, v in self.cfg["detector"].items()
                   if k in ("score_th", "channels", "peak_radius")}
        self.det = BallDetector(infer_fn, onnx_path, **det_cfg)

        # warmup：把 TRT kernel / ORT 内存池预热，避免首帧开销算进推理计时
        self.det.detect(np.zeros((IN_H, IN_W, 3), np.uint8), IN_W, IN_H)
        # 预热后把整帧（H2D+kernel+D2H）捕获成 CUDA Graph：回放 0.14ms vs 逐帧入队 1.8ms
        cap = getattr(infer_fn, "capture", None)
        if callable(cap):
            context.info["cuda_graph"] = bool(cap())

        context.info["backend"] = backend
        context.info["channels"] = list(self.det.channels)
        context.info["score_th"] = self.det.score_th

        # 开发用：PP_DUMP_DET=1 时把逐帧观测落盘，供 dev/tune.py 离线调参
        self._dump_dir = (Path(context.output_dir) / "detections"
                          if os.environ.get("PP_DUMP_DET") else None)

    def reset(self, video_meta: VideoMeta) -> None:
        self.det.reset()
        self.angle = video_meta.angle
        self.n_frames = int(video_meta.n_frames)
        self.width = int(video_meta.width)
        self.height = int(video_meta.height)
        self.video_id = video_meta.video_id
        self.obs: dict[int, list[tuple[float, float, float]]] = {}

    def process_frame(self, frame_rgb, frame_meta) -> list[Landing]:
        fid = int(frame_meta["frame_id"])
        if self._pipe:
            # 流水线：提交本帧推理，取回上一帧的观测（前向与解码重叠）
            obs = self.det.push(frame_rgb, fid,
                                frame_meta["width"], frame_meta["height"])
        else:
            obs = [(fid + off, x, y, conf) for off, x, y, conf in
                   self.det.detect(frame_rgb, frame_meta["width"],
                                   frame_meta["height"])]
        for key, x, y, conf in obs:
            if key >= 0:
                self.obs.setdefault(key, []).append((x, y, conf))
        return []  # 延迟到 finish_video 统一分析（框架允许）

    def finish_video(self, video_meta: VideoMeta) -> list[Landing]:
        try:
            if self._pipe:                      # 取回最后一帧的推理结果
                for key, x, y, conf in self.det.flush():
                    if key >= 0:
                        self.obs.setdefault(key, []).append((x, y, conf))
            return self._analyze()
        except Exception as e:  # 单视频失败不能崩掉整场
            print(f"[solution] {video_meta.video_id} 落点分析失败："
                  f"{type(e).__name__}: {e}")
            return []

    # ── 内部 ────────────────────────────────────────────────────────────────
    def _analyze(self) -> list[Landing]:
        fused = fuse_obs(self.obs)
        points = sorted((f, x, y, c) for f, (x, y, c) in fused.items())
        if len(points) < 3:
            return []
        if self._dump_dir is not None:
            # 落盘"原始融合观测"（未做 conf 门控/清洗），dev/tune.py 才能重放全流程
            self._dump(points, [])

        det_cfg = self.cfg["detector"]
        conf_gate = float(det_cfg.get("conf_gate", 0.6))
        gated = [p for p in points if p[3] >= conf_gate]
        if len(gated) >= 3:
            points = gated

        land_cfg = self.cfg["landing"]
        cfg = _deep_merge(land_cfg.get("default", {}),
                          land_cfg.get(self.angle, {}))
        points = clean_track(points,
                             max_resid=float(cfg.get("max_resid", 30.0)),
                             max_gap=int(cfg.get("max_gap", 6)),
                             mode=str(cfg.get("clean_mode", "chord")))

        raw = find_landings(points, cfg)

        # 球台投影区门控：触台只可能发生在台面上。台外的"折角"是球已离台后在地面/
        # 挡板上的真实反弹，Δv_y 信号与台内触台完全一致，只能靠落点位置判掉。
        # ROI 是归一化机位常量（见 landing.table_roi / keep_in_roi 的 fail-open）。
        rois = table_roi(cfg.get("roi"), self.width, self.height)
        raw = keep_in_roi(raw, rois,
                          min_keep=float(cfg.get("roi_min_keep", 0.5)))

        # 回合状态机（drlugu 24c2fce）：抑制回合后弹跳/桌边反弹/二次弹跳/90度同侧连续。
        raw = rally_filter(points, raw, rois, angle=self.angle)

        out: list[Landing] = []
        for fid, x, y, _score in raw:
            if not (0 <= fid < self.n_frames):
                continue
            # 框架不会自动 clip，越界直接判非法 → 自己夹紧到合法半开区间
            cx = min(max(float(x), 0.0), self.width - 1e-3)
            cy = min(max(float(y), 0.0), self.height - 1e-3)
            out.append(Landing(frame_id=fid, x=cx, y=cy))
        return out

    def _dump(self, points, landings) -> None:
        try:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            doc = {
                "video_id": self.video_id, "angle": self.angle,
                "n_frames": self.n_frames, "width": self.width,
                "height": self.height,
                "points": [[float(p[0]), float(p[1]), float(p[2]), float(p[3])]
                           for p in points],
                "landings": [[int(l[0]), float(l[1]), float(l[2])]
                             for l in landings],
            }
            (self._dump_dir / f"{self.video_id}.json").write_text(
                json.dumps(doc), encoding="utf-8")
        except Exception as e:
            print(f"[solution] dump {self.video_id} 失败：{e}")

