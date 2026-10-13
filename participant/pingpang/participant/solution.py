"""乒乓球落点检测：冗余观测融合 + 轨迹清洗 + 速度折角触台判定（参考实现的重写版）。

**参赛者代码区**——本文件实现 run.py 调用的四个生命周期方法
（``prepare / reset / process_frame / finish_video``）。

流水线（相对参考实现的差异）：
  1. ``prepare``：TensorRT 从 ``weights/ball.onnx`` 现场构建（缓存到临时目录，
    刻意不放 output_dir——engine 体积会被 run.py 的 os.walk 计入「输出物 <1GB」
    硬限制）；若 TRT 不可用则自动回退 onnxruntime（CUDA→CPU），
    避免 engine 构建失败直接导致整场 0 分。
  2. ``process_frame``：逐帧检测球心，按"真实所属帧"存进缓冲，本帧先不输出。
  3. ``finish_video``：对整段轨迹做一次性分析——融合同帧的冗余通道观测 →
    剔除离群点 → 用前后速度矢量的**折角**（Δv_y 负尖峰 = 竖直速度被瞬间反向）
    判定触台帧 → 按视角参数 NMS 去重。框架明确允许延迟输出，用完整轨迹信息
    判定落点远比逐帧比较 y 稳（俯视角下触台后 y 会继续增大，极值法失效，
    实测 0°/45° 的 y 极值法 F1 仅 0.05，折角法 0.79/0.82）。

参数在 ``configs/landing.json``（按 0/45/90 度分别配置），缺失则用内置默认值。
本地评分：``python3 tune_eval.py score --traj ... --input public_data``。
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
                               # 模型三通道各对应一帧，实际提交用 [0,1,2] 提高召回）
        "peak_radius": 14,     # 峰值邻域半径（288×512 尺度）
    },
    "landing": {
        "default": {"dv_y": -12.0, "max_dvx_ratio": 0.5, "min_gap": 5.0,
                    "nms": "maxy", "k": 2, "span": 4.0, "max_step": 1.0,
                    "max_resid": 30.0, "max_gap": 6, "roi_min_keep": 0.5},
        "0度": {"dv_y": -10.0, "roi": [0.4222, 0.274, 0.6014, 0.6429]},
        "45度": {"dv_y": -12.0, "roi": [0.2731, 0.3453, 0.7277, 0.617],
                 "dv_strong": 25.0, "ratio_strong": 0.7},
        "90度": {"dv_y": -12.0, "max_dvx_ratio": 1.0, "min_gap": 6.0,
                 "nms": "combo", "roi": [0.3417, 0.3246, 0.669, 0.7528]},
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


def _trt_cache_dir(output_dir) -> Path:
    """TRT engine 缓存目录——**刻意不放在 output_dir 里**。

    run.py 用 os.walk(output_dir) 递归求和，超 1GB 直接判 failed。engine 体积
    无法预估，放进输出目录等于给这条硬限制挂上不可控风险；且评测容器每次全新，
    缓存本来也命中不了。优先级：PP_TRT_CACHE > $TMPDIR/pp_trt_cache >
    /tmp/pp_trt_cache > output_dir（兜底）。
    """
    cands = []
    env = os.environ.get("PP_TRT_CACHE", "").strip()
    if env:
        cands.append(Path(env))
    tmp = os.environ.get("TMPDIR", "").strip()
    if tmp:
        cands.append(Path(tmp) / "pp_trt_cache")
    cands.append(Path("/tmp/pp_trt_cache"))
    for c in cands:
        try:
            c.mkdir(parents=True, exist_ok=True)
            return c
        except OSError:
            continue
    return Path(output_dir) / "trt_cache"


def _onnx_session(onnx_path: Path, decoder: str):
    """建 onnxruntime session：既用于读 ONNX 输入签名，也作为 TRT 不可用时的回退。"""
    import onnxruntime as ort

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if decoder == "gpu" else ["CPUExecutionProvider"])
    return ort.InferenceSession(str(onnx_path), providers=providers)


def _build_infer(onnx_path: Path, ctx: RunContext):
    """返回 (infer_fn, backend_name)。优先 TensorRT，失败回退 onnxruntime。"""
    from .pp_fast import select_device

    import torch
    select_device(torch, os.environ.get("PP_GPU_INDEX", "").strip() or None)

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

        model, how = load_or_build(onnx_path, _trt_cache_dir(ctx.output_dir),
                                   input_name=in_name, shape=shape, log=print)
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


class Solution(BaseSolution):
    """轨迹建模 + 速度折角判定的落点检测。"""

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def prepare(self, context: RunContext) -> None:
        onnx_path = _HERE / "weights" / "ball.onnx"
        assert onnx_path.exists(), f"缺少权重: {onnx_path}"

        infer_fn, backend = _build_infer(onnx_path, context)
        self.cfg = load_config()
        # 只把 BallDetector 认识的参数透传给它（conf_gate 是上层轨迹参数）
        det_cfg = {k: v for k, v in self.cfg["detector"].items()
                   if k in ("score_th", "channels", "peak_radius")}
        # 通道必须在 [0,1,2] 内，越界会索引热图报错 → 兜底回 [1]
        chs = tuple(c for c in det_cfg.get("channels", [1]) if c in (0, 1, 2)) or (1,)
        det_cfg["channels"] = chs
        self.det = BallDetector(infer_fn, onnx_path, **det_cfg)

        # warmup：把 TRT kernel / ORT 内存池预热，避免首帧开销算进推理计时
        self.det.detect(np.zeros((IN_H, IN_W, 3), np.uint8), IN_W, IN_H)

        context.info["backend"] = backend
        context.info["channels"] = list(self.det.channels)
        context.info["score_th"] = self.det.score_th

        # 开发用：PP_DUMP_DET=1 时把逐帧观测落盘，供离线调参
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
        for off, x, y, conf in self.det.detect(
                frame_rgb, frame_meta["width"], frame_meta["height"]):
            key = fid + off
            if key >= 0:
                self.obs.setdefault(key, []).append((x, y, conf))
        return []  # 延迟到 finish_video 统一分析（框架允许）

    def finish_video(self, video_meta: VideoMeta) -> list[Landing]:
        try:
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
            self._dump(points)

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
                             max_gap=int(cfg.get("max_gap", 6)))

        # 落点判定：速度折角判据（landing.py）→ 球台投影区门控（ROI）。
        raw = find_landings(points, cfg)
        roi = table_roi(cfg.get("roi"), self.width, self.height)
        raw = keep_in_roi(raw, roi, min_keep=float(cfg.get("roi_min_keep", 0.5)))
        # 回合状态机：抑制回合后/桌边反弹/二次弹跳等干扰落点（不改 Δv_y 判据）。
        raw = rally_filter(points, raw, roi, angle=self.angle)

        out: list[Landing] = []
        for fid, x, y, _score in raw:
            if not (0 <= fid < self.n_frames):
                continue
            # 框架不会自动 clip，越界直接判非法 → 自己夹紧到合法半开区间
            cx = min(max(float(x), 0.0), self.width - 1e-3)
            cy = min(max(float(y), 0.0), self.height - 1e-3)
            out.append(Landing(frame_id=fid, x=cx, y=cy))
        return out

    def _dump(self, points) -> None:
        try:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            doc = {
                "video_id": self.video_id, "angle": self.angle,
                "n_frames": self.n_frames, "width": self.width,
                "height": self.height,
                "points": [[float(p[0]), float(p[1]), float(p[2]), float(p[3])]
                           for p in points],
            }
            (self._dump_dir / f"{self.video_id}.json").write_text(
                json.dumps(doc), encoding="utf-8")
        except Exception as e:
            print(f"[solution] dump {self.video_id} 失败：{e}")
