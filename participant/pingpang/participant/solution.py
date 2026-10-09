"""参赛实现：落点检测 —— 与参考实现**逐字节等价**的精度 + 消除每帧重复开销的快速管线。

本文件相对官方参考实现（原 solution.py + ball_detector.py 组合）的**唯一**改动是
「怎么算」，「算什么」完全不变：

  不变（精度保证）：
    * 预处理仍是 cv2.resize(frame, (512,288), INTER_LINEAR) -> /255 -> ImageNet mean/std，
      全程 float32，逐元素同算式
    * 仍是 3 帧滑窗 (1,9,288,512)，通道组 [0:3]/[3:6]/[6:9] = [旧,中,新]
    * 仍解码 ch1（中间帧，固有滞后 1 帧），阈值 0.7 blob 加权质心，sx=W/512, sy=H/288
    * 落点启发式的判定条件与 4 个参数逐字未动——实现已搬到 emit_rule.py（唯一实现，
      提交管线 / 离线调参工具 / 等价性测试三方共用，避免参数各自漂移）
  改动（速度）：
    * 预处理零分配（resize 写持久 dst + 就地归一化 + 3 个持久 slot 组装滑窗），
      取代每帧 astype/减/除/transpose/ascontiguousarray/concat 的多次大数组分配
    * 推理零设备分配（持久 CUDA 输入输出），只 D2H 需要的 ch1 热图（少拷 1.18MB/帧）
    * _emit 由「每帧全量重扫 O(n^2)」改为「游标增量扫描 O(n) 总计」——
      已扫描过的 j 结论不会再变（局部极大只依赖 j-1/j/j+1；last_emit 单调不减，
      被 MIN_GAP 挡掉的 j 之后只会更被挡掉），故结果与全量重扫严格相同
    * 未做 ROI 裁剪：当前 ball.onnx 只在整帧 288x512 尺度下训练，裁剪会改变球的
      像素尺度/位置分布，属于「必须重训模型才能做」的改动，不与提速混在一起

速度影响：单帧省下 4~5 次大数组分配 + 1.18MB D2H 拷贝 + O(n) 重扫；每阶段实测耗时见
run.log 的 `[pp_profile]` 行（每视频一行，StageTimer 输出）。
精度影响：无（算术与判定逐元素等价，预测应与参考实现完全一致）。
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter as _now      # 单调时钟；4 次/帧，开销 ~0.3us

from core.types import Landing, RunContext, VideoMeta

from .base_solution import BaseSolution
from . import pp_fast
from .pp_fast import FrameTiler, RefTrt

# 落点判定的唯一实现在 emit_rule.py（7 个参数）：调参工具 tune_eval.py（在仓库根，
# 刻意放在 participant/ 之外——它读 gt_public.jsonl 真值，不应打进提交镜像）
# 与等价性测试 _check_emit_equiv.py 都 import 它，确保「扫出来的参数」与「线上跑的判定」
# 是同一份代码。默认值 = 参考实现原值（peak_win=1 / peak_conf=0 / smooth=0 即完全等价关闭）；
# 可用 PP_MIN_GAP/PP_CONF_TH/PP_FALL_AMP/PP_RISE_AMP/PP_PEAK_WIN/PP_PEAK_CONF/PP_SMOOTH
# 做 A/B（不设则行为不变）。实测结论见 emit_rule.py 文件头。
from .emit_rule import EmitRule, describe as describe_params  # noqa: E402

PROFILE = os.environ.get("PP_PROFILE", "1").strip() != "0"


def _trt_cache_dir(output_dir) -> Path:
    """TRT engine 缓存目录——**刻意不放在 output_dir 里**。

    赛道限定条件：输出物大小 < 1 GB；框架约定输出目录只含
    predictions.jsonl / run_status.json / run.log 三样。engine 体积无法预估，
    放进去等于给「输出物 < 1 GB」这条硬限制挂上一份不可控风险；
    而评测容器每次都是全新的、缓存本来也命中不了，放输出目录零收益。

    优先级：PP_TRT_CACHE > $TMPDIR/pp_trt_cache > /tmp/pp_trt_cache > output_dir（兜底）。
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


class Solution(BaseSolution):
    """落点启发式：球心 y 的局部极大值 + 下落/反弹门控（含快速管线 + 耗时画像）。"""

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def prepare(self, context: RunContext) -> None:
        # 1. 权重：只带 ONNX，TRT engine 现场构建（首跑约 5 分钟，之后从缓存秒级加载）
        onnx_path = Path(__file__).resolve().parent / "weights" / "ball.onnx"
        assert onnx_path.exists(), f"缺少权重: {onnx_path}"
        trt_cache = _trt_cache_dir(context.output_dir)   # 不放 output_dir，避免计入「输出物 <1GB」
        dev = pp_fast.gpu_index_from_env()

        # 2. 耗时画像（默认开；PP_PROFILE=0 关闭）：每视频结束写一行到 run.log
        self._timer = pp_fast.StageTimer(context.output_dir, enabled=PROFILE)
        self._tiler = FrameTiler()   # 预处理缓冲必须先于 warmup 建好

        # 3. TRT：优先零分配的 FastTrt；失败则回退 core.trt.TrtModel（语义相同）
        try:
            infer = pp_fast.FastTrt(onnx_path, trt_cache, device=dev, log=print)
            backend = f"tensorrt_{infer.how}_fast"
        except Exception as exc:  # noqa: BLE001 —— 任何构建/绑定异常都退回参考路径
            print(f"[pp_fast] FastTrt 不可用，回退 core.trt.TrtModel: {exc}", flush=True)
            infer = RefTrt(onnx_path, trt_cache, device=dev, log=print)
            backend = f"tensorrt_{infer.how}_ref"
        self._infer = infer.infer

        # 4. warmup：TRT 首次推理会触发 kernel JIT，先跑热，避免算进 infer_sec
        for _ in range(2):
            self._infer(self._tiler.inp)

        # 5. 回填 backend 给 run_status（让平台看到是 built 还是 loaded）
        context.info["backend"] = backend
        context.info["pipeline"] = "pp_fast_v1"
        if dev is not None:
            context.info["infer_device"] = dev

        # 落点判定器：唯一实现见 emit_rule.py；参数默认值 = 参考实现原值
        self._rule = EmitRule()

    def reset(self, video_meta: VideoMeta) -> None:
        self._tiler.reset()
        self._rule.reset(video_meta.width, video_meta.height)
        self._timer.reset(video_meta.video_id or "")

    # ── 主循环 ──────────────────────────────────────────────────────────────
    def process_frame(self, frame_rgb, frame_meta) -> list[Landing]:
        fid = frame_meta["frame_id"]
        tm = self._timer

        t0 = _now()
        inp = self._tiler.push(frame_rgb)          # resize + 归一化 + 滑窗组装
        tm.add("preproc", _now() - t0)

        t0 = _now()
        heat = self._infer(inp)                    # TRT + D2H(ch1)
        tm.add("infer", _now() - t0)

        t0 = _now()
        det = pp_fast.decode_heat(heat, frame_meta["width"], frame_meta["height"])
        tm.add("decode", _now() - t0)

        # ch1 解码本身滞后 1 帧；再等 2 个轨迹点确认局部极大（判定见 emit_rule.EmitRule）
        t0 = _now()
        out = self._rule.tick(fid, det)
        tm.add("emit", _now() - t0)
        tm.n += 1
        return out

    def finish_video(self, video_meta: VideoMeta) -> list[Landing]:
        # 尾帧降级：至少 1 个未来点也允许判定，避免丢视频末尾落点
        out = self._rule.flush()
        self._timer.report(
            extra=f"traj={self._rule.n_traj} landings={self._rule.n_emitted} "
                  f"[{describe_params()}]")
        return out

    # 落点判定逻辑已整体移至 emit_rule.EmitRule（唯一实现）：
    # 好处是离线调参工具、等价性测试、提交管线用的是同一份代码，参数不会各自漂移。

