"""参赛者自带的推理辅助：GPU 设备选择（仅此一项被 solution.py 使用）。

历史说明：本文件早期包含一套「零分配推理 + 锁页搬运」的快速管线
（FrameTiler / FastTrt / RefTrt / decode_heat），用于在「参考实现的逐帧
y 局部极大」判定下优化速度。自落点判定改为「整轨迹速度折角」后，
solution.py 不再走那条逐帧快速路径，那些代码随之移除，只保留：

  * select_device —— 解析并校验 PP_GPU_INDEX，越界/非法一律回退当前设备，
    避免 torch.cuda.set_device 越界抛异常拖垮 prepare（见其 docstring）。
"""

from __future__ import annotations

import os

__all__ = ["gpu_index_from_env", "select_device"]


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
        这个异常会被 solution.py 的兜底捕获并回退 onnxruntime，而回退路径用同一个
        非法序号**再抛一次** -> 最终 prepare 失败 -> 整个提交 0 分。
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
