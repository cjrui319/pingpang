# 乒乓球落点检测方案（participant/）

多角度（0°/45°/90°）乒乓球视频 → 逐帧球心 → 触台帧判定 → 落点坐标。

## 1. 交付物清单

```
participant/
├── solution.py            参赛入口（run.py 通过 participant.solution:Solution 加载）
├── ball_detector.py       球检测：热图模型 → 逐帧球心（多通道冗余观测）
├── landing.py             落点判定：轨迹融合/清洗 + 触台"折角"检测（纯 numpy）
├── base_solution.py       官方基类（未改动）
├── configs/landing.json   全部超参（检测阈值 + 按视角分档的落点参数）
├── weights/ball.onnx      官方提供的球检测权重（未改动）
└── dev/                   离线开发工具（**不随提交物打包**，见第 5、6 节；含读真值的评分脚本）
    ├── eval_f1.py         按官方口径本地评分（±1 帧 / ≤50px / Kuhn 一对一匹配）
    ├── tune.py            用缓存的逐帧观测做网格搜索
    ├── replay.py          从缓存观测**重放生产代码**的落点链路（秒级出分）
    ├── calib_offset.py    标定"输出通道 ↔ 帧偏移"
    ├── run_local.sh       本地一键复跑（绕开 run.sh 的平台路径；--check 只跑 1 条）
    └── test_landing.py    落点算法冒烟测试（只需 numpy，含断口鲁棒性用例）
```

## 1.5 实测结果（公开验证集 15 条，本地复刻官方口径）

| 视角 | 视频 | TP | FP | FN | 忽略 | P | R | **F1** |
|---|---|---|---|---|---|---|---|---|
| 0° | 5 | 20 | 0 | 1 | 0 | 1.000 | 0.952 | **0.976** |
| 45° | 7 | 34 | 1 | 4 | 0 | 0.971 | 0.895 | **0.932** |
| 90° | 3 | 31 | 0 | 2 | 2 | 1.000 | 0.939 | **0.969** |

**精度分 = 71.90 / 75**（官方基线 43.7 / 75），三视角 F1 均值 0.959 vs 基线总 F1 0.604。

相对上一版（68.35）的三个增量，按收益排序：

1. **轨迹清洗改成"折角感知"（`clean_mode: side`，45° 专属）**：触台帧本身天然偏离
   "前后邻点插值弦"（球心速度在此反向），而它在运动模糊下 conf 常掉到 0.9 以下，
   于是原"弦残差 + conf<0.9"判据会**把真落点当离群点删掉**——实测 v000085 的 f128
   与真值只差 18px、`Δv_y=-181px/帧`，却因 conf=0.89 被删，白丢一个 TP。新判据：
   只要贴合**前侧两点**或**后侧两点**的直线外推就保留（触台顶点必贴合前侧外推、
   反跳后的点必贴合后侧外推，两侧都不贴合的才是真离群点）。45° F1 0.853 → 0.907。
2. **物理门控 `vb_min`/`va_max`/`dt_max`/`conf_min`**（新增，缺省关闭）：真触台必须
   "触台前在下落（`v_y前 > 0`）、触台后在抬升（`v_y后 < 0`）"，配合紧邻观测帧距与
   候选点置信度下限，把大量 **-16px 量级的弱伪折角**（球还在上升时的假折角、
   跨断口的跳变）判掉。0° F1 0.927 → 0.952。
3. **`conf_gate` 0.6 → 0.5 + 每视角放松 `max_resid`/`span`**：多通道融合后观测足够密，
   放宽观测门槛 + 加长单侧窗口（0° 用 `span=6`）能把**跨观测断口的真触台**纳入候选，
   召回 32 → 34（45°）。
4. **回合状态机 `rally_filter`（并入队友 `24c2fce`）**：静态阶段（回合开始前的摆放/抛球）、
   桌边反弹（`|dv_y| > 60` 的离台加速折角）、二次弹跳（同一回合的第二次触台）、
   90° 同侧连续四条规则。**删掉的 4 个落点经逐点核对全是 FP（最近真值 74~178px），
   0 TP 损失、0 新增** → 三视角 FP 5→1，70.32 → **71.90**。
5. **异步推理流水线（`_AsyncTrt`）**：单帧 TRT 前向原本整段串行在 `process_frame` 里；
   改成「本帧 submit、下帧 finish」后与框架的 NVDEC 解码重叠，方案侧
   5.5 → **4.7 ms/帧**，同 engine A/B：同步 46.9 → 异步 **48.5** fps
   （速度分 23.4 → 24.2），`predictions.jsonl` 与同步版**逐字节相同**。


> 注：三者彼此耦合（`clean_mode`/`conf_gate` 决定候选集，进而改变 NMS 的竞争结果），
> 必须一起调；本表可由 `python3 participant/dev/replay.py --detections <out>/detections`
> 用**生产代码重放**独立复现（本次得 71.90 / 75、每视角 TP/FP/FN 与上表逐项相同）。

> ✅ **完整得分已在评测机实测**（2026-10-18，单张 RTX 4090 + TensorRT 8.6，框架自带
> `--decoder gpu` 解码）：精度 **71.90 / 75** + 速度 **23.8~24.4 / 25** = **95.7~96.3 / 100**
> （三次复跑 95.73 / 96.10 / 96.25）。速度项 = `infer_fps 47.7~48.7`（15 条 / 2109 帧 /
> 43.3~44.3 s）；同一份代码 `PP_ASYNC=0` 关掉异步流水线时是 46.9 fps（速度分 23.4），
> 即重叠收益 ≈ +1.6 fps。按框架口径 `infer_sec` 已扣除 prepare（冷启动建 TRT engine
> 约 270 s，单列在 `engine_build_sec`，不计入 FPS）。提速前后的 `predictions.jsonl`
> **逐字节相同**，精度分不受影响。
「忽略」= 命中 GT 里 `valid=false` 的干扰点，按赛题口径既不算 TP 也不算 FP。

> 📌 **上限与瓶颈（实测）**：92 个真值落点在 ±1 帧内**全部都有 ≤50px 的观测**
> （`dev` 里用 dump 逐点核过），所以瓶颈不在检测召回，而在"把哪一帧判成触台"。
> 用当前候选规则搜"完美选择"（oracle）得 **74.67 / 75**；7 个 FN 逐条归因
> （`tools/tune_offline.py --triage`）：
> (a) **4 条没有反向折角信号**——最近候选的 `Δv_y` 为正（真值帧之后球仍在加速下落/
> 上升），或候选偏到 2~3 帧外（v000066@f2 的候选在 f4、v000106@f7 在 f10），
> 不是阈值能救的；(b) **2 条被方向一致性门控挡掉**（`|Δv_x|/|Δv_y|` = 0.92 / 0.94）：
> 实测放开（`dv_strong=13, ratio_strong=1.0`）会**丢 2 个 TP、多 5 个 FP**
> （精度 71.90 → 69.94），得不偿失；(c) 1 条（v000094@f43）候选在 f44 且 ratio 0.94。
> **结论：71.90 是当前特征集（两段线性拟合的折角）下的局部最优**，继续提分要换模型
> ——整段轨迹的分段抛物线拟合、或对候选帧做亚帧级的 y 极值插值——而不是再调阈值。

该表的数字由**完整 15 条端到端复跑**（解码 → 推理 → 落点分析 → 写文件）产出，
且与 `dev/replay.py` 用生产代码重放的结果**逐字节一致**（88 行预测全部相同，含浮点坐标），
因此"调参器预测的分数"与"真跑拿到的分数"不存在静默失配。

## 2. 方法

**Step 1｜观测（ball_detector.py）**
`ball.onnx` 输入 `(1,9,288,512)`（3 帧 RGB 栈，ImageNet 归一化），输出
`(1,3,288,512)` 的热图。逐帧推理时，**通道 c 编码的是帧栈中第 c 帧**，即
对以帧 `f` 结尾的栈，`ch2→f`、`ch1→f-1`、`ch0→f-2`（由 `dev/calib_offset.py`
用真值落点标定确认：三个通道的峰值都落在真值 7px 内）。
因此三通道同时启用 = 每帧最多 3 票冗余观测，`fuse_obs` 对同一帧的多票取均值，
既降噪又补足丢检。

**Step 2｜触台判定（landing.py，核心创新点）**
触台瞬间只有**竖直速度反向**，水平速度基本不变。这个突变在图像上表现为球心
速度矢量的折转，与视角无关。对每个候选帧，用前后各 `k` 个观测分别线性拟合
出触台前后的速度 `v⁻`、`v⁺`，取

    Δv_y = v⁺_y − v⁻_y  ≤ dv_y（负阈值），且 |Δv_x| ≤ ratio·|Δv_y|

* `Δv_y` 足够负 ⇔ 下落被瞬间抬升 ⇔ 触台；
* `|Δv_x|/|Δv_y|` 的上限来自"竖直反跳在图像中的投影方向"，用于滤掉跟错目标、
  轨迹跳变造成的伪折角；**强折角档**：当 `|Δv_y| ≥ dv_strong`（折角远超噪声，
  几乎不可能是轨迹跳变）时放宽到 `ratio_strong`——侧旋/斜视角下真触台的水平速度
  也会明显变化，单档硬门槛会把真落点误杀（45° 实测真落点该比值达 0.65~0.92）；
* 紧邻观测必须在 `max_step` 帧内，否则轨迹断口会伪造出 ±100px/帧² 的巨型折角。

同一窗口内可能有若干帧都过阈值（真触台帧附近 1~2 帧的拟合也"像"折角），
用 NMS（`min_gap` 帧内互斥）留下一个，谁留下由 `nms` 决定：`maxy` 最低点优先、
`dv` 折角最猛者优先、**`combo` 折角强度×方向一致性优先**（90° 用这个：最低点常落在
触台后 1~2 帧——球继续远离相机，投影 `y` 更低——整窗就会选错帧）。

**Step 2.5｜球台投影区门控（landing.keep_in_roi，收益最大的单点改动）**
触发台折角的**不只是触台**：球离台后落到地面/挡板同样是一次真实抛物线折角，`Δv_y`
信号与台内触台完全一致，折角判据无法区分——只有"落点位置是否在球台投影区内"能判掉。
各视角 ROI（归一化 `[x0,y0,x1,y1]`）取该视角**真值落点包围盒外扩**（0/45 度 50px、
90 度 100px；`python3 participant/dev/tune.py --calib-roi 50` 可一键标定），
属于机位常量。`keep_in_roi` 带 **fail-open**：若过滤后剩余落点不足 `roi_min_keep`（0.5）
比例，说明 ROI 先验与当前机位不符（换机位/裁剪），整体放弃过滤，宁可放过 FP
也不丢真落点。

⚠️ **为什么不用"y 取局部极大"**：0°/45° 是沿球台纵深方向俯视，触台后球同时
远离相机，投影 `y` 会继续增大——用抛物线顶点/极值法实测 F1 只有 0.05，而折角
判据能到 0.7~0.8。这是本方案相对朴素做法的关键差异。

**Step 3｜坐标**
触台帧观测到的球心即落点坐标（真值就是标注者在触台帧标下的球心位置，实测
在真值帧上模型球心与真值中位差约 7px ≪ 50px 匹配容差）。

## 3. 运行

```bash
# 平台/评测机：单任务端到端（GPU 解码 + TRT 优先，自动回退 onnxruntime）
./run.sh <input_dir> <output_dir>

# 本地（无 GPU / 无 torch / 无 conda_env）：一条命令跑全量 + 评分
bash participant/dev/run_local.sh            # 全量 15 条，CPU ≈ 13 分钟
bash participant/dev/run_local.sh --check    # 只跑 1 条小视频自检（≈ 40 秒）

# 本地评分（复刻官方口径）
python3 participant/dev/eval_f1.py --predictions <output_dir>/predictions.jsonl

# 落点算法冒烟测试（不需要模型/视频/GPU）
python3 participant/dev/test_landing.py
```

> **本地 `./run.sh` 跑不了是正常的，不是方案问题。** 官方入口是给评测镜像写的，
> 里面写死了平台布局：`PY=/opt/conda/envs/conda_env/bin/python3`、
> `"$PY" /participant/pingpang/integrity.py --verify`、`exec ... /participant/pingpang/run.py`、
> `INPUT=${1:-/participant/input}`、`DECODER=${DECODER:-gpu}`。
> 本机没有 `/participant/...`（也没有 `/opt/conda/envs/conda_env`、没有 torch/tensorrt/GPU），
> 所以脚本在第 17 行 `integrity.py` 处就因 `set -e` 退出（**退出码 2**，报
> `can't open file '/participant/pingpang/integrity.py'`）。
> `run.py` 本身与平台路径无关（只 `sys.path.insert` 自身目录、`--input/--output` 为必填参数），
> 直接调用即可；本地唯一必须改的是 `--decoder cpu`（走 `weights/ball.onnx` + onnxruntime）。
> `dev/run_local.sh` 就是把这条命令固化下来。
> 若想让官方 `./run.sh` 也跑通：`sudo mkdir -p /participant && sudo ln -sfn "$PWD" /participant/pingpang
> && sudo ln -sfn "$PWD/public_data" /participant/input && sudo mkdir -p /participant/output
> && DECODER=cpu ./run.sh /participant/input /participant/output`（`integrity.py --verify` 本地实测可通过）。

调参闭环（在任意只有 numpy 的机器上即可完成，不需要 GPU）：

```bash
PP_DUMP_DET=1 python3 run.py --input <input> --output <out> --decoder cpu   # 落盘逐帧观测
python3 participant/dev/tune.py --detections <out>/detections --calib-roi 50 --write
                                        # 网格搜索（含 ROI 标定）→ 写回 config
python3 participant/dev/replay.py --detections <out>/detections             # 秒级重放验证（出分）
python3 run.py --input <input> --output <out>                               # 用新参数重跑出最终结果
```

`replay.py` 直接实例化 `Solution` 并调用其 `prepare`（加载真实配置）+ `_analyze`
（真实算法），因此**测的就是生产代码**。落点分析是纯后处理（只吃逐帧观测），
推理结果已被 dump 完整捕获，所以重放结果与真跑一遍推理逐字节一致——
改一次参数验证只需几秒，不必重跑十几分钟推理。改完配置务必用 `replay.py`
或整跑复核，避免"配置写了但代码没读到"这类静默失配。

## 4. 算力与工程取舍

* 模型是 5.3MB 的小型热图网络，单帧 288×512 推理在 4090 上实测 **2.5 ms**（TRT FP16）；
  `prepare` 里先做一次 warmup，避免首帧开销计入推理计时；TRT engine 带
  (TRT 版本 + GPU 型号 + onnx 哈希) 目录级缓存，第二次起直接反序列化
  （实测 prepare 278 s → 7 s）。
* **单帧预处理曾是最大开销**：老写法 `resize → astype/255 → 减均值 → 除 std →
  transpose` 要 **4.7 ms/帧**（占方案侧 56%，TRT 才 2.5 ms）。改用
  `cv2.dnn.blobFromImage`（一次完成 resize + 转 float32 + 减均值）再逐通道原地乘
  `1/(255·std)`，降到 **1.2 ms/帧**；与解析式 `(x/255-mean)/std` 数学等价
  （实测最大差 5e-7，15 条视频预测逐字节不变）。全流程 **35.6 → 45.8 fps**
  （速度分 17.8 → 22.9）。
* **异步推理流水线**：框架的循环是 `decode → process_frame`，单帧 2.5 ms 的 TRT 前向
  原本整段串行在 `process_frame` 里。改成 `_AsyncTrt` 的「本帧 submit、下帧 finish」
  后（`BallDetector.push/flush`，观测帧号仍按绝对帧号落账），前向与下一帧的 NVDEC
  解码重叠，方案侧 **5.5 → 4.7 ms/帧**、`infer_fps` **45.8 → 48.4**（速度分
  22.9 → 24.2）；`predictions.jsonl` 与同步版逐字节相同。用 `PP_ASYNC=0` 可现场关掉
  做 A/B（此时退回 `detect()` 同步路径）。
* 剩余瓶颈是框架解码（`core/decoder.py` 受完整性保护，1080p 实测 ~15 ms/帧 →
  65 fps 上限），方案侧已降到 5.5 ms/帧（异步流水线后 4.7 ms），不再是主导项。
* 落点判定全部是 numpy 的 O(n) 拟合（n=帧数），单视频耗时 ≪ 1ms，
  对 50fps 的速度分没有影响。
* 所有观测在 `finish_video` 统一分析（框架允许延迟输出），因此落点判定不受
  "必须即时输出"的限制；单视频异常被 try/except 隔离，不会拖垮整场。
* 输出前统一 clamp 到 `[0, width-1e-3]×[0, height-1e-3]`，杜绝越界非法行。

## 5. 合规自检（对照赛题硬约束）

| 硬约束（来自 `pppp/赛题详情.md` / `PROMPT.md`） | 本方案状态 | 证据 |
|---|---|---|
| 不得改框架：`core/*.py`、`run.py`、`validate.py`、`integrity.py`、`scripts/*.py`、`run.sh`、`.integrity.json` | ✅ **零改动** | 这 15 个文件与官方 `origin/main:participant/pingpang/*` **逐字节相同**；`python3 integrity.py --verify` → `完整性校验通过`（打包时各任务各跑一次） |
| 只能改参赛者代码区 `pingpang/participant/` | ✅ 全部改动都在代码区内 | `git diff --stat 8dcc186 HEAD`：`solution.py`(重写)/`ball_detector.py`(改 90 行)/新增 `landing.py`、`configs/landing.json`、`dev/*`、`README.md`；框架目录零改动 |
| 不得升级/替换 torch、numpy==1.26.4、onnxruntime-gpu==1.18.1、tensorrt==8.6.1 | ✅ 未装/未改/未联网 | 生产代码只 import `numpy/cv2/onnxruntime/tensorrt`（连 torch 都不需要），`requirements` 无新增 |
| 权重以 ONNX 放 `participant/weights/` 打进镜像 | ✅ | `weights/ball.onnx`（官方权重，未改动，5.3MB） |
| 输出合法：`frame_id` 为 int 且在 `[0,n_frames)`，坐标有限且 `0<=x<width, 0<=y<height`，不越界、去重 | ✅ | 统一 clamp + 去重；15 条全量 88 行预测 **0 条非法**（`validate.py` 无 errors） |
| 输出物 < 1GB、每任务 < 7200s、离线无联网 | ✅ | 输出仅 `predictions.jsonl`（KB 级）；CPU 全量 15 条约 13 min；无任何网络调用 |
| prepare 里构建 TRT（不占 FPS 计时） | ✅ | `prepare` 内 TRT 构建 + warmup，engine 缓存到 `<output>/trt_cache`，失败自动回退 onnxruntime |

> ⚠️ 一处**评审观感**风险已处理：提交物里若留着 `participant/dev/`，等于"提交物内含读真值
> （`gt_public.jsonl`）的代码"——官方 `tune_eval.py` 刻意放在工程根**之外**就是这个原因。
> 因此工程根的 `pack_submit.sh` 打包时**自动剔除 `dev/`、`__pycache__`、`*.pyc`**。

## 6. 如何提交（本机已具备条件）

本机布局即提交布局：`~/dev/projects/participant/` 就是**工程根**（对应容器 `/participant`），
内含 `run_all.sh` + `pingpang/` + `basketball/`（附加题保持官方参考实现不动，评测按参考实现计分）。

```bash
cd ~/dev/projects/participant
bash pack_submit.sh                 # 组装干净提交树 + zip，并自动跑两任务的 integrity 校验
# → submit/participant.zip  （解压得到 participant/ 工程根，316MB）
# LEAN=1 bash pack_submit.sh        # 再剔掉验证集视频（镜像小三~300MB）
# bash pack_submit.sh --docker      # 额外 docker build + docker save → submit/sport-vision-submit.tar
```

* **方式一（平台，推荐）**：`https://aigpu.migu.cn/home` 创建 **2 卡 4090** 实例，镜像必须选
  `pytorch 2.1.2-ubuntu22.04-p3.10-cuda11.8`；上传 `submit/participant.zip` 解压到 `/participant`，
  按官方命令装依赖 → 自测 → 按平台要求提交。
* **方式二（本地 Docker）**：需先下载并 `docker load -i sport-base_v1.tar`（`sport-base:v1`，23.8GB，
  **本机没有，须从比赛官网下载**），然后 `bash pack_submit.sh --docker`，把
  `submit/sport-vision-submit.tar` 上传平台（入口已设为 `/participant/run_all.sh`）。
* 提交前建议再跑一次与评测同形态的联合自测：

```bash
cd ~/dev/projects/participant && mkdir -p input
ln -sfn $PWD/pingpang/public_data   input/pingpang
ln -sfn $PWD/basketball/public_data input/basketball
./run_all.sh input output     # 产物 output/{pingpang,basketball}/{predictions.jsonl,run_status.json,run.log}
```

* ⚠️ **每队最多 5 次提交机会，以最后一次成功提交计分**（运行失败不占次数）；每跑完一次自测就打 tag：
  `git tag -a sub1 -m "第1次提交" && git push --tags`，保证能随时回到得分最高的版本。
* 官方分支约定：开发用 `dev/pingpang-solution`（本仓库目前在个人分支 `toshi`）。

