# 动感地带 AI+高校创智计划 · AI技术赛道 —— 乒乓球多视角落点检测

本仓库是**比赛工程的开发仓库**，镜像提交用的工程根目录就是 `participant/`。

| 项目 | 内容 |
|---|---|
| 主赛题 | 乒乓球多视角落点检测（0° / 45° / 90° 三视角，输出触台落点 `frame_id, x, y`） |
| 附加题 | 篮球球员跟踪（MOT，可选做，不影响主赛排名，单独排名） |
| 训练平台 | https://aigpu.migu.cn/home |
| 平台手册 | https://www.kdocs.cn/l/cuxPZDwgX2z3 |
| 成绩查询 | 平台顶部导航「AI + 高校创智大赛」→ 专题详情页（仅 PC 端） |

---

## 一、文件都在哪（重要）

工作区里**只有一部分东西适合进 Git**。完整清单如下：

| 文件 / 目录 | 大小 | 在仓库里？ | 在哪 / 说明 |
|---|---|---|---|
| `participant/` | 16 MB | ✅ 已提交 | **工程根目录**，镜像里对应容器路径 `/participant` |
| `pppp/赛题详情.md` | 13 KB | ✅ 已提交 | 赛题解析 + 评分规则 + 提交形式 + 赛道限定条件 |
| `pppp/pp_train_data.zip` | 680 MB | ❌ 忽略 | 乒乓球训练集，解压后用于训练模型 |
| `pppp/bb_train_data.zip` | 4.9 GB | ❌ 忽略 | 篮球训练集（68 视频 / 40.5 万框 / 815 轨迹） |
| `pppp/sport-base_v1.tar` | **23.8 GB** | ❌ 忽略 | 基础环境镜像，**仅方式二（本地 Docker 提交）需要**；平台方式由平台提供 |
| `participant/*/public_data/videos/` | 218 MB | ❌ 忽略 | 公开验证集视频（乒乓球 15 条 + 篮球 2 条） |

> ⚠️ **`sport-base_v1.tar` 有 23.8 GB，是全部资料里最大的东西。**
> 走「方式一：咪咕仝学平台开发提交」的话，**这个文件完全用不上**——平台的开发环境本身就是这个镜像。
> 只有走「方式二：本地开发 + 提交 Docker 镜像」才需要 `docker load` 它。
> 建议先确认走哪条路，再决定是否保留（本机 D 盘尚有 356 GB 可用）。

### 大文件不入 Git 的原因

GitHub 单文件硬上限 100 MB（超 50 MB 就告警），而上面这些都是 GB 级或可重新下载的比赛素材。
**队友或新机器要拿到它们，请从比赛官方资料重新下载**，不要试图 `git add -f`——
一旦进历史，清理成本远高于重新下载。

---

## 二、快速开始（新机器 / 队友）

```bash
# 1. 克隆工程（不含验证集视频）
git clone https://github.com/drlugu/pingpang.git
cd pingpang

# 2. 补齐公开验证集视频（从 participant.zip 解压到对应目录）
#    注意：压缩包内文件名是 GBK 编码，用支持编码选择的解压工具，
#    确保落成  participant/pingpang/public_data/videos/{0度,45度,90度}/
#    篮球那 2 条落到  participant/basketball/public_data/videos/

# 3. 校验工程完整性（核心，必须通过）
cd participant
python pingpang/integrity.py --verify      # 期望：完整性校验通过
python basketball/integrity.py --verify
```

### 环境依赖（方式一，平台实例内执行）

```shell
apt-get update && apt-get install ffmpeg -y
pip install av numpy==1.26.4 opencv-python-headless scipy \
    onnx==1.16.2 onnxruntime-gpu==1.18.1 \
    tensorrt-libs==8.6.1 tensorrt-bindings==8.6.1 \
    --extra-index-url https://pypi.nvidia.com
pip install tensorrt==8.6.1 --no-build-isolation
```

⚠️ **不得升级 torch / numpy / onnxruntime / TensorRT**——评测环境无网络安装环节，
改了版本可能导致作品在评测机上直接跑不起来。

---

## 三、自测（与评测形态完全一致）

`run_all.sh` 接受「输入目录 / 输出目录」两个参数，输入目录下必须有 `pingpang/` 和 `basketball/` 两个子目录，
对应评测时的挂载形态 `/participant/input/`。**用软链接组织，不要复制数据**：

```bash
cd participant
mkdir -p input
ln -s ../pingpang/public_data   input/pingpang
ln -s ../basketball/public_data input/basketball
./run_all.sh input output
```

自检脚本：

| 脚本 | 作用 | 什么时候跑 |
|---|---|---|
| `pingpang/scripts/check_env.py` | 环境自检：CUDA / torch / 依赖 / 路径 | 改代码**之前** |
| `run_all.sh input output` | 双任务联合自测，用公开验证集打分 | 改完代码 |
| `pingpang/scripts/check_solution.py` | 提交物体检：`Solution` 接口、`weights/`、`configs/` 是否齐全 | 固化镜像**之前** |
| `pingpang/integrity.py --verify` | 框架完整性校验 | 每次准备提交前 |

输出目录会生成 `predictions.jsonl`、`run_status.json`、`run.log`；
**失败时优先看 `run.log` 里的 traceback 和失败阶段**。

---

## 四、只改这三个地方

```text
participant/<task>/
├── core/           ❌ 框架：runner / decoder / types / dataset —— 不得改动
├── run.py          ❌ 单任务入口 —— 不得改动
├── validate.py     ❌ 校验 —— 不得改动
├── scripts/        ❌ 自检脚本 —— 不得改动
├── public_data/    ❌ 验证集 —— 不得改动
└── participant/    ✅ 参赛者代码区（内置官方参考实现，可整体替换）
    ├── solution.py   ✅ 实现 Solution 类（接口见 core/types.py）
    ├── weights/      ✅ 放模型权重
    └── configs/      ✅ 放配置
```

- **完整性校验会 SHA256 逐个比对**（清单在各任务的 `.integrity.json`），
  改了 `core/`、`run.py`、`validate.py` 等 → **校验不过 = 无成绩**，不是扣分。
- 官方参考实现就在代码区内，**可以直接改写**（它只是接入样例，不是精度上限）。
- **只做一个任务时，另一个任务保持参考实现不动**，评测按参考实现计分。

### 🚨 行尾（CRLF）是这套工程的头号杀手

受保护文件按**字节**算 SHA256，Windows 上任何一次 CRLF 转换都会让校验失败。
本仓库已用 `.gitattributes`（`* -text` + `*.sh/.py eol=lf`）防止这件事。
**克隆后如果 `integrity.py --verify` 报校验失败，第一个要怀疑的就是行尾**：

```bash
git config core.autocrlf false     # 仓库里已设，换机器后确认一下
git rm --cached -r . && git reset --hard   # 以 LF 重新检出（谨慎，先确认无未提交改动）
```

---

## 五、提交与评分（必读）

### 提交形式

- **方式一（平台）**：在咪咕仝学平台创建 **2 卡 4090**，镜像必须选
  **`pytorch 2.1.2-ubuntu22.04-p3.10-cuda11.8`**；开发自测完成后按平台要求提交。
- **方式二（本地）**：基于 `sport-base_v1.tar` 构建，入口设为 `/participant/run_all.sh`，
  `docker save sport-vision-submit:latest -o sport-vision-submit.tar` 后上传 tar。

### 评分口径

满分 100 = **精度分 75 + 推理耗时分 25**，两任务分开排名。

| 赛题 | 精度分公式 |
|---|---|
| 乒乓球 | `F1_0° × 25 + F1_45° × 25 + F1_90° × 25`（帧号 ±1 帧、位置 50 px 容错，一对一匹配） |
| 篮球 | `(0.6 × IDF1 + 0.4 × MOTA) × 75`（IoU ≥ 0.5 + 帧内 Hungarian；每帧最多 15 个预测） |

**耗时分**：平均推理 FPS，`1 fps` = 1 分，`50 fps` = 满分 25 分，超过 50 封顶。
→ **推理速度占 1/4 总分，别只堆精度。**

官方参考实现基线：乒乓球 F1 0.604（0° 0.408 / 45° 0.644 / 90° 0.697）；
篮球 IDF1 ≈ 0.34、MOTA ≈ 0.53。

### 硬性限制

- **每队最多 5 次提交成功机会**，**以最后一次成功提交的结果计入排名**。
  ⚠️ 镜像**运行失败不占次数**，**运行成功才消耗**——所以提交前务必自测。
- 环境：Intel CPU + 2×4090，单卡显存 < 24 GB，镜像新增空间 < 50 GB，输出物 < 1 GB，
  每任务总时长上限 **7200 秒**（从任务启动全程计时）。
- 不得修改 `core/`、`run.py`、`validate.py`；不得升级基础组件版本。
- 输出字段越界（如坐标超出 `[0, width)`）**直接判非法，框架不做自动修正**。

---

## 六、分支约定

| 分支 | 用途 |
|---|---|
| `main` | 工程基线，保持与 `participant.zip` 一致，**不在此分支做实验** |
| `dev/pingpang-solution` | 主赛题开发 |
| `dev/basketball-solution` | 附加题开发（如做） |

**每跑一次 `run_all.sh` 自测就提交一次**，并用 tag 记录有效成绩：

```bash
git tag -a sub1 -m "第1次提交，public F1=0.71" && git push --tags
```

因为排名取**最后一次成功提交**，你必须能随时回到得分最高的那个版本。

---

## 七、已知缺口

- [ ] 乒乓球公开验证集视频需从 `participant.zip` 补齐（本机已补齐 15/15，克隆者需自取）
- [ ] 训练集 `pp_train_data.zip`（680 MB）、`bb_train_data.zip`（4.9 GB）尚未解压
      （两者在 `pppp/` 下；解压前先确认输出目录在工程目录**之外**，避免误入库）
- [x] 根目录原 GitHub 初始 README 已被本文件替换
