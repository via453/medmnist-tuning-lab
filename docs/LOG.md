# 实验日志

> 逐条记录**机器字段之外的判断与现象**；结构化字段由 `src/run.py` / `src/sweep.py` / `src/repeat.py` 自动写入 `results/*.csv`。
> 环境：`d2l-zh` / Python 3.9.25 / torch 2.5.1+cpu / numpy 1.21.5 / **纯 CPU，无 CUDA** / `num_workers=0`
> 记录规则：每组实验（含失败组）一行，失败组不删；test 全程只评一次；每条带 `git rev-parse --short HEAD`。

## 日志条目

### 2026-11-13 02:47 · run_id: base
- 配置：`configs/base.yaml`（SmallCNN / w32 / Adam / lr 1e-3 / batch 64 / wd 1e-4 / dropout 0.2 / BN / flip / light pooling / 20 epoch）
- 命令：`python -m src.run --config configs/base.yaml`
- 现象：20 轮跑满未触发早停，val AUC 0.984404、ACC 0.9275，参数量 19,042（手算核对一致），1.79 min。
- 判断：基线可用，但 val ACC 与 AUC 都还有空间；先做单因子扫描定位敏感旋钮，不盲目上复杂模型。
- git commit：`440c127`

### 2026-11-13 02:50 – 04:54 · run_id: s1_01 – s1_33（33 组）
- 配置：`configs/sweep.yaml`，单因子（OFAT），seed=0，每组只改一个字段
- 命令：`python -m src.sweep --config configs/sweep.yaml`
- 现象：
  - 13 组「等价 base」的重复全部精确复现 0.984404 → 三处 seed 固定有效，结果可复现。
  - 极差排序：lr 0.0220 > optimizer 0.0204 > use_bn 0.0186 > conv_width 0.0100 ≫ 其余七项（≤0.0062）。
  - `s1_07`（SGD）0.9642，20 轮内明显没收敛完；`s1_02`（lr 1e-4）0.9658，跑满 20 轮仍欠拟合——两头都是「训练预算不足」而不是模型不行。
  - `s1_06`（lr 1e-2）第 16 轮早停、`s1_03`（lr 3e-4）第 17 轮早停 → 学习率两端都会让 val 提前停止改善。
  - `s1_25`（class_weight=balanced）AUC 0.9856 但 ACC 掉到 0.8760，且耗时 6.01 min（最长）。
  - `s1_23`（flip+旋转）0.9821 低于 base 与「不增强」；`s1_33`（不池化）0.9782 低于 base。
- 判断：最优 `conv_width=64`（0.988956）相对 base 仅 +0.0046，**小于 lr 组自身极差 0.022**，单 seed 不能下结论 → 必须做 3 seed 复跑（S3）。同时记下两条反直觉现象待解释（增强有害、不池化更差）。
- git commit：`440c127`

### 2026-11-13 13:24 – 13:29 · run_id: grid_*（4 组）
- 配置：`configs/base.yaml` + 网格 `lr ∈ {1e-3, 3e-3} × conv_width ∈ {32, 64}`，seed=0
- 命令：`python -m src.repeat --mode grid --grid "lr=0.001,0.003;conv_width=32,64"`
- 现象：0.984404 / 0.988956 / 0.987870 / **0.990669**（lr 3e-3 + w64 最高），单组 1.2–1.9 min。两旋钮近似可加，无明显交互。
- 判断：确认 OFAT 的假设成立；但网格最优高于 S1 单因子最优 0.0017，仍在 seed 波动待验证范围内 → 交给 S3 判断，**不据此改配置**。
- git commit：`1010240`

### 2026-11-13 13:31 – 13:43 · run_id: seeds_*（9 组，S3）
- 配置：base / `conv_width=64` / `lr=3e-3` 各 3 个 seed（0/1/2）
- 命令：`python -m src.repeat --mode seeds --results results/runs.csv`
- 现象：
  - base 0.984537 ± 0.000215（极稳）；w64 0.988645 ± 0.000490；lr 3e-3 0.987127 ± 0.000933。
  - 逐 seed 配对差值：+0.004552 / +0.003657 / +0.004113，**三个 seed 全部同向**。
  - lr 3e-3 的标准差是 base 的 4.3 倍，且 seed1/seed2 分别在 15/17 轮早停。
- 判断：差值 ≈19 倍 base 标准差且方向一致 → **提升真实**，可以写「调参带来稳定提升」。同时决定最终配置取低方差的 `conv_width=64 + lr 1e-3`，不取均值略高的 lr 3e-3（稳定性优先，且网格最优组留作遗留改进）。
- git commit：`1010240`

### 2026-11-13 13:46 · run_id: best
- 配置：`configs/best.yaml`（base + `conv_width: 64`），seed 0
- 命令：`python -m src.run --config configs/best.yaml`
- 现象：20 轮，权重取 **epoch 19（val AUC 0.9890）**，末轮 epoch 20 为 0.9837；`outputs/best_history.json` 记录双损失曲线，val 损失在第 18 轮后开始回升 → 轻度过拟合。
- 判断：保存最佳轮而非末轮是必要的；这与 C1「用验证集做模型选择」一致。若没有早停/最佳轮保存，最终交付的会是那个更差的末轮模型。
- git commit：`440c127`

### 2026-11-13 13:46:57 · run_id: final（test，**仅此一次**）
- 命令：`python -m src.run --config configs/best.yaml --split test`
- 现象：test AUC 0.954635 / ACC 0.834936 / F1 0.882554，混淆矩阵 [[134, 100], [3, 387]]。
- 对比 val（0.988956 / 0.952290，[[118, 17], [8, 381]]）：AUC −0.0343、ACC −0.1174，假阳性 17 → 100。
- 判断：不是 bug。MedMNIST 的 test 来自独立的 source validation set（train/val 是同一来源的 9:1 切分），且阳性率 74.2% → 62.5%，固定阈值 0.5 在低先验下必然偏激进。**按纪律不回头调阈值、不换配置重评**，把落差写成 C11 协变量偏移的实证，并在答辩主动说明。
- git commit：`440c127`

### 2026-11-13 13:47 · run_id: report（val 出图）
- 命令：`python -m src.report --config configs/best.yaml --ckpt outputs/best.pth --split val --history outputs/best_history.json`
- 现象：生成 `roc.png` / `confusion_matrix.png` / `samples.png` / `metrics.json`；`metrics.json` 的 AUC 0.988956 与 CSV 中 best 行一致，确认权重与记录对得上。
- 判断：出图统一用 val，不再碰 test。
- git commit：`440c127`

### 2026-11-13 15:08 – 15:10 · run_id: mlp_s0 / mlp_s1 / mlp_s2（结构对照）
- 配置：`model: mlp`（784→128→2，dropout 0.2），其余同 base；3 个 seed；结果写入 `results/structure.csv`
- 命令：`python -m src.run --config .workbuddy/mlp.yaml --seed <0|1|2> --run-id mlp_s<n> --out results/structure.csv --no-ckpt`
- 现象：val AUC **0.992796 ± 0.000699**、ACC 0.9466、参数量 100,738、单次仅 0.74 min；seed1/seed2 在 13/18 轮早停。**比调参后的 SmallCNN（0.988645）还高 0.0042。**
- 判断：原假设「图像任务 CNN 必赢」被推翻。归因：28×28 上病灶表现为整片肺野的强度/纹理统计，全局特征判别力强；而 SmallCNN 的 dropout+BN+池化+GAP 把容量压到 19k，属于正则过强。**这提示结构/容量的收益大于超参调优**。MLP 只做 val 对比，**不上 test**（test 已消耗一次）。
- git commit：`440c127`

### 2026-11-13 19:05 – 19:30 · 延伸实验（5 组，全部只评 val）
- 背景：笔记结尾「五件动手的事」里有三件（FGSM、剪枝、attention）能用本项目直接兑现；另补 Grad-CAM（C10）与先验拆解（C11）。**test 已消耗，一律不碰**。
- 命令与结果：
  - `python -m src.run --config .workbuddy/vit.yaml --seed 0/1/2 --run-id vit_s* --out results/structure.csv` → `TinyViT`（71,554 参数）0.993907 / 0.992821 / 0.993792，均值 **0.993507 ± 0.000597**、ACC 0.9548、2.4 min。**高于 MLP 0.9928 与 CNN w64 0.9886**，且与 w64 参数量几乎相同 → 排除「容量更大」这一解释，指向信息混合方式（卷积感受野仅 9×9，注意力第一层即全局）。（**「ViT 最优」这一条在第二轮被容量对齐修正，见下一条日志**）
  - `python -m src.fgsm` → ε=0 / 0.01 / 0.02 / 0.03 / 0.05 对应 AUC 0.9890 / 0.9139 / 0.7140 / **0.4501** / 0.1015；ε≥0.05 后 F1→0、ACC 停在 0.16–0.25（≈val 阴性占比 25.8%），模型退化成「全判正常」。
  - `python -m src.gradcam` → 边框响应占比 0.382、边框面积占比 0.490、富集 **0.78×**，注意力偏中央（肺野）。第一次算出 0.762 是因为 CAM 是 14×14 却按 28×28 统计，已修（见 PITFALLS 第 12 条）。
  - `python -m src.compress` → 剪 50% 0.9837（37,280 非零）、剪 80% 0.9720（14,912）、剪 90% 0.9418（7,457）；学生 w16 从头训 **0.9791（4,914）** 反而优于剪 80%/90%。蒸馏 T=1/4/8 → 0.9783 / 0.9774 / 0.9775，**全部没赢从头训**。
  - `python -m src.prior_shift` → val 0.9523 → 仅把阳性率对齐到 62.5% 后 0.9399±0.0036 → test 实际 0.8349。**先验只解释 −0.0124（11%），其余 −0.1050（89%）是真实分布偏移**。
- 判断：三条负结果（剪枝不如训小的、蒸馏无增益、先验只解释 11%）都写进 `docs/EXTENSIONS.md`，并各自给出前提条件。它们比继续刷 AUC 更有价值：说明「技巧有效」不是无条件命题。
- 产物：`docs/EXTENSIONS.md`、`results/structure.csv`（+3 行）、`results/compress.csv`、`results/fgsm.csv`、`figures/fgsm_curve.png`、`figures/gradcam.png`、新脚本 `fgsm.py` / `gradcam.py` / `compress.py` / `prior_shift.py`、`model.py` 新增 `TinyViT`。

### 2026-11-13 19:59 – 20:38 · 延伸实验第二轮：对抗训练 + 容量对齐（24 组训练，全部只评 val）

- 背景：第一轮留下两个悬案——①「ε=0.03 就崩」能不能修；②「ViT 最优」是不是参数量没对齐造成的假象。两个都只评 val，test 仍不碰。
- 命令与结果：
  - `python -m src.adv_train --seeds 0,1,2`（clean / adv 各 3 seed，共 6 次训练）→ 干净 AUC **0.988600 ± 0.000451** → **0.980539 ± 0.002786**（−0.0081）；ε=0.03 处 AUC **0.467778 → 0.871091**（+0.4033），ACC 0.4847 → 0.8041。逐 seed 配对：干净 −0.0089 / −0.0100 / −0.0053，鲁棒 +0.4168 / +0.4375 / +0.3557，**9 个差值方向全一致**。
  - `python -m src.capacity_match --levels 19042,71554`（2 档 × 3 结构 × 3 seed，共 18 次训练）→ 各档参数量对齐到 18,890–19,394 / 71,554–72,641（偏差 ≤1.8%）：
    - ~1.9 万档：MLP **0.992161 ± 0.000920** / CNN 0.984538 ± 0.000234 / ViT **0.992243 ± 0.001542**
    - ~7.2 万档：MLP **0.992656 ± 0.000127** / CNN 0.987921 ± 0.001376 / ViT **0.993507 ± 0.000597**
- 判断：
  - 对抗训练确实能修，但**只修同强度邻域**：ε=0.08 时已掉到 0.46，ε≥0.2 两者都归零；且代价是干净精度 −0.008 AUC / −1.65 点 ACC。seed 0/1 因「用干净 AUC 早停」被截断在第 12 轮（seed 2 跑满 20 轮、鲁棒性最好 0.8860），**所以 0.8711 是保守下界**（PITFALLS 第 13 条）。
  - 容量对齐后：**CNN 落后是结构性的**（两档都低 0.005–0.008，且 72k 档 CNN 参数还最多 72,641），但 **「ViT > MLP」不成立**（差 0.00008 / 0.00085，落在 seed std 内；逐 seed +0.0011 / +0.0002 / +0.0012）。1.9 万参数的 MLP 与 7.2 万参数的 ViT 只差 0.0013，说明**容量在 1.9 万就饱和**，真正起作用的是「全局混合 vs 局部卷积」。
- 产物：`results/adv_train.csv`、`results/adv_curve.csv`、`results/capacity.csv`、`results/capacity_summary.csv`、`figures/fgsm_adv_compare.png`、`figures/capacity_curve.png`、新脚本 `adv_train.py` / `capacity_match.py`、`configs/adv.yaml`、`model.py` 的 MLP 宽度改为可配、`train.py` 增加 `fgsm_perturb` 与 `adv_eps/adv_ratio`；`docs/EXTENSIONS.md` 新增 §7–§8 并修正 §2 结论。

## 异常与失败组（已保留，未删除）

| run_id | 现象 | 处置 |
| --- | --- | --- |
| `s1_02`（lr 1e-4） | 0.9658，20 轮跑满仍欠拟合，耗时 3.56 min | 保留，作为「学习率过小」的下界证据 |
| `s1_06`（lr 1e-2） | 第 16 轮早停，0.9855 | 保留，作为高 lr 方差的上界证据 |
| `s1_07`（SGD） | 0.9642，最差组 | 保留；说明 20 轮预算下 SGD 不可比，需更长训练或动量 |
| `s1_20`（无 BN） | 0.9658，掉 1.9 点 | 保留，BN 必要性的直接证据 |
| `s1_25`（balanced） | ACC 0.8760、耗时 6.01 min | 保留，AUC/ACC 分离的典型样本 |
| `s1_33`（不池化） | 0.9782，低于 base | 保留，用来给 C3「不堆池化」加边界条件 |

以上组 `status` 均为 `ok`（能跑完、指标可算），不属于程序崩溃；真正的环境级失败只有下面一条。

## 环境问题记录

| 日期 | 现象 | 处理 |
| --- | --- | --- |
| 2026-11-13 | `d2l 0.17.6 requires numpy==1.21.5, but you have numpy 2.0.2`；pandas / matplotlib / sklearn 因 NumPy 2 的 C ABI 变化无法导入 | 按 `requirements.txt` 把整套依赖锁回 numpy 1.x ABI 组合（numpy 1.21.5 / scipy 1.7.3 / scikit-learn 1.0.2 / scikit-image 0.19.3）；`pip check` 无冲突，冒烟测试通过 |
| 2026-11-13 | `ModuleNotFoundError: No module named 'src'` / `Could not open requirements file` | 在 `C:\Users\Zelvia` 家目录执行导致；必须先 `cd` 到仓库根目录。环境本身没坏（`pip check` 与 cwd 无关，不能用来判断目录对不对） |

## 统计口径备注

- std 一律用**样本标准差**（ddof=1，n=3）；`docs/FINAL_ANALYSIS.md` 中用的是总体标准差（ddof=0），故同一组数值略小（base 0.000175 vs 0.000215），两者结论一致。
- 「提升是否显著」的判据不是 p 值，而是：差值 / seed 标准差 ≥ 数倍 **且** 逐 seed 配对差值全部同向。样本量只有 3，做 t 检验意义有限。
