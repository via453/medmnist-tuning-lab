# MedMNIST 医学影像分类调参实验室

中期大作业验收资料库：**选题论证 + 项目全生命周期教程 + 协作边界 + 可运行代码骨架**。

> **当前状态（2026-11-13 晚，主实验 + 延伸实验完成）**：S1 单因子 33 组、S2 小网格 4 组、S3 三 seed 9 组、结构对照 6 组（MLP×3 + TinyViT×3）、最终 test 一次评测与 8 张图件全部完成；延伸实验（FGSM / Grad-CAM / 剪枝蒸馏 / 先验拆解）5 组已完成并写入 `docs/EXTENSIONS.md`。`docs/REPORT.md`、`LOG.md`、`PITFALLS.md`、`REVIEW.md` 已写完。仓库含脚本、配置、结果 CSV 与图件；**不含训练权重**（`outputs/` 已忽略）与**原始数据集**（`data/` 已忽略）。

## 项目一句话目标

在同一个 MedMNIST 二分类子数据集上，固定数据划分与评估口径，用自己写的 MLP / SmallCNN 作为基础预测模型，通过系统化调参把 AUC / ACC 稳定抬高，并说清是哪些超参在起作用。

- 主数据集：PneumoniaMNIST（5,856 张；train / val / test = 4,708 / 524 / 624），可选 BreastMNIST 做泛化验证
- 基线阶梯：多数类常数 / 逻辑回归 → MLP → SmallCNN → SmallCNN + 最优超参
- 核心纪律：val 选配置、test 只评一次、3 个 seed 报均值 ± 标准差
- 运行环境：本机 CPU（torch 2.5.1+cpu，无 GPU），单次训练约 1–3 分钟

## 仓库内容

| 文件 | 说明 |
| --- | --- |
| `index.html` | 导航页，三份文档的在线入口（开启 GitHub Pages 后可直接访问） |
| `中期大作业选题建议.html` | 选题建议报告 v2：5 个候选选题（2 复刻 + 3 自建）、横向对比、首推 B-1 及理由；内含可复用的项目流程图 SVG |
| `MedMNIST调参实验室_完整教程.html` | **主教程**：按立项 → 数据理解 → 实验配置 → 调参对比 → 结果评估 → 过程记录 → 最终归档组织，含可填写章节映射表、三类协作边界与最小工作清单 |
| `MedMNIST调参实验室_详细教程.html` | 另一版详细教程：技术边界、数据子集划分、超参清单、异常判定、已学章节覆盖对照与协作边界 |
| `src/` | **代码**：`data` / `model` / `train` / `evaluate` / `run` / `sweep` / `plots` / `repeat`（S2·S3）/ `report`（ROC·混淆矩阵·预测样例）/ `analyze`（数据摘要）/ `select_best`（选优）/ `utils`；延伸实验：`fgsm`（对抗攻击）/ `gradcam`（可解释性）/ `compress`（剪枝·蒸馏）/ `prior_shift`（先验拆解）/ `adv_train`（对抗训练对照）/ `capacity_match`（容量对齐后比结构） |
| `configs/` | `base.yaml`（基线）、`sweep.yaml`（单因子扫描）、`best.yaml`（最终配置：conv_width=64）、`vit.yaml`（TinyViT 结构对照）、`adv.yaml`（对抗训练 ε=0.03 / ratio=0.5） |
| `docs/pipeline.svg` | 项目流程图 / 框图，可直接用于 README 与 PPT |
| `docs/EXTENSIONS.md` | **延伸实验**：结构谱系（ViT / MLP / CNN 同参数量对照）、FGSM 鲁棒性、Grad-CAM、剪枝与蒸馏、val→test 落差拆解、对抗训练、容量对齐修正 |
| `docs/*.md` | `REPORT.md`（实验报告：结构对照 + 超参对比 + 显著性 + test + 局限）、`LOG.md`（实验日志）、`PITFALLS.md`（踩坑清单）、`REVIEW.md`（学习复盘 + 章节映射表）为署名文档；`S1_ANALYSIS.md` / `FINAL_ANALYSIS.md` 为自动整理的数据摘要（参考用，非结论）；`PPT_OUTLINE.md` / `DEFENSE_QA.md` 为汇报与答辩建议稿 |
| `requirements.txt` | 钉死 numpy 1.x ABI 兼容组合，防止依赖冲突复发 |
| `notebooks/demo.ipynb` | 单图推理演示（需先有本地训练出的权重） |

建议阅读顺序：选题建议 → 完整教程 → 详细教程（按需查阅）。

## 目录结构

```
medmnist-tuning-lab/
├── README.md / index.html / LICENSE（待确认）
├── requirements.txt / .gitignore
├── configs/     base.yaml · sweep.yaml · best.yaml · vit.yaml · adv.yaml
├── src/         data · model · train · evaluate · run · sweep · plots · repeat
│                report · analyze · select_best · utils
│                fgsm · gradcam · compress · prior_shift · adv_train · capacity_match   ← 延伸实验
├── docs/        pipeline.svg · REPORT.md · LOG.md · PITFALLS.md · REVIEW.md
│                EXTENSIONS.md · S1_ANALYSIS.md · FINAL_ANALYSIS.md
│                PPT_OUTLINE.md · DEFENSE_QA.md
├── results/     runs.csv · grid.csv · repeats.csv · structure.csv
│                compress.csv · fgsm.csv · adv_train.csv · adv_curve.csv
│                capacity.csv · capacity_summary.csv · test_final.csv
├── figures/     6 张主实验图 + fgsm_curve.png · gradcam.png
│                fgsm_adv_compare.png · capacity_curve.png
├── notebooks/   demo.ipynb
├── data/        .npz 自动下载，已忽略
└── outputs/     训练权重，已忽略
```

## 延伸实验一览（详见 `docs/EXTENSIONS.md`）

| 实验 | 结论（val 口径） |
| --- | --- |
| 结构谱系（C4/C5） | 原始对照：TinyViT 0.9935（71,554）> MLP 0.9928（100,738）> SmallCNN 0.9886（74,946）；**容量对齐后被修正**，见下一行 |
| 容量对齐（C4/C5/C3） | 压到 ~1.9 万 / ~7.2 万两档后：CNN 稳定低 0.005–0.008（加参数救不回），**ViT 与 MLP 落到 seed 噪声内**（0.9922 vs 0.9922；0.9935 vs 0.9927）→ 真正的差别是「全局混合 vs 局部卷积」 |
| FGSM（C9） | ε=0.03 时 AUC 0.9890 → 0.4501；ε≥0.05 退化成「全判正常」 |
| 对抗训练（C9） | 用 −0.0081 干净 AUC 换 ε=0.03 处 +0.4033 鲁棒 AUC（崩溃填掉 79%）；保护范围只到 ε≈0.08 |
| Grad-CAM（C10） | 边框富集 0.78×，注意力偏肺野，未见边缘伪影作弊 |
| 剪枝（C14） | 剪 50% 只掉 0.005；但剪 80%（14,912）不如直接训 w16（4,914，0.9791） |
| 蒸馏（C14） | T=1/4/8 全部没赢从头训练（师生差距仅 0.010，软标签无额外信息） |
| 先验拆解（C11） | val→test 落差中先验只解释 11%，89% 是真实分布偏移 |

> 以上全部只评 val；`results/test_final.csv` 仍是唯一一次 test 记录，未被二次使用。

## 快速开始（在 `d2l-zh` 环境）

> **所有命令都必须在仓库根目录执行。** `src` 是本地包，`requirements.txt` / `configs/` / `results/` 都是相对路径；在家目录或任意别处运行会报 `No module named 'src'` 或 `Could not open requirements file`。

**PowerShell（本机）**：

```powershell
conda activate d2l-zh
cd "C:\Users\Zelvia\DeskBox\NOW_11\深度学习大作业1"
python -m pip install -r requirements.txt
python -m pip check

# 冒烟：1 个 epoch、只取少量样本，结果写到临时路径，不污染 results/
python -m src.run --config configs/base.yaml --epochs 1 --limit-train 256 --limit-val 128 --no-ckpt --out .workbuddy/smoke_runs.csv

# 正式（由本人执行，会写 results/runs.csv 与 outputs/base.pth）
python -m src.run --config configs/base.yaml
python -m src.sweep --config configs/sweep.yaml
python -m src.plots --results results/runs.csv

# 填好 configs/best.yaml 后：先训练出 outputs/best.pth，再 test 只评一次
python -m src.run --config configs/best.yaml
python -m src.run --config configs/best.yaml --split test
```

`src/plots.py` 在没有实验结果时会明确提示并退出，**不会生成示意数据或假图**。

### 常见启动错误

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `Could not open requirements file: ... 'requirements.txt'` | 当前目录不是仓库根 | `cd` 到仓库根目录再执行 |
| `No module named 'src'` | 同上；`src` 是本地包，Python 只从当前目录 / `PYTHONPATH` 里找 | 同上 |
| `pip check` 通过但上面的命令仍报错 | `pip check` 只检查**环境里已装包**的依赖一致性，与当前目录无关 | 它不能用来验证 cwd 是否正确 |

若从 GitHub clone 到别的目录，把 `cd` 换成 clone 后的实际路径；之后所有命令都保持在该目录下执行。用 `Get-Location`（PowerShell）或 `pwd`（Git Bash）可随时确认当前目录。

## 实验命令全记录（S1 → 交付）

> 以下命令已由本人本地执行完毕，结果见 `results/` 与 `figures/`。保留此处作为一键复现记录。
> 当初的推进逻辑：S1 单因子扫描只有 1 个 seed，差值可能小于种子波动，因此补做 S3 三 seed 复跑再下结论；实测差值约为 base 标准差的 23 倍，提升稳定（详见 `docs/FINAL_ANALYSIS.md`）。

```powershell
# 1. 整理 S1 数据（机器摘要，供填写 docs/REPORT.md 参考，非本人结论）
python -m src.analyze --results results/runs.csv --out docs/S1_ANALYSIS.md
python -m src.select_best --results results/runs.csv

# 2. S2 小网格（可选，看 lr × conv_width 的交互作用，4 组）
python -m src.repeat --mode grid --grid "lr=0.001,0.003;conv_width=32,64"

# 3. S3 三 seed 复跑（必做：base / 最优 / 次优 各 3 个 seed，9 组）
python -m src.repeat --mode seeds --results results/runs.csv
#    脚本末尾会打印均值 ± 标准差，据此判断提升是否显著

# 4. 若 S3 显示最优组稳定优于 base，从 runs.csv 取完整超参写出 best.yaml
#    （repeats.csv 只用于判断显著性，不含完整超参列，不要拿它选配置）
python -m src.select_best --results results/runs.csv --write   # --write 才写，仍需本人核对
python -m src.run --config configs/best.yaml                 # 训练出 outputs/best.pth + history
python -m src.run --config configs/best.yaml --split test   # test 只评一次，写 results/test_final.csv

# 5. 补齐图件（ROC / 混淆矩阵 / 预测样例 / loss 曲线）
python -m src.report --config configs/best.yaml --ckpt outputs/best.pth --split val --history outputs/best_history.json
python -m src.plots --results results/runs.csv --history outputs/best_history.json

# 6. 延伸实验（只评 val，不再碰 test）
python -m src.fgsm                    # ε 曲线
python -m src.gradcam                 # 可解释性
python -m src.compress                # 剪枝 + 蒸馏
python -m src.prior_shift             # 先验拆解
python -m src.adv_train               # 对抗训练 vs 干净训练（3 seed，约 22 min）
python -m src.capacity_match --dry-run    # 先看旋钮能否对齐到目标参数量
python -m src.capacity_match              # 2 档 × 3 结构 × 3 seed（约 39 min）
python -m src.capacity_match --replot     # 只由 CSV 重画图，不重训

# 7. 归档：填 docs/REPORT.md、LOG.md、PITFALLS.md、REVIEW.md 与章节映射表
git add results figures docs README.md
git commit -m "exp: S1-S3 results, figures and report"
git push origin main
```

`src/report.py` 用 `--split val` 出图，避免反复碰 test；`test_final.csv` 始终由 `src.run --split test` 独占写入，只应有一行。

## 中期验收三件套对照

| 验收项 | 本仓库当前提供 | 待补 |
| --- | --- | --- |
| 项目流程图 / 框图 | `docs/pipeline.svg` 独立文件；选题建议报告内另有一张五泳道 SVG；PPT 用同一张 | 已与代码结构核对 |
| 可访问的 GitHub 项目链接 | 本仓库（Public）；GitHub Pages 已开启 | — |
| 项目流程与代码运行效果 | 可运行 `src/` 骨架、`results/` 十一个 CSV、`figures/` 九张图 + `metrics.json`、`notebooks/demo.ipynb` | 现场演示由本人执行（15 分钟汇报 + 5 分钟提问） |

## 交付前检查清单

| # | 事项 | 状态 |
| --- | --- | --- |
| 1 | 流程图 / 框图入库，PPT 用同一张 | 已完成 |
| 2 | 仓库 Public、commit 已推送、Pages 可访问 | 已完成 |
| 3 | `results/` 十一个 CSV + `figures/` 九张图 + `metrics.json` 归档 | 已完成 |
| 4 | `configs/best.yaml` 定稿、`test_final.csv` 只有一行 | 已完成 |
| 5 | `docs/REPORT.md` 写结论与机制解释并署名 | 已完成（含 MLP 结构对照与 val/test 落差分析） |
| 6 | `docs/LOG.md` 实验日志、`docs/PITFALLS.md` 至少 2 条真实踩坑 | 已完成（日志 8 条、踩坑 15 条） |
| 7 | `docs/REVIEW.md` 三段式学习复盘 + 章节映射表 | 已完成（C1–C16 + D/E/F 全量映射） |
| 8 | 按 `docs/PPT_OUTLINE.md` 做 PPT，按 `docs/DEFENSE_QA.md` 预演答辩 | **待本人完成** |
| 9 | 现场演示（终端实跑 / `demo.ipynb` 单图推理） | **待本人完成** |
| 10 | 延伸实验跑通并归档（结构谱系 / FGSM / 对抗训练 / 容量对齐 / Grad-CAM / 剪枝蒸馏 / 先验拆解） | 已完成 |

> **答辩加分点**：`docs/EXTENSIONS.md` 里有四条负结果（剪枝不如训小的、蒸馏无增益、先验只解释 11%、对抗训练要拿干净精度换）和**一次自我修正**（容量对齐后「ViT 最优」不成立）。负结果配前提条件、旧结论能被自己的新实验推翻，比再刷 0.002 AUC 更能说明你真的跑懂了。

> **不能返工的两件事**：`test` 只评过一次，不要为了 S2 里更优的组合回头改 `best.yaml` 再评第二次；也不要拿 test 调阈值。两者都写进「遗留改进方向」并在答辩主动说明（见 `docs/DEFENSE_QA.md` Q4/Q5）。

## 如何在浏览器打开

- 直接点开仓库里的 `.html` 文件并选「Raw」附近预览，或 clone 到本地双击打开；
- 更推荐开启 GitHub Pages：仓库 **Settings → Pages → Build and deployment → Deploy from a branch → 选 `main` 分支、目录 `/`（root）→ Save**，稍等片刻即可通过 `https://via453.github.io/medmnist-tuning-lab/` 访问 `index.html` 导航页。

完整教程页面里的输入框、下拉框和检查清单会自动保存在**你自己的浏览器本地**（localStorage），不上传、不写入仓库；可用页面顶部按钮导出 JSON / Markdown 或打印成 PDF 归档。

## 环境与复现口径

- Conda 环境 `d2l-zh`：Python 3.9.25、torch 2.5.1+cpu、torchvision 0.20.1+cpu、d2l 0.17.6、Jupyter
- 依赖已锁定在 **numpy 1.x ABI 兼容组合**：numpy 1.21.5 / pandas 1.2.4 / matplotlib 3.5.1 / scipy 1.7.3 / scikit-learn 1.0.2 / scikit-image 0.19.3 / Pillow 9.5.0 / medmnist 3.0.2 / PyYAML / tqdm；`pip check` 返回无冲突，14 个关键包实测可导入
- **不要裸装** `pip install medmnist scikit-learn ...`：会经 `scipy` 把 numpy 顶到 2.x，触发 `d2l 0.17.6 requires numpy==1.21.5`，且 pandas / matplotlib / sklearn 因 C ABI 变化直接无法导入。复现请用钉版本命令（完整教程 3.1 / 附录 B）；更彻底的做法是给项目单独建 conda 环境
- 数据集：官方 API `download=True` 自动下载 `.npz` 到 `~/.medmnist/`（PneumoniaMNIST 约 4.2 MB，已实测 4708 / 524 / 624、28×28）；`data/` 不入库
- 安装与版本锁定见 `requirements.txt`（**不要用裸包名安装**）；`python -m pip check` 应返回 `No broken requirements found`
- 详细命令见完整教程「附录 B · 命令速查」

## 数据与学术口径

- MedMNIST 采用 CC 许可，官方声明**仅用于教育与研究，不用于临床**；本仓库不包含任何原始数据集文件。
- 所有实验结论以 val 选配置、test 一次评测、3 seed 均值 ± 标准差为准；若提升不显著，结论改写为「超参敏感性分析」。
- 训练与调参链路须由本人本地执行并署名，过程记录见完整教程第 6、7、9 节。

## 参考

- [MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST) · [MedMNIST/experiments](https://github.com/MedMNIST/experiments) · [medmnist.com](https://medmnist.com/)
- [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
