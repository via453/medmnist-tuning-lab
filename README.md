# MedMNIST 医学影像分类调参实验室

中期大作业验收资料库：**选题论证 + 项目全生命周期教程 + 协作边界 + 可运行代码骨架**。

> **当前状态（2026-11-13，第一版骨架）**：文档、配置与训练链路代码已就位，并在本机通过冒烟验证（`src/run.py` 与 `src/sweep.py` 可跑通）。**仓库内不含任何实验结果、训练权重或数据集文件**——按协作边界，训练、调参、结果记录与现场演示必须由本人本地执行并署名。

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
| `src/` | **代码骨架**：`data` / `model` / `train` / `evaluate` / `run` / `sweep` / `plots` / `repeat`（S2·S3）/ `report`（ROC·混淆矩阵·预测样例）/ `analyze`（数据摘要）/ `select_best`（选优）/ `utils` |
| `configs/` | `base.yaml`（基线）、`sweep.yaml`（单因子扫描）、`best.yaml`（待本人跑完调参后填写） |
| `docs/pipeline.svg` | 项目流程图 / 框图，可直接用于 README 与 PPT |
| `docs/*.md` | 实验日志、踩坑清单、实验报告、学习复盘模板；`S1_ANALYSIS.md` 由 `src/analyze.py` 自动整理（参考用，非结论） |
| `requirements.txt` | 钉死 numpy 1.x ABI 兼容组合，防止依赖冲突复发 |
| `notebooks/demo.ipynb` | 单图推理演示（需先有本地训练出的权重） |

建议阅读顺序：选题建议 → 完整教程 → 详细教程（按需查阅）。

## 目录结构

```
medmnist-tuning-lab/
├── README.md / index.html / LICENSE（待确认）
├── requirements.txt / .gitignore
├── configs/     base.yaml · sweep.yaml · best.yaml
├── src/         data · model · train · evaluate · run · sweep · plots · utils
├── docs/        pipeline.svg · LOG.md · PITFALLS.md · REPORT.md · REVIEW.md
├── results/     由本人跑出的 runs.csv / ablation.csv / test_final.csv（当前为空）
├── figures/     由 src/plots.py 生成的图件（当前为空）
├── notebooks/   demo.ipynb
├── data/        .npz 自动下载，已忽略
└── outputs/     训练权重，已忽略
```

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

## 下一阶段：从 S1 到交付

> S1 单因子扫描只跑 **1 个 seed**。最优组相对 base 的差值若小于种子波动，不能写成「调参提升了精度」——按教程第 5 节改写成「超参敏感性分析」。下面是完整推进顺序。

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

# 6. 归档：填 docs/REPORT.md、LOG.md、PITFALLS.md、REVIEW.md 与章节映射表
git add results figures docs README.md
git commit -m "exp: S1-S3 results, figures and report"
git push origin main
```

`src/report.py` 用 `--split val` 出图，避免反复碰 test；`test_final.csv` 始终由 `src.run --split test` 独占写入，只应有一行。

## 中期验收三件套对照

| 验收项 | 本仓库当前提供 | 待补 |
| --- | --- | --- |
| 项目流程图 / 框图 | `docs/pipeline.svg` 独立文件；选题建议报告内另有一张五泳道 SVG | 与最终代码结构同步核对 |
| 可访问的 GitHub 项目链接 | 本仓库（Public）；GitHub Pages 已开启 | — |
| 项目流程与代码运行效果 | 完整教程命令 + 可运行 `src/` 骨架 + 冒烟验证通过 | `results/`、`figures/` 真实实验产出、单图推理 demo 现场演示 |

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
