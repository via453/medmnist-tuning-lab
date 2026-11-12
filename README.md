# MedMNIST 医学影像分类调参实验室

中期大作业验收资料库：**选题论证 + 项目全生命周期教程 + 协作边界**。

> **当前状态（2026-11-13）**：文档与设计阶段已完成，训练代码、配置与实验结果待按教程补齐。本仓库目前只含 HTML 文档，不含已训练模型、实验数据或数据集文件。

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

建议阅读顺序：选题建议 → 完整教程 → 详细教程（按需查阅）。

## 中期验收三件套对照

| 验收项 | 本仓库当前提供 | 待补 |
| --- | --- | --- |
| 项目流程图 / 框图 | 选题建议报告内已含 SVG 五泳道流程图，可直接用于 README 与 PPT | 与最终代码结构同步核对 |
| 可访问的 GitHub 项目链接 | 本仓库（Public） | — |
| 项目流程与代码运行效果 | 完整教程给出完整命令、目录结构与检查清单 | `src/`、`configs/`、`results/`、`figures/`、单图推理 demo |

## 如何在浏览器打开

- 直接点开仓库里的 `.html` 文件并选「Raw」附近预览，或 clone 到本地双击打开；
- 更推荐开启 GitHub Pages：仓库 **Settings → Pages → Build and deployment → Deploy from a branch → 选 `main` 分支、目录 `/`（root）→ Save**，稍等片刻即可通过 `https://via453.github.io/medmnist-tuning-lab/` 访问 `index.html` 导航页。

完整教程页面里的输入框、下拉框和检查清单会自动保存在**你自己的浏览器本地**（localStorage），不上传、不写入仓库；可用页面顶部按钮导出 JSON / Markdown 或打印成 PDF 归档。

## 环境与复现口径

- Conda 环境 `d2l-zh`：Python 3.9.25、torch 2.5.1+cpu、torchvision 0.20.1+cpu、d2l 0.17.6、Jupyter
- 依赖已锁定在 **numpy 1.x ABI 兼容组合**：numpy 1.21.5 / pandas 1.2.4 / matplotlib 3.5.1 / scipy 1.7.3 / scikit-learn 1.0.2 / scikit-image 0.19.3 / Pillow 9.5.0 / medmnist 3.0.2 / PyYAML / tqdm；`pip check` 返回无冲突，14 个关键包实测可导入
- **不要裸装** `pip install medmnist scikit-learn ...`：会经 `scipy` 把 numpy 顶到 2.x，触发 `d2l 0.17.6 requires numpy==1.21.5`，且 pandas / matplotlib / sklearn 因 C ABI 变化直接无法导入。复现请用钉版本命令（完整教程 3.1 / 附录 B）；更彻底的做法是给项目单独建 conda 环境
- 数据集：官方 API `download=True` 自动下载 `.npz` 到 `~/.medmnist/`（PneumoniaMNIST 约 4.2 MB，已实测 4708 / 524 / 624、28×28）；`data/` 不入库
- 详细命令见完整教程「附录 B · 命令速查」

## 数据与学术口径

- MedMNIST 采用 CC 许可，官方声明**仅用于教育与研究，不用于临床**；本仓库不包含任何原始数据集文件。
- 所有实验结论以 val 选配置、test 一次评测、3 seed 均值 ± 标准差为准；若提升不显著，结论改写为「超参敏感性分析」。
- 训练与调参链路须由本人本地执行并署名，过程记录见完整教程第 6、7、9 节。

## 参考

- [MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST) · [MedMNIST/experiments](https://github.com/MedMNIST/experiments) · [medmnist.com](https://medmnist.com/)
- [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
