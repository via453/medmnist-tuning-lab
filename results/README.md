# results/ · 实验记录

本目录存放**由本人本地跑出来的真实结果**。仓库第一版只提供骨架，不含任何实验数据。

| 文件 | 由谁生成 | 说明 |
| --- | --- | --- |
| `runs.csv` | `src/run.py` / `src/sweep.py` | S1 单因子扫描，每组配置一行，含成功与失败组；字段见完整教程 6.1 |
| `grid.csv` | `src/repeat.py --mode grid` | S2 小网格，看超参交互作用（可选） |
| `repeats.csv` | `src/repeat.py --mode seeds` | S3 三 seed 复跑，用于算均值 ± 标准差、判断提升是否显著 |
| `structure.csv` | `src/run.py`（`model: mlp` / `vit`） | 结构对照：MLP 与 TinyViT 各三个 seed 的 val 指标，用于「换模型 ≠ 调参功劳」那一节 |
| `adv_train.csv` | `src/adv_train.py` | 对抗训练对照：同一 seed 下 clean / adv 各一次训练，含干净 AUC 与 ε=0.03 处 AUC |
| `adv_curve.csv` | `src/adv_train.py` | ε 曲线按 3 个 seed 聚合的均值 ± std，对应 `figures/fgsm_adv_compare.png` |
| `capacity.csv` | `src/capacity_match.py` | 容量对齐：每个 (目标参数量, 结构, seed) 一行，含实际旋钮取值与参数量 |
| `capacity_summary.csv` | `src/capacity_match.py` | 上面的均值 ± std 汇总，`--replot` 直接读它画 `figures/capacity_curve.png` |
| `ablation.csv` | 本人整理 | 消融表，也可见 `docs/REPORT.md` 第 2、3 节 |
| `test_final.csv` | `src/run.py --split test` | **只应有一行**，覆盖写；生成时间必须晚于所有调参实验 |

约定：

- 失败组也要留在 `runs.csv`，不许删。
- `test_final.csv` 只评一次，重复运行会覆盖，若确需重跑必须在 `docs/LOG.md` 写明原因。
- 图件由 `python -m src.plots --results results/runs.csv` 从这些 CSV 现场生成，不放手工截图。
