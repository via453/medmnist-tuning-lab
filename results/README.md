# results/ · 实验记录

本目录存放**由本人本地跑出来的真实结果**。仓库第一版只提供骨架，不含任何实验数据。

| 文件 | 由谁生成 | 说明 |
| --- | --- | --- |
| `runs.csv` | `src/run.py` / `src/sweep.py` | 每组配置一行，含成功与失败组；字段见完整教程 6.1 |
| `ablation.csv` | 本人整理 | 消融表，也可见 `docs/REPORT.md` 第 2、3 节 |
| `test_final.csv` | `src/run.py --split test` | **只应有一行**，覆盖写；生成时间必须晚于所有调参实验 |

约定：

- 失败组也要留在 `runs.csv`，不许删。
- `test_final.csv` 只评一次，重复运行会覆盖，若确需重跑必须在 `docs/LOG.md` 写明原因。
- 图件由 `python -m src.plots --results results/runs.csv` 从这些 CSV 现场生成，不放手工截图。
