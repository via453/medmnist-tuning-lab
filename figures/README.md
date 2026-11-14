# figures/ · 图件

本目录存放**由脚本从结果数据生成的图**，以及两张流程图（`docs/pipeline.svg` 与 `docs/pipeline_roadmap.svg`）。

| 文件 | 来源 | 状态 |
| --- | --- | --- |
| `../docs/pipeline.svg` | 仓库提供 | 已有，可直接用于 README 与 PPT |
| `hp_sensitivity.png` | `src/plots.py` | S1 扫描后由 `src/plots.py` 生成 |
| `before_after.png` | `src/plots.py` | S1 扫描后由 `src/plots.py` 生成 |
| `loss_curves.png` | `src/plots.py --history outputs/best_history.json` | 待最优配置训练后生成 |
| `roc.png` / `confusion_matrix.png` / `samples.png` | `src/report.py --ckpt outputs/best.pth --split val` | 待最优配置训练后生成 |
| `metrics.json` | `src/report.py` | 与上面三张图同源，记录 AUC / ACC / F1 / 混淆矩阵 |
| `pipeline_roadmap.png` | Chrome 渲染 `docs/pipeline_roadmap.svg`（2×） | 技术路线图：六段主干 + 六个追问 + val 回路 |

约定：所有图都由 `results/*.csv` 或训练历史现场生成；`src/plots.py` 在无数据时会明确提示并退出，**不会生成示意数据或假图**。
