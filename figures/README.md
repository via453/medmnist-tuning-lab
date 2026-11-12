# figures/ · 图件

本目录存放**由脚本从结果数据生成的图**，以及项目流程图。第一版没有实验结果图。

| 文件 | 来源 | 状态 |
| --- | --- | --- |
| `../docs/pipeline.svg` | 仓库提供 | 已有，可直接用于 README 与 PPT |
| `hp_sensitivity.png` | `src/plots.py` | 待本人跑完 sweep 后生成 |
| `before_after.png` | `src/plots.py` | 待本人跑完 sweep 后生成 |
| `loss_curves.png` / `roc.png` / `confusion_matrix.png` | 后续实验脚本 | 待补 |
| `samples.png` | 数据冒烟脚本 | 待补 |

约定：所有图都由 `results/*.csv` 或训练历史现场生成；`src/plots.py` 在无数据时会明确提示并退出，**不会生成示意数据或假图**。
