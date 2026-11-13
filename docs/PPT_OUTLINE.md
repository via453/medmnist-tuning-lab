# 中期汇报 PPT 大纲（建议稿）

> 由本人按实际汇报习惯调整。培训方案要求 **15 分钟汇报 + 5 分钟提问**，建议 12–14 页、每页 1 分钟左右，重点放在「流程图 + 实验设计 + 结果 + 现场演示」。所有数字以 `docs/FINAL_ANALYSIS.md` 与 `results/*.csv` 为准。

## 页面与时间分配

| 页 | 内容 | 建议用时 | 素材 |
| --- | --- | --- | --- |
| 1 | 封面：题目、姓名、日期、**GitHub 仓库链接** | 0.5 min | README |
| 2 | 验收要求 ↔ 本项目交付（三件套对照表） | 1 min | README 三件套表 |
| 3 | 任务与数据：PneumoniaMNIST 5856 张、28×28、4708/524/624、CC BY 4.0、**不用于临床** | 1 min | `docs/S1_ANALYSIS.md` |
| 4 | **项目流程图 / 框图**（指图讲 60 秒，说明虚线是 val 调参不是 test） | 1.5 min | `docs/pipeline.svg` |
| 5 | 基础模型：MLP → SmallCNN，参数量、light pooling 与 C3 笔记的关系 | 1.5 min | `src/model.py` |
| 6 | 实验设计：S1 单因子 33 组 → S2 网格 4 组 → S3 三 seed 9 组 → test 一次 | 2 min | `configs/sweep.yaml` |
| 7 | 调参结果：超参敏感性（lr / optimizer / BN 最敏感，其余几乎不敏感） | 1.5 min | `figures/hp_sensitivity.png` |
| 8 | **S3 显著性**：均值 ± 标准差表，差值 ≈ base std 的 23 倍 | 1.5 min | `results/repeats.csv` |
| 9 | 最终配置与训练曲线：最佳轮次而非末轮、loss 与 val AUC | 1 min | `figures/loss_curves.png` |
| 10 | test 结果与 **val/test 落差**：ROC、混淆矩阵、C11 分布偏移解释 | 2 min | `figures/roc.png`、`confusion_matrix.png` |
| 11 | 预测样例 + 现场演示（切到终端实跑一条命令） | 1.5 min | `figures/samples.png`、`notebooks/demo.ipynb` |
| 12 | 结论、遗留方向（S2 更优组合未采纳）、学习复盘与章节映射 | 1 min | `docs/REVIEW.md`、教程第 7 节 |
| 13 | 仓库与一键复现：README、requirements、tag、commit | 0.5 min | README |

## 每页必须落地的三件事

1. **第 2 页就把三件套点名**：流程图（第 4 页）、GitHub 链接（第 1/13 页）、运行效果（第 11 页现场跑）。
2. **第 4 页的框图与代码结构一致**：数据 → 预处理 → SmallCNN → 训练/验证 → 调参环 → test 一次 → 出图。
3. **第 11 页真跑，不放静态截图**：`python -m src.run --config configs/best.yaml --split test` 会覆盖 `test_final.csv`，**汇报现场不要重跑这条**；改用 `python -m src.report ... --split val` 或 `demo.ipynb` 演示，避免二次评测 test。

## 演示脚本（现场，按实际情况二选一）

```powershell
# 方案 A：终端一条命令，展示真实推理与指标
python -m src.report --config configs/best.yaml --ckpt outputs/best.pth --split val --history outputs/best_history.json

# 方案 B：Jupyter 单图推理（需从仓库根目录启动 jupyter）
jupyter notebook notebooks/demo.ipynb
```

## 答辩前最后核对

- `results/test_final.csv` 只有一行，生成时间晚于所有调参实验。
- `docs/REPORT.md`、`LOG.md`、`PITFALLS.md`、`REVIEW.md` 已由本人填写并署名。
- 章节映射表已从教程第 7 节导出并贴入 `docs/REVIEW.md`。
- 仓库为 Public、commit 已推送、Pages 可访问。
- 预测题准备好：`docs/DEFENSE_QA.md`。
