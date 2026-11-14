# 中期汇报 PPT 大纲（建议稿）

> 可按实际汇报习惯调整。培训方案要求 **15 分钟汇报 + 5 分钟提问**，建议 14–15 页、每页 1 分钟左右，重点放在「流程图 + 实验设计 + 结果 + 现场演示」。所有数字以 `docs/FINAL_ANALYSIS.md` 与 `results/*.csv` 为准。

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
| 12 | 延伸实验一：结构谱系 + **容量对齐后的自我修正**（CNN 落后是结构性的；ViT 与 MLP 打平） | 2 min | `figures/capacity_curve.png`、`results/capacity_summary.csv` |
| 13 | 延伸实验二：FGSM 崩溃曲线 + **对抗训练换来的鲁棒性**（ε=0.03 处 0.47→0.87，干净 −0.008） | 2 min | `figures/fgsm_adv_compare.png`、`results/adv_curve.csv` |
| 14 | 结论、遗留方向（S2 更优组合未采纳、PGD 对抗训练未做）、学习复盘与章节映射 | 1 min | `docs/REVIEW.md`、教程第 7 节 |
| 15 | 仓库与一键复现：README、requirements、tag、commit | 0.5 min | README |

> 第 12/13 页可以合成一页「延伸实验的两条主线」，但**容量对齐那条一定要讲**：它是全场唯一一处「我用新实验推翻了自己上一轮结论」，比多刷 0.002 AUC 有说服力。

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
- `docs/REPORT.md`、`LOG.md`、`PITFALLS.md`、`REVIEW.md` 已填写并署名。
- 章节映射表已从教程第 7 节导出并贴入 `docs/REVIEW.md`。
- 仓库为 Public、commit 已推送、Pages 可访问。
- 预测题准备好：`docs/DEFENSE_QA.md`。

---

## 第二版（2026-11-14，`MedMNIST调参实验室_汇报_v2`）

第一版按「项目流程」组织，读起来容易变成流水账。第二版改成四个问题，并把带图的页从 8 页扩到 9 页、每页最多两张：

| 页 | 内容 | 图 |
| --- | --- | --- |
| 1 | 封面：做了什么会让模型崩 | — |
| 2 | 四个问题：做了什么 / 为什么 / 结果 / 收获 | — |
| 3 | 五步 + 三个数（79 次训练、46 组对照、1 次测试） | — |
| 4 | 为什么分三份数据（练习册 / 模拟考 / 高考） | — |
| 5 | 为什么写三种模型（三种看图方式） | 自绘 SVG |
| 6 | 为什么先单因子扫描 | `hp_sensitivity.png` |
| 7 | 结果一：调参有用但幅度有限 | `before_after.png` + `loss_curves.png` |
| 8 | 结果二：结构比旋钮更管用（容量对齐） | `capacity_curve.png` |
| 9 | 结果三：三份成绩单 + 错例 | `samples.png` |
| 10 | 结果四：分数高不等于靠得住 | `fgsm_curve.png` |
| 11 | 结果五：掺脏图能救回八成 | `fgsm_adv_compare.png` |
| 12 | 结果六：它在看片子的哪里 | `gradcam.png` |
| 13 | 结果七：落差 12 个点从哪来 | `roc.png` + `confusion_matrix.png` |
| 14 | 三处自我修正 | — |
| 15 | 三个习惯 + 交付清单 | — |

两版的取舍：v1 适合「按流程讲」，v2 适合「按问题讲」。答辩用 v2，备查用 v1。
