# DESIGN.md — MedMNIST 医学影像分类调参实验室 · 中期验收汇报

## 一、全局视觉约定

### ① 画布与安全区
- 画布 1280×720
- 内容页：顶条 72px，内容区左右 padding 60px、上 22px、下 54px（内容可用高 572px），页脚固定 y=670
- 封面 / 结尾页自定义版式，不加顶条与页脚

### ② 配色（学术宝蓝 + 白底，来自 academic.md §1.2 兜底）
| 槽位 | Hex | 用途 |
| --- | --- | --- |
| 主色 | `#1E4FA8` | 顶条、编号块、表头底线、强调词 |
| 辅色 | `#3D7BD9` / `#E8EFF8` | 图注、结论条底、第二序列 |
| 强调色 | `#D9534F` | 关键差值、失效点，全篇每页 ≤2 处 |
| 正文 | `#1A2230` / `#4A5568` / `#8B97A8` | 主 / 副 / 辅助 |
| 底色 | `#FFFFFF` | 白，≥60% |
| 分隔 | `#D6DCE5` | 1px 细线、表格线 |

纪律：蓝族 20–40%，红 ≤3%，金色 0，无紫/粉/渐变。

### ③ 字体系统
- 中文：思源黑体（回退 Microsoft YaHei）；英文数字：Arial
- 阶梯：封面主标 60 / 页主标 32 Bold / 卡内小标 24 Bold / 正文 20 / 表格 18 / 图注与页脚 14 / KPI 64 Bold / 章节英文 16

### ④ 视觉母题
- 每页顶条左标题 + 右侧「NN · English Section」编号（16px，`#C7D8F2`）
- 结论文本统一压在页底浅蓝条（`#E8EFF8`，高 52–56px）里，只有一条

### ⑤ 配图清单
| 文件 | 等级 | 来源 | 真实内容 | 使用页 |
| --- | --- | --- | --- | --- |
| `assets/sensitivity.png` | L1 | 已有实验图 | 10 个超参的 val AUC 波动范围 | P7 |
| `assets/loss.png` | L2 | 已有实验图 | 训练 loss 与 val AUC 双曲线 | P9 |
| `assets/roc.png` | L2 | 已有实验图 | test ROC，AUC 0.9546 | P10 |
| `assets/cm.png` | L2 | 已有实验图 | test 混淆矩阵 118/17/8/381 | P10 |
| `assets/samples.png` | L1 | 已有实验图 | 8 张预测样例与概率 | P11 |
| `assets/capacity.png` | L1 | 本轮新实验 | 参数量-精度曲线，三种结构两档对齐 | P12 |
| `assets/adv.png` | L1 | 本轮新实验 | FGSM ε 曲线，干净 vs 对抗训练 | P13 |
| `assets/before_after.png` | L2 | 已有实验图 | base 与 w64 的对比 | P8 |

## 二、逐页字段

### P1 封面
- type: cover / role: hero / rhythm: peak / layout: 左对齐大字封面
- visual: 无图，纯排版 + 一条主色细线；L3 无
- density: 字数约 60 / 图片 0 / 留白约 45%
- anti_pattern: 禁止封面放指标数字；禁止居中对称构图；禁止英文大标题与中文主标抢层级

### P2 验收要求 ↔ 交付
- type: content / role: supporting / rhythm: valley / layout: 三行对照表 + 底部结论条
- visual: L3（表头主色底线）
- density: 字数约 210 / 表格 1 / 留白约 25%
- anti_pattern: 禁止三个等宽卡片的「三件套」装饰化排版；禁止表格整行填色

### P3 任务与数据
- type: content / role: supporting / rhythm: valley / layout: 左文右三联划分
- visual: L3（数字卡）
- density: 字数约 190 / 留白约 25%
- anti_pattern: 禁止把数据集介绍写成维基百科；禁止出现没有判断的裸数字

### P4 项目流程
- type: content / role: hero / rhythm: peak / layout: 五阶段流程链 + 反馈虚线段 + 输出条
- visual: 原生 SVG 流程链，占内容区 ≥70%
- density: 字数约 160 / 图 1 / 留白约 20%
- anti_pattern: 禁止画成软件架构图；禁止把 test 画进循环回路

### P5 模型与感受野
- type: content / role: supporting / rhythm: valley / layout: 左右两栏（结构 vs 感受野示意）
- visual: L2 自绘 28×28 与 9×9 感受野示意
- density: 字数约 180 / 图 1 / 留白约 22%
- anti_pattern: 禁止罗列全部超参；禁止三模型并列（ViT 留到 P12）

### P6 实验设计
- type: content / role: supporting / rhythm: valley / layout: 四阶段表 + 纪律条
- density: 字数约 200 / 表格 1 / 留白约 22%
- anti_pattern: 禁止把 33 组逐条列出；禁止用「科学严谨」这类形容词代替判定标准

### P7 超参敏感性
- type: content / role: evidence / rhythm: valley / layout: 左图右排序
- visual: L1 `sensitivity.png` 占左 52%
- density: 字数约 150 / 图 1 / 留白约 22%
- anti_pattern: 禁止只贴图不给排序；禁止把七项不敏感的超参也逐条讲

### P8 S3 显著性
- type: content / role: supporting / rhythm: valley / layout: 三列均值表 + 配对差值
- visual: L2 `before_after.png`（右下）
- density: 字数约 170 / 图 1 / 留白约 22%
- anti_pattern: 禁止只报均值不报 std；禁止用「显著提升」而不给倍数

### P9 最终配置与训练曲线
- type: content / role: supporting / rhythm: valley / layout: 左配置清单 右曲线
- visual: L2 `loss.png`
- density: 字数约 160 / 图 1 / 留白约 25%
- anti_pattern: 禁止把 best.yaml 整份贴上来；禁止回避「最佳轮 ≠ 末轮」

### P10 test 与落差
- type: content / role: hero / rhythm: peak / layout: 双图并排 + 落差拆解条
- visual: L2 `roc.png` + `cm.png`
- density: 字数约 170 / 图 2 / 留白约 18%
- anti_pattern: 禁止把落差说成 bug；禁止建议「按 test 调阈值」

### P11 预测样例与现场演示
- type: content / role: supporting / rhythm: valley / layout: 左样例图 右命令块
- visual: L1 `samples.png` 占左 45%
- density: 字数约 140 / 图 1 / 代码 1 / 留白约 22%
- anti_pattern: 禁止演示命令里出现 `--split test`；禁止放静态截图代替实跑

### P12 延伸一：结构谱系与容量对齐
- type: content / role: hero / rhythm: peak / layout: 左两档数值表 右曲线
- visual: L1 `capacity.png` 占右 50%
- density: 字数约 190 / 图 1 / 留白约 20%
- anti_pattern: 禁止只说「ViT 最好」；禁止隐藏自我修正

### P13 延伸二：对抗训练
- type: content / role: hero / rhythm: peak / layout: 宽图在上 + 三数在下
- visual: L1 `adv.png` 占满宽（2.5:1）
- density: 字数约 170 / 图 1 / 留白约 18%
- anti_pattern: 禁止把 +0.40 说成免费；禁止不提 ε≥0.2 仍失效

### P14 结论与局限
- type: content / role: supporting / rhythm: valley / layout: 左结论五条 右局限与后续
- density: 字数约 240 / 留白约 20%
- anti_pattern: 禁止「重大突破」式措辞；禁止把局限写成展望

### P15 仓库与复现
- type: content / role: supporting / rhythm: valley / layout: 左仓库信息 右三条命令
- density: 字数约 160 / 代码 2 / 留白约 25%
- anti_pattern: 禁止把结尾写成致谢页；禁止出现「谢谢聆听」类内容
