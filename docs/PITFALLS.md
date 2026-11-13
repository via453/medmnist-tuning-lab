# 踩坑清单

> 只记**真实遇到**并已定位的坑。每条都给出可复核的证据（run_id / 文件 / 数值）。
> 「现象」写看到的东西，「原因」写查证后的机制，两者分开，避免把猜测当结论。

## 一、真实踩坑（10 条）

| # | 日期 | 现象 | 原因 | 处理方式 | 对应已学内容 |
| --- | --- | --- | --- | --- | --- |
| 1 | 11-13 | `pip install medmnist scikit-learn` 后报 `d2l 0.17.6 requires numpy==1.21.5, but you have numpy 2.0.2`，且 `pandas` / `matplotlib` / `sklearn` 直接导入失败（`numpy.dtype size changed`、`_ARRAY_API not found`） | 装包时 `scipy 1.13.1` 要求 `numpy>=1.22.4`，把 numpy 从 1.21.5 顶到 2.0.2；NumPy 2 改了 C ABI，而 pandas 1.2.4 / matplotlib 3.5.1 是编译到 1.x 的二进制扩展 | 不能只忽略警告：按 `requirements.txt` 把 numpy / scipy / sklearn / scikit-image / Pillow 一起锁回 1.x 兼容组合，`pip check` 确认无冲突，14 个包全部可导入 | 环境与版本管理（A 段）；「依赖是树不是点」 |
| 2 | 11-13 | 在 `C:\Users\Zelvia` 执行 `python -m src.run` 报 `No module named 'src'`；`pip install -r requirements.txt` 报 `Could not open requirements file` | 两条命令都是**相对路径 / 本地包**：`src` 只在当前目录被 Python 搜索，`requirements.txt` 也按 cwd 解析。而当时 cwd 是家目录不是仓库根 | 所有项目级命令前先 `cd` 到仓库根；区分「环境级命令（任意目录可跑，如 `pip check`）」与「项目级命令（必须在仓库根）」 | 环境级 vs 项目级命令的区别；**`pip check` 通过不能证明 cwd 正确** |
| 3 | 11-13 | S1 单 seed 显示最优 `conv_width=64` 比 base 高 0.0046，差点直接写进报告 | 0.0046 **小于 lr 组自身极差 0.022**，单次运行的差值完全可能来自初始化/数据顺序 | 补 S3 三 seed 复跑：差值 +0.004107 ≈ 19 倍 base 标准差且逐 seed 同向，才敢写「提升」 | C1 用验证集做模型选择；**「差值必须与噪声尺度对比」** |
| 4 | 11-13 | `s1_33` 完全不池化反而从 0.9844 掉到 0.9782，与笔记 C3 订正版「微小结节会在池化中丢失」相反 | 28×28 上去掉池化后感受野利用效率下降，且 GAP 前特征图带空间噪声；单次 2×2 池化实际起了轻量正则作用 | 把结论改成有边界条件的版本：**「不堆池化」成立，「完全不用池化」不成立**；写进报告 3.4 节 | C3 卷积与池化；**订正版结论也要给适用区间** |
| 5 | 11-13 | 加数据增强反而更差：不增强 0.9851 > flip 0.9844 > flip+旋转 0.9821 | 旋转把肺野推到画布边缘并引入黑边填充，是**分布外变换**；小数据上它带来的分布偏移大于正则收益。flip 理论安全但收益也被噪声淹没 | 默认保留 flip（无害、与官方基线一致），去掉旋转；把「增强不是越多越好」写成实测结论 | C2 正则化；**增强必须保持标签语义与数据分布** |
| 6 | 11-13 | `s1_25` `class_weight=balanced` 后 AUC 升到 0.9856（最高之一），但 ACC 掉到 0.8760，耗时还从 3.3 min 涨到 6.0 min | 类别权重不改变样本的**排序**（AUC 基本不动），但把决策阈值往少数类方向推；本数据 val 阳性占 74%，加权后大量阴性被判成阳性 → ACC 掉、召回升 | 主指标锁定 AUC，不用 ACC 判断权重是否有效；若关心 ACC/F1 应显式做阈值搜索而不是靠 class_weight | C1 指标选择；**AUC 与 ACC 度量的是不同东西** |
| 7 | 11-13 | 以为 `python -m src.run` 跑完存的是最后一轮模型，核对时发现末轮 val AUC 只有 0.9837，低于记录里的 0.9890 | `train.py` 保存的是**val AUC 最高的那一轮**（epoch 19），不是 epoch 20；末轮已轻度过拟合 | 用 `figures/metrics.json` 与 CSV 的 best 行交叉核对，确认权重与记录一致；报告里显式写明「权重取第 19 轮」 | C1 早停与模型选择；**「跑完」≠「最好的那个」** |
| 8 | 11-13 | test 结果比 val 差一大截：AUC 0.9890→0.9546、ACC 0.9523→0.8349，假阳性 17→100 | 不是代码 bug：MedMNIST 的 test 来自**独立 source validation set**（train/val 是同源 9:1 切分），且阳性率 74.2%→62.5%，固定阈值 0.5 在低先验下偏激进 | 按纪律不回头调阈值、不换配置重评；把落差作为协变量偏移的实证写入报告 4.3 节，答辩主动讲 | C11 领域自适应 / 协变量偏移；**「指标掉」先查分布再查代码** |
| 9 | 11-13 | 本地 `git status` 显示 `ahead 1`，远端文件内容却和本地一模一样；逐文件比对发现 8 个脚本生成的文件 blob sha 不同 | Windows 工作区是 CRLF、git 提交时按 `autocrlf` 转成 LF；上传时若直接读工作区原始字节，blob 就与提交里的不一致 | 上传前按仓库规则做行尾规范化（或 `git show HEAD:<path>` 取提交内容），再重建 tree | 版本控制细节；**「看到的内容一样」不等于「对象一样」** |
| 10 | 11-13 | `git push` 偶发 `CONNECT tunnel failed, response 502`，`github.com` 不通但 `api.github.com` 正常 | 本地代理对 github.com 的 git 端点转发失败（直连 / IPv4 / ssh:443 均不通），与仓库权限无关 | 能直连时正常 `git push`；不通时走 Git Data API 提交，事后 `git fetch origin && git reset --soft origin/main` 对齐历史（`--soft` 不动工作树） | 不要把网络故障误判成认证/权限问题 |

## 二、已预先标注但本项目**未**触发的坑（保留备查）

以下是在教程里提前写好的高频风险，本轮实际没遇到，留着下次省时间：

- **BN 模式切错**：`model.train()` / `model.eval()` 混用会让验证指标失真。本轮 `evaluate.predict()` 内已显式 `model.eval()`，未出现。
- **Windows 多进程**：`num_workers>0` 触发 spawn 报错。全程 `num_workers=0`，未出现。
- **medmnist 版本 API 漂移**：`Evaluator` / `root` 参数各版本不同。本轮先用 `inspect.signature` 确认再调用，未出现。
- **种子只设一处**：`random` / `numpy` / `torch` 三处都要设。本轮 13 次等价 base 全部精确复现 0.984404，反证三处都设对了。
- **测试集纪律**：用 test 反复调参等于泄题。`results/test_final.csv` 全程只有 1 行。

## 三、从这十条里提炼的三条硬规则

1. **任何「变好了」都要先问噪声有多大。** 单 seed 差值必须与 seed 标准差、与同组极差对比；否则就是自欺（第 3 条）。
2. **指标掉了先查数据分布，再查代码。** val/test 落差 90% 是来源差异，不是 bug（第 8 条）。
3. **环境报错先分清是「环境坏了」还是「当前位置不对」。** `pip check` 只能证明前者（第 2 条）。
