# 实验日志

> 由本人本地执行实验时逐行填写。`src/run.py` 与 `src/sweep.py` 会自动把结构化字段写进 `results/runs.csv`；本文件用于记录**机器字段之外的判断与现象**。

## 记录规则

- 每组实验（含失败组）都要有一行，**失败组不许删除**。
- 用 val 选配置，`results/test_final.csv` 全程只应有一行。
- 每次实验记下 `git rev-parse --short HEAD`，把结果锁到具体代码版本。
- 日期用 `YYYY-MM-DD HH:MM`，与 CSV 的 `timestamp` 一致。

## 日志条目模板

```
### YYYY-MM-DD HH:MM · run_id: s1_01
- 配置：configs/base.yaml，改动字段 / 取值：
- 环境：d2l-zh / torch 2.5.1+cpu / numpy 1.21.5 / CPU
- 命令：
- 现象（loss 曲线、是否发散、是否早停、耗时）：
- 判断（下一步动哪个超参，为什么）：
- git commit：
```

## 环境问题记录（已完成）

| 日期 | 现象 | 处理 |
| --- | --- | --- |
| 2026-11-13 | `d2l 0.17.6 requires numpy==1.21.5, but you have numpy 2.0.2`；pandas / matplotlib / sklearn 因 NumPy 2 C ABI 变化无法导入 | 按 `requirements.txt` 把整套依赖锁回 numpy 1.x ABI；`pip check` 无冲突，冒烟测试通过 |

## 条目

（待本人填写）
