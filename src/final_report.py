"""把 S1/S2/S3/test/val 全部结果整理成交付摘要（只给数据，结论写在 REPORT.md）。

    python -m src.final_report --out docs/FINAL_ANALYSIS.md

只呈现数据与可核对的事实，不做结论性判断。最终结论与解释写在
docs/REPORT.md；本文件可作为填写报告与准备答辩的素材。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd


def read_csv(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    return df


def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v, n=6):
    try:
        return ("%." + str(n) + "f") % float(v)
    except (TypeError, ValueError):
        return "n/a"


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate all results into a delivery summary")
    p.add_argument("--runs", default="results/runs.csv")
    p.add_argument("--grid", default="results/grid.csv")
    p.add_argument("--repeats", default="results/repeats.csv")
    p.add_argument("--test", default="results/test_final.csv")
    p.add_argument("--val-metrics", default="figures/metrics.json")
    p.add_argument("--history", default="outputs/best_history.json")
    p.add_argument("--out", default="docs/FINAL_ANALYSIS.md")
    args = p.parse_args(argv)

    s1, s2, s3 = read_csv(args.runs), read_csv(args.grid), read_csv(args.repeats)
    test_df = read_csv(args.test)
    val = read_json(args.val_metrics)
    hist = read_json(args.history)

    L = []
    L.append("# 全阶段结果摘要")
    L.append("")
    L.append("> 由 `src/final_report.py` 生成于 %s。" %
             datetime.now().strftime("%Y-%m-%d %H:%M"))
    L.append("> 只呈现数据与可核对事实，供填写 `docs/REPORT.md` 与准备答辩参考；")
    L.append("> **结论与机制解释以 `docs/REPORT.md` 为准**。")
    L.append("")

    L.append("## 1. 各阶段概况")
    L.append("")
    L.append("| 阶段 | 文件 | 有效组数 | 说明 |")
    L.append("| --- | --- | --- | --- |")
    extra = 0
    if s1 is not None and "run_id" in s1.columns:
        known = s1["run_id"].astype(str).str.match(r"^(s1_\d+|base)$")
        extra = int((~known).sum())
    s1_note = "每次只改一个超参，seed=0"
    if extra:
        s1_note += "；另含 %d 条最终 best 训练记录（非扫描组）" % extra
    L.append("| S1 单因子扫描 | `%s` | %s | %s |" %
             (args.runs, len(s1) if s1 is not None else "—", s1_note))
    L.append("| S2 小网格 | `%s` | %s | lr × conv_width 组合，seed=0 |" %
             (args.grid, len(s2) if s2 is not None else "—"))
    L.append("| S3 三 seed 复跑 | `%s` | %s | base / 最优 / 次优 各 3 个 seed |" %
             (args.repeats, len(s3) if s3 is not None else "—"))
    L.append("| 最终 test 评测 | `%s` | %s | **只应有一行** |" %
             (args.test, len(test_df) if test_df is not None else "—"))
    L.append("")

    if s3 is not None and len(s3):
        L.append("## 2. S3 显著性与种子波动")
        L.append("")
        g = s3.groupby("tag")["val_auc"]
        st = pd.DataFrame({"n": g.size(), "mean": g.mean(),
                           "std": g.std(ddof=0), "min": g.min(), "max": g.max()})
        L.append("| 配置 | n | mean | std | min | max |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for tag, r in st.iterrows():
            L.append("| `%s` | %d | %s | %s | %s | %s |" %
                     (tag, r["n"], fmt(r["mean"]), fmt(r["std"]),
                      fmt(r["min"]), fmt(r["max"])))
        L.append("")
        if "base" in st.index:
            base_mean, base_std = st.loc["base", "mean"], st.loc["base", "std"]
            best_tag = st["mean"].idxmax()
            best_mean, best_std = st.loc[best_tag, "mean"], st.loc[best_tag, "std"]
            delta = best_mean - base_mean
            L.append("- 最优组：`%s`，mean %s，相对 base 差值 **%+.6f**。" %
                     (best_tag, fmt(best_mean), delta))
            L.append("- base std %s、最优组 std %s；差值约为 base std 的 **%.1f 倍**。" %
                     (fmt(base_std), fmt(best_std),
                      delta / max(base_std, 1e-9)))
            L.append("- 判断口径（教程 5 节）：差值明显大于两者 std 才可写「调参带来提升」，"
                     "否则改写成超参敏感性分析。**结论写在报告里，摘要只给口径**。")
        L.append("")

    if s2 is not None and len(s2):
        L.append("## 3. S2 组合网格（注意：未采纳的组合不要事后评测）")
        L.append("")
        L.append("| 组合 | seed | val_auc | val_acc |")
        L.append("| --- | --- | --- | --- |")
        for _, r in s2.iterrows():
            L.append("| `%s` | %s | %s | %s |" %
                     (r.get("tag", "?"), r.get("seed", ""),
                      fmt(r.get("val_auc")), fmt(r.get("val_acc"))))
        best_grid = s2.nlargest(1, "val_auc").iloc[0]
        L.append("")
        L.append("- S2 最高组合：`%s`，val_auc %s。" %
                 (best_grid.get("tag", "?"), fmt(best_grid["val_auc"])))
        L.append("- 最终 `configs/best.yaml` 取自 S1 单因子最优（经 S3 验证），"
                 "**不是** S2 最高组合。")
        L.append("- test 已经评过一次，不能为了更优组合回头改配置再评第二次；"
                 "该组合只能写成「遗留改进方向」，并在答辩时主动说明。")
        L.append("")

    if val is not None and test_df is not None and len(test_df):
        t = test_df.iloc[0]
        L.append("## 4. val 与 test 的落差（答辩必问点）")
        L.append("")
        L.append("| 指标 | val（report.py, %s 张） | test（run.py --split test, 624 张） | 差 |" %
                 val.get("n_samples", "?"))
        L.append("| --- | --- | --- | --- |")
        va, ta = val.get("auc"), t.get("test_auc")
        vq, tq = val.get("acc"), t.get("test_acc")
        L.append("| AUC | %s | %s | %+.4f |" %
                 (fmt(va, 4), fmt(ta, 4), float(ta) - float(va)))
        L.append("| ACC | %s | %s | %+.4f |" %
                 (fmt(vq, 4), fmt(tq, 4), float(tq) - float(vq)))
        L.append("| 混淆矩阵 | %s | %s | — |" %
                 (val.get("confusion_matrix"), t.get("confusion_matrix")))
        L.append("")
        L.append("- test 的假阳性（正常判成肺炎）明显上升，说明固定阈值 0.5 在 test 上偏高。")
        L.append("- 事实依据：MedMNIST 官方 INFO 说明 train/val 来自 source training set"
                 " 的 9:1 划分，test 来自独立的 source validation set —— 两者来源不同，"
                 "存在分布差异（对应已学 C11 领域自适应 / 协变量偏移）。")
        L.append("- **不能事后拿 test 调阈值或选配置**，那等于泄题（C1 自测第三题）。"
                 "落差应作为现象解释与后续工作，而不是返工理由。")
        L.append("")

    if hist:
        L.append("## 5. 最佳权重对应的训练轮次")
        L.append("")
        L.append("- 训练共 %d 轮；`run.py` 保存的是 **val_auc 最高的那一轮**，不是最后一轮。" %
                 len(hist))
        last = hist[-1]
        best_ep = max(hist, key=lambda h: h["val_auc"])
        L.append("- 最后一轮 epoch %d val_auc %s；最佳轮 epoch %d val_auc %s。" %
                 (last["epoch"], fmt(last["val_auc"], 4),
                  best_ep["epoch"], fmt(best_ep["val_auc"], 4)))
        L.append("- 该数值应与 `figures/metrics.json` 的 val AUC 一致，可用于核对权重是否存对。")
        L.append("")

    L.append("## 6. 与已学章节的可挂钩点（供 REVIEW.md / 答辩使用）")
    L.append("")
    L.append("| 现象 | 可对应的已学内容 |")
    L.append("| --- | --- |")
    L.append("| val 选配置、test 只评一次 | C1 自测第三题「用训练集调参是看答案」 |")
    L.append("| lr / optimizer / BN 最敏感 | C2 优化技巧（动量、自适应学习率、BN） |")
    L.append("| conv_width=64 最优、不堆池化 | C3 卷积参数量与「微小结节会在池化中丢失」 |")
    L.append("| val 高、test 明显回落 | C11 协变量偏移 / 分布不匹配 |")
    L.append("| 权重保存最佳轮而非末轮 | C1 早停与模型选择 |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("本文件不替代 `docs/REPORT.md`；结论、机制解释、遗留问题与改进方向写在报告里，")
    L.append("遗留问题与改进方向，并在 `docs/REVIEW.md` 完成学习复盘与章节映射表。")
    L.append("")

    text = "\n".join(L)
    print(text)
    if args.out:
        from .utils import ensure_dir
        ensure_dir(os.path.dirname(args.out) or ".")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
