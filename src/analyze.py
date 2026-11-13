"""把 results/runs.csv 整理成阶段分析摘要（机器整理，供本人填写 REPORT.md 参考）。

    python -m src.analyze --results results/runs.csv
    python -m src.analyze --results results/runs.csv --out docs/S1_ANALYSIS.md

不做任何判断性结论，只呈现：base、Top-N、各超参波动、差值与波动的比较、
AUC/ACC 分离现象、seed 数量提醒。最终结论与署名由本人在 docs/REPORT.md 完成。
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

HP_COLS = ["lr", "optimizer", "batch_size", "weight_decay", "dropout",
           "use_bn", "augment", "class_weight", "scheduler", "conv_width", "pooling"]


def build_summary(path, top_n=5):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df[df["val_auc"].notna()]
    if df.empty:
        return None, "没有有效 val_auc 行"

    base_rows = df[df["changed_key"] == "base"]
    base_auc = float(base_rows["val_auc"].min()) if len(base_rows) else float(df["val_auc"].min())
    n_seeds = int(df["seed"].nunique())
    best = df.nlargest(1, "val_auc").iloc[0]
    delta = float(best["val_auc"]) - base_auc

    lines = []
    lines.append("# S1 单因子扫描 · 数据摘要（机器整理，非本人结论）")
    lines.append("")
    lines.append("> 由 `src/analyze.py` 从 `%s` 自动整理，生成时间 %s。" % (
        path, datetime.now().strftime("%Y-%m-%d %H:%M")))
    lines.append("> 本文件**只呈现数据**，用于辅助填写 `docs/REPORT.md`；结论、判断与署名须由本人完成。")
    lines.append("> 底层数据 `results/runs.csv` 由本人本地跑出，随实验结果一并归档；本摘要不替代它。")
    lines.append("")
    lines.append("## 概况")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| 有效实验组数 | %d |" % len(df))
    lines.append("| 不同 seed 数 | %d |" % n_seeds)
    lines.append("| base val_auc | %.6f |" % base_auc)
    lines.append("| 最优组 | `%s`（%s=%s） |" % (
        best.get("run_id", "?"), best.get("changed_key", ""), best.get("changed_value", "")))
    lines.append("| 最优 val_auc | %.6f |" % best["val_auc"])
    lines.append("| 相对 base 差值 | %+.6f |" % delta)
    lines.append("")
    lines.append("## Top %d 组" % top_n)
    lines.append("")
    cols = ["run_id", "changed_key", "changed_value", "val_auc", "val_acc",
            "epochs_run", "duration_min"]
    cols = [c for c in cols if c in df.columns]
    t = df.nlargest(top_n, "val_auc")[cols]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("## 各超参的 val_auc 极差（越大越敏感）")
    lines.append("")
    g = df[df["changed_key"].notna() & (df["changed_key"] != "base")] \
        .groupby("changed_key")["val_auc"]
    spread = (g.max() - g.min()).sort_values(ascending=False)
    lines.append("| 超参 | 组数 | min | max | 极差 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for k, v in spread.items():
        grp = g.get_group(k)
        lines.append("| `%s` | %d | %.4f | %.4f | %.4f |" % (
            k, len(grp), grp.min(), grp.max(), v))
    lines.append("")
    max_spread = float(spread.max()) if len(spread) else 0.0
    lines.append("## 需要本人判断的三件事")
    lines.append("")
    lines.append("1. **差值是否显著**：最优相对 base %+.6f，而单个超参内部最大极差为 %.4f。"
                 % (delta, max_spread))
    if n_seeds < 3:
        lines.append("   当前只有 %d 个 seed，**必须先跑 S3（base / 最优 / 次优 各 3 个 seed）**，"
                     "用均值 ± 标准差判断提升是否大于种子波动。" % n_seeds)
    else:
        lines.append("   已有 %d 个 seed，请在 REPORT.md 里给出均值 ± 标准差再下结论。" % n_seeds)
    lines.append("2. **AUC 与 ACC 是否分离**：检查 `class_weight=balanced` 等组，"
                 "若 AUC 升而 ACC 明显降，需在报告里解释阈值与类别不平衡的关系。")
    lines.append("3. **结论写法**：若提升不显著，按教程第 5 节改写成「超参敏感性分析」"
                 "——哪些超参敏感、哪些几乎不影响、小数据方差来源，不宣称刷点。")
    lines.append("")
    lines.append("## 本次数据中值得单独说明的现象")
    lines.append("")
    if "class_weight" in df.columns:
        cw = df[df["changed_key"] == "class_weight"]
        if len(cw):
            b = cw[cw["class_weight"] == "balanced"]
            if len(b):
                r = b.iloc[0]
                lines.append("- `class_weight=balanced`：val_auc %.4f，但 val_acc 仅 %.4f"
                             "（base 的 val_acc 见上表）——典型 AUC/ACC 分离。" % (
                                 r["val_auc"], r["val_acc"]))
    if len(spread):
        lines.append("- 最敏感的三个超参：`%s`。" % "`、`".join(list(spread.index[:3])))
        lines.append("- 几乎不敏感（极差 < 0.005）：`%s`。" % (
            "`、`".join(list(spread[spread < 0.005].index)) or "无"))
    lines.append("")
    return "\n".join(lines) + "\n", None


def main(argv=None):
    p = argparse.ArgumentParser(description="Summarize runs.csv")
    p.add_argument("--results", default="results/runs.csv")
    p.add_argument("--out", help="写出 Markdown 文件")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args(argv)
    if not os.path.isfile(args.results):
        print("ERROR: %s 不存在" % args.results)
        return 2
    text, err = build_summary(args.results, args.top)
    if err:
        print("ERROR:", err)
        return 2
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
