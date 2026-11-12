"""由 results/*.csv 现场生成图件。绝不生成假数据：没有实验结果就明确提示并退出。

    python -m src.plots --results results/runs.csv

产出（有数据时）：
    figures/hp_sensitivity.png   各超参的 val_auc 波动范围
    figures/before_after.png     base 与最优配置的对比（多 seed 时带误差棒）
ROC / 混淆矩阵 / loss 曲线需要预测概率或逐 epoch 历史，由后续实验脚本补充。
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def read_runs(path):
    if not os.path.isfile(path):
        print("NO DATA: %s 不存在。" % path)
        print("先由本人本地执行 python -m src.run / src.sweep 产出真实实验结果；"
              "本脚本不会生成示意数据或假图。")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
    if "val_auc" not in ok.columns or len(ok) == 0:
        print("NO DATA: %s 中没有有效的 val_auc 行。" % path)
        return None
    return ok


def plot_sensitivity(df, out_dir):
    d = df[df["changed_key"].notna() & (df["changed_key"] != "")] \
        if "changed_key" in df.columns else pd.DataFrame()
    if d.empty:
        return False
    groups = d.groupby("changed_key")["val_auc"]
    keys = list(groups.groups.keys())
    lows = [float(groups.get_group(k).min()) for k in keys]
    highs = [float(groups.get_group(k).max()) for k in keys]
    means = [float(groups.get_group(k).mean()) for k in keys]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(keys) + 1)))
    y = np.arange(len(keys))
    ax.barh(y, [h - l for l, h in zip(lows, highs)], left=lows,
            color="#b5d4f4", edgecolor="#378add", height=0.55)
    ax.scatter(means, y, color="#185fa5", zorder=3, s=28)
    ax.set_yticks(y, keys)
    ax.set_xlabel("val AUC (bar = min~max, dot = mean)")
    ax.set_title("Hyperparameter sensitivity")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "hp_sensitivity.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("written:", p)
    return True


def plot_before_after(df, out_dir):
    if "changed_key" not in df.columns:
        return False
    base = df[df["changed_key"] == "base"]["val_auc"]
    rest = df[(df["changed_key"] != "base") & df["changed_key"].notna()]["val_auc"]
    if len(base) == 0 or len(rest) == 0:
        return False
    labels = ["base", "best-so-far"]
    means = [float(base.mean()), float(rest.max())]
    errs = [float(base.std(ddof=0)), 0.0]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(labels, means, yerr=errs, capsize=6,
           color=["#b5d4f4", "#9fe1cb"], edgecolor="#185fa5")
    for i, v in enumerate(means):
        ax.text(i, v, "%.4f" % v, ha="center", va="bottom", fontsize=10)
    ax.set_ylim(max(0.0, min(means) - 0.08), min(1.0, max(means) + 0.08))
    ax.set_ylabel("val AUC")
    ax.set_title("Baseline vs best configuration")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "before_after.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("written:", p)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate figures from results CSV")
    p.add_argument("--results", default="results/runs.csv")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args(argv)

    df = read_runs(args.results)
    if df is None:
        return 1
    ensure_dir(args.out_dir)
    made = plot_sensitivity(df, args.out_dir) + plot_before_after(df, args.out_dir)
    if not made:
        print("提示：CSV 里缺少 changed_key 分组列，暂无法出图；"
              "请用 src/sweep.py 产出带 changed_key 的扫描结果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
