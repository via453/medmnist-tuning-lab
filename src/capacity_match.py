"""延伸实验 B：容量严格对齐后，再比结构。

上一轮（docs/EXTENSIONS.md §2）的三组结构参数量并不齐：19,042 / 74,946 / 100,738 / 71,554。
「ViT 赢」可能只是参数量更接近某个甜点。本脚本把三者压到同一参数量再比一次。

做法：
  1. 给定目标参数量 target，对每个结构族反解它的容量旋钮
     （MLP 的 width、SmallCNN 的 conv_width、TinyViT 的 dim），选实际参数量最接近 target 的整数取值；
  2. 每个 (level, 结构, seed) 用与主实验完全相同的优化设置训练，只评 val；
  3. 输出图表：同一横坐标（参数量）上三条曲线——如果 ViT 仍在上方，才说明是结构而非容量。

    python -m src.capacity_match --dry-run          # 只解旋钮，不训练
    python -m src.capacity_match                    # 2 个 level × 3 结构 × 3 seed = 18 次训练
    python -m src.capacity_match --levels 71554 --seeds 0 --epochs 2 --limit-train 512   # 冒烟

输出：results/capacity.csv、results/capacity_summary.csv、figures/capacity_curve.png。
"""
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from .data import get_loaders
from .model import build_model
from .train import fit
from .utils import count_params, ensure_dir, load_config, short_git_sha

FIELDS = ["run_id", "timestamp", "seed", "git_commit", "status", "note",
          "level", "target_params", "model", "knob", "params",
          "val_auc", "val_acc", "epochs_run", "duration_min"]
SUM_FIELDS = ["level", "target_params", "model", "knob", "params", "n",
              "val_auc_mean", "val_auc_std", "val_acc_mean", "val_acc_std"]

# 每个结构族的容量旋钮候选值
KNOBS = {
    "mlp": ("width", list(range(8, 601))),
    "smallcnn": ("conv_width", list(range(4, 301))),
    "vit": ("dim", [d for d in range(16, 257) if d % 4 == 0]),
}


def resolve_knob(base_cfg, model_name, target):
    """返回 (cfg, params, 实际差值比例)。按真实实例化参数量搜索，不用解析公式。"""
    key, values = KNOBS[model_name]
    best = None
    for v in values:
        cfg = dict(base_cfg)
        cfg["model"] = model_name
        cfg[key] = v
        p = count_params(build_model(cfg))
        score = abs(p - target)
        if best is None or score < best[0]:
            best = (score, cfg, p, v)
    _, cfg, p, v = best
    return cfg, p, v, abs(p - target) / float(target)


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_curve(summary, models, fig_path):
    """同一横坐标（参数量，log）上画三条 val AUC 曲线，便于看「结构 vs 容量」。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"mlp": "#2471a3", "smallcnn": "#c0392b", "vit": "#1e8449"}
    labels = {"mlp": "MLP", "smallcnn": "SmallCNN", "vit": "TinyViT"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in models:
        sub = summary[summary["model"] == name].sort_values("params")
        if sub.empty:
            continue
        ax.errorbar(sub["params"], sub["val_auc_mean"], yerr=sub["val_auc_std"],
                    fmt="o-", color=colors.get(name, "k"), capsize=3,
                    label=labels.get(name, name))
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log)")
    ax.set_ylabel("val AUC (mean ± std)")
    ax.set_title("Architecture comparison at matched capacity (val)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    ensure_dir(os.path.dirname(fig_path) or ".")
    fig.savefig(fig_path, dpi=140)
    return fig_path


def main(argv=None):
    p = argparse.ArgumentParser(description="param-matched architecture comparison")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--levels", default="19042,71554",
                   help="目标参数量，逗号分隔；默认取现有 w32 与 TinyViT 的参数量")
    p.add_argument("--models", default="mlp,smallcnn,vit")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, help="覆盖 epochs（冒烟用）")
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--out", default="results/capacity.csv")
    p.add_argument("--summary", default="results/capacity_summary.csv")
    p.add_argument("--fig", default="figures/capacity_curve.png")
    p.add_argument("--dry-run", action="store_true", help="只解旋钮并打印参数量")
    p.add_argument("--replot", action="store_true",
                   help="不训练，直接用已有 %s 重画图" % "results/capacity_summary.csv")
    args = p.parse_args(argv)

    if args.replot:
        summary = pd.read_csv(args.summary, encoding="utf-8-sig")
        models = [m.strip() for m in args.models.split(",")]
        print("figure:", plot_curve(summary, models, args.fig))
        return 0

    base_cfg = load_config(args.config)
    device = torch.device(base_cfg.get("device", "cpu"))
    levels = [int(x) for x in args.levels.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    plan = []
    print("=== 旋钮解算（目标参数量 -> 实际参数量）===")
    for target in levels:
        for name in models:
            cfg, params, knob_v, gap = resolve_knob(base_cfg, name, target)
            key = KNOBS[name][0]
            print("  target %6d | %-8s %s=%-4d -> %6d 参数 (偏差 %.1f%%)" % (
                target, name, key, knob_v, params, gap * 100))
            plan.append((target, name, cfg, params, knob_v))
    if args.dry_run:
        print("\n--dry-run：未训练。")
        return 0

    rows = []
    for target, name, cfg0, params0, knob_v in plan:
        for seed in seeds:
            cfg = dict(cfg0)
            cfg["seed"] = seed
            if args.epochs:
                cfg["epochs"] = args.epochs
            if args.limit_train:
                cfg["limit_train"] = args.limit_train
            if args.limit_val:
                cfg["limit_val"] = args.limit_val
            run_id = "cap%d_%s_s%d" % (target, name, seed)
            row = {"run_id": run_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "seed": seed, "git_commit": short_git_sha(), "status": "ok",
                   "note": "容量对齐对照：target=%d" % target,
                   "level": target, "target_params": target, "model": name,
                   "knob": "%s=%s" % (KNOBS[name][0], knob_v), "params": params0}
            print("[%s] %s seed=%d" % (run_id, row["knob"], seed))
            try:
                loaders = get_loaders(cfg)
                res = fit(cfg, loaders, run_id=run_id, ckpt_path=None, verbose=False)
                row.update({"val_auc": round(res["val_auc"], 6),
                            "val_acc": round(res["val_acc"], 6),
                            "epochs_run": res["epochs_run"],
                            "duration_min": res["duration_min"]})
                print("    val_auc=%.4f  val_acc=%.4f  %.2f min" % (
                    res["val_auc"], res["val_acc"], res["duration_min"]))
            except Exception as exc:
                row.update({"status": "failed",
                            "note": "%s: %s" % (type(exc).__name__, exc)})
                print("    FAILED: %s: %s" % (type(exc).__name__, exc))
            rows.append(row)
            write_csv(args.out, rows, FIELDS)

    # ---- 汇总 ----
    ok = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    if ok.empty:
        print("没有成功行，跳过汇总。")
        return 1
    g = ok.groupby(["level", "model"])
    summary = pd.DataFrame({
        "n": g.size(),
        "val_auc_mean": g["val_auc"].mean(),
        "val_auc_std": g["val_auc"].apply(lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
        "val_acc_mean": g["val_acc"].mean(),
        "val_acc_std": g["val_acc"].apply(lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
    }).reset_index()
    meta = ok.drop_duplicates(["level", "model"])[
        ["level", "model", "params", "knob", "target_params"]]
    summary = summary.merge(meta, on=["level", "model"], how="left")
    summary = summary.round(6)
    write_csv(args.summary, summary.to_dict("records"), SUM_FIELDS)

    print("\n=== 容量对齐后的 val AUC（mean ± std, n=%d）===" % len(seeds))
    print(summary[["level", "model", "knob", "params", "n",
                   "val_auc_mean", "val_auc_std",
                   "val_acc_mean"]].to_string(index=False))
    for level, sub in summary.groupby("level"):
        best = sub.sort_values("val_auc_mean", ascending=False).iloc[0]
        print("  level %d 最优：%s（%.6f，参数量 %d）" % (
            level, best["model"], best["val_auc_mean"], best["params"]))

    try:
        print("figure:", plot_curve(summary, models, args.fig))
    except Exception as exc:
        print("plot skipped:", exc)

    print("written:", args.out, "|", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
