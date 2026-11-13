"""从 results/runs.csv 选出 val_auc 最高的配置，给出 best.yaml 建议。

    python -m src.select_best --results results/runs.csv
    python -m src.select_best --results results/runs.csv --write   # 写出 configs/best.yaml

重要：本脚本只做「按 val_auc 排序」这一件机械的事，不代替本人判断。
若结果只有单个 seed，选出的差值可能小于种子波动，必须先按 S3 跑 3 个 seed
并确认提升显著后，才把配置写进 best.yaml 并评测 test。
"""
import argparse
import os
import sys

import pandas as pd

from .utils import load_config

HP_KEYS = [
    "dataset", "size", "model", "conv_width", "use_bn", "activation", "pooling",
    "dropout", "n_classes", "n_channels", "optimizer", "lr", "momentum",
    "weight_decay", "batch_size", "epochs", "scheduler", "warmup_epochs",
    "label_smoothing", "augment", "class_weight", "early_stop_patience",
    "device", "num_workers", "limit_train", "limit_val", "out_dir", "ckpt_dir",
]

HEADER = """# 由 src/select_best.py 按 val_auc 从 results/runs.csv 选出，待本人核对后使用。
# 注意：单次 seed 的最优不一定稳定；请先完成 3 seed 复跑（S3）再评测 test。
"""


def yaml_value(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return str(v)


def main(argv=None):
    p = argparse.ArgumentParser(description="Select best config from runs.csv")
    p.add_argument("--results", default="results/runs.csv")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--out", default="configs/best.yaml")
    p.add_argument("--write", action="store_true", help="写出 best.yaml")
    args = p.parse_args(argv)

    if not os.path.isfile(args.results):
        print("ERROR: %s 不存在，先跑 src.run / src.sweep。" % args.results)
        return 2
    df = pd.read_csv(args.results, encoding="utf-8-sig")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df[df["val_auc"].notna()]
    if df.empty:
        print("ERROR: 没有有效的 val_auc 行。")
        return 2
    if "lr" not in df.columns or "optimizer" not in df.columns:
        print("[警告] 该文件缺少完整超参列（lr / optimizer / ...），无法还原配置。")
        print("        请用 results/runs.csv 选配置；repeats.csv / grid.csv 只用于判断显著性。")

    best = df.nlargest(1, "val_auc").iloc[0]
    base_row = df[df["changed_key"] == "base"]
    base_auc = float(base_row["val_auc"].min()) if len(base_row) else float(df["val_auc"].min())
    n_seeds = df["seed"].nunique()

    print("runs: %d  | distinct seeds: %d  | base val_auc: %.6f" % (
        len(df), n_seeds, base_auc))
    print("best: %s  changed_key=%s  changed_value=%s  val_auc=%.6f  delta=%+.6f" % (
        best.get("run_id", "?"), best.get("changed_key", ""),
        best.get("changed_value", ""), best["val_auc"],
        best["val_auc"] - base_auc))
    print("\n与 base 相比需要改动的配置字段：")
    cfg = load_config(args.base)
    diff = {}
    for k in HP_KEYS:
        if k in df.columns and pd.notna(best.get(k)) and k in cfg:
            if str(best[k]) != str(cfg[k]):
                diff[k] = best[k]
                print("  %-20s %s -> %s" % (k, cfg[k], best[k]))
    if not diff:
        print("  （无差异：最优组就是 base 本身）")

    print("\n超参敏感性（各字段 val_auc 极差，越大越敏感）：")
    g = df[df["changed_key"].notna() & (df["changed_key"] != "base")] \
        .groupby("changed_key")["val_auc"]
    spread = (g.max() - g.min()).sort_values(ascending=False)
    for k, v in spread.items():
        print("  %-20s %.4f" % (k, v))

    if n_seeds < 3:
        print("\n[提醒] 当前只有 %d 个 seed，最优与 base 的差值 %.4f 可能小于种子波动。" % (
            n_seeds, best["val_auc"] - base_auc))
        print("        建议先跑 S3（base / 最优 / 次优 各 3 seed）再做决定。")

    if args.write:
        out = dict(cfg)
        out.update(diff)
        lines = [HEADER]
        for k in HP_KEYS:
            if k in out:
                lines.append("%s: %s" % (k, yaml_value(out[k])))
        lines.append("seed: 0")
        lines.append("")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("\nwritten:", args.out, "（请本人核对后再用于 test 评测）")
    else:
        print("\n未写入（加 --write 才写 configs/best.yaml）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
