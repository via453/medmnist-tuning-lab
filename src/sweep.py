"""单因子扫描（完整教程 4.3 的 S1 阶段）。

    python -m src.sweep --config configs/sweep.yaml
    python -m src.sweep --config configs/sweep.yaml --max-runs 2 --epochs 1 --out .workbuddy/smoke_runs.csv

规则：每组只改 base_config 里的一个字段，其余完全一致；失败组也写进 CSV，不删除。
"""
import argparse
import os
import sys
from datetime import datetime

from .data import get_loaders
from .run import FIELDS
from .train import fit
from .utils import append_csv, load_config, short_git_sha


def resolve_base(sweep_path, declared):
    """base_config 允许相对仓库根或相对 configs/ 目录。"""
    if os.path.isfile(declared):
        return declared
    alt = os.path.join(os.path.dirname(os.path.abspath(sweep_path)),
                       os.path.basename(declared))
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError("base_config not found: %s (also tried %s)" % (declared, alt))


def build_runs(sweep_cfg, sweep_path):
    base = load_config(resolve_base(sweep_path, sweep_cfg["base_config"]))
    runs = [("base", "", dict(base))]
    for key, values in sweep_cfg.get("sweep", {}).items():
        for value in values:
            cfg = dict(base)
            cfg[key] = value
            runs.append((key, value, cfg))
    return runs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MedMNIST tuning lab - single-factor sweep")
    p.add_argument("--config", default="configs/sweep.yaml")
    p.add_argument("--max-runs", type=int, help="冒烟测试时限制组数")
    p.add_argument("--epochs", type=int, help="覆盖 epochs（冒烟用）")
    p.add_argument("--limit-train", type=int, help="只取前 N 个训练样本（冒烟用）")
    p.add_argument("--limit-val", type=int, help="只取前 N 个验证样本（冒烟用）")
    p.add_argument("--out", help="覆盖 runs.csv 输出路径")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spec = load_config(args.config)
    runs = build_runs(spec, args.config)
    if args.max_runs:
        runs = runs[:args.max_runs]

    seeds = spec.get("seeds", [0])
    out_path = args.out or spec.get("out", "results/runs.csv")
    print("sweep: %d runs, seed=%s, out=%s" % (len(runs), seeds[0], out_path))

    for i, (key, value, cfg) in enumerate(runs, 1):
        cfg = dict(cfg)
        cfg["seed"] = seeds[0]
        if args.epochs:
            cfg["epochs"] = args.epochs
        if spec.get("epochs_override"):
            cfg["epochs"] = spec["epochs_override"]
        if args.limit_train:
            cfg["limit_train"] = args.limit_train
        if args.limit_val:
            cfg["limit_val"] = args.limit_val
        tag = "%s=%s" % (key, value) if key != "base" else "base"
        print("[%d/%d] %s" % (i, len(runs), tag))
        try:
            loaders = get_loaders(cfg)
            result = fit(cfg, loaders, run_id="s1_%02d" % i,
                         ckpt_path=None, verbose=False)
            row = {
                "run_id": "s1_%02d" % i,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": cfg["seed"], "git_commit": short_git_sha(),
                "status": "ok", "note": "single-factor sweep",
                "changed_key": key, "changed_value": str(value),
                "dataset": cfg.get("dataset"), "model": cfg.get("model"),
                "lr": cfg.get("lr"), "optimizer": cfg.get("optimizer"),
                "batch_size": cfg.get("batch_size"),
                "weight_decay": cfg.get("weight_decay"),
                "dropout": cfg.get("dropout"), "use_bn": cfg.get("use_bn"),
                "augment": cfg.get("augment"),
                "class_weight": cfg.get("class_weight"),
                "scheduler": cfg.get("scheduler"),
                "conv_width": cfg.get("conv_width"),
                "pooling": cfg.get("pooling"),
                "epochs_run": result["epochs_run"],
                "duration_min": result["duration_min"],
                "params": result["params"],
                "val_auc": round(result["val_auc"], 6),
                "val_acc": round(result["val_acc"], 6),
            }
            print("    val_auc=%.4f  time=%.2f min" % (
                result["val_auc"], result["duration_min"]))
        except Exception as exc:  # 失败组也要留痕
            row = {
                "run_id": "s1_%02d" % i,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": cfg.get("seed"), "git_commit": short_git_sha(),
                "status": "failed", "note": "%s: %s" % (type(exc).__name__, exc),
                "changed_key": key, "changed_value": str(value),
            }
            print("    FAILED: %s: %s" % (type(exc).__name__, exc))
        append_csv(out_path, row, FIELDS)

    print("done:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
