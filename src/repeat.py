"""S2 小网格 / S3 三 seed 复跑（完整教程 4.3）。

S3（必做，判断提升是否显著）：
    python -m src.repeat --mode seeds --results results/runs.csv
S2（可选，看超参交互作用）：
    python -m src.repeat --mode grid --grid "lr=0.001,0.003;conv_width=32,64"

输出：S3 -> results/repeats.csv；S2 -> results/grid.csv。
失败组同样留痕；本脚本不保存权重、不评测 test。
"""
import argparse
import itertools
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

from .data import get_loaders
from .train import fit
from .utils import append_csv, load_config, short_git_sha

HP_KEYS = ["lr", "optimizer", "batch_size", "weight_decay", "dropout",
           "use_bn", "augment", "class_weight", "scheduler", "conv_width", "pooling"]

FIELDS = ["run_id", "timestamp", "mode", "tag", "source_run_id", "seed",
          "git_commit", "status", "note", "changed_key", "changed_value",
          "val_auc", "val_acc", "epochs_run", "duration_min", "params"]


def _native(v):
    if hasattr(v, "item"):
        return v.item()
    return v


def _parse(v):
    if not isinstance(v, str):
        return _native(v)
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def row_to_cfg(row, base_path):
    cfg = load_config(base_path)
    for k in HP_KEYS:
        if k in row and pd.notna(row[k]):
            cfg[k] = _parse(row[k])
    return cfg


def choose_seeds_runs(df, top_n=2):
    """S3：base + val_auc 最高的 top_n 组（排除 base 自身）。"""
    base_rows = df[df["changed_key"] == "base"]
    picks = []
    if len(base_rows):
        b = base_rows.iloc[0]
        picks.append(("base", b))
    rest = df[(df["changed_key"] != "base") & df["val_auc"].notna()]
    for _, r in rest.nlargest(top_n, "val_auc").iterrows():
        picks.append(("%s=%s" % (r.get("changed_key", "?"),
                                 r.get("changed_value", "?")), r))
    return picks


def parse_grid(spec):
    """'lr=0.001,0.003;conv_width=32,64' -> [(key, values), ...]"""
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("bad grid part: %s" % part)
        k, vals = part.split("=", 1)
        out.append((k.strip(), [_parse(v.strip()) for v in vals.split(",")]))
    return out


def run_one(cfg, tag, source, seed, mode, changed_key, changed_value, out_path,
            epochs, limit_train, limit_val):
    cfg = dict(cfg)
    cfg["seed"] = seed
    if epochs:
        cfg["epochs"] = epochs
    if limit_train:
        cfg["limit_train"] = limit_train
    if limit_val:
        cfg["limit_val"] = limit_val

    row = {
        "run_id": "%s_%s_%s" % (mode, tag.replace("=", "-"), seed),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode, "tag": tag, "source_run_id": source, "seed": seed,
        "git_commit": short_git_sha(), "status": "ok", "note": "",
        "changed_key": changed_key, "changed_value": str(changed_value),
    }
    try:
        loaders = get_loaders(cfg)
        result = fit(cfg, loaders, run_id=row["run_id"],
                     ckpt_path=None, verbose=False)
        row.update({
            "val_auc": round(result["val_auc"], 6),
            "val_acc": round(result["val_acc"], 6),
            "epochs_run": result["epochs_run"],
            "duration_min": result["duration_min"],
            "params": result["params"],
        })
        print("    val_auc=%.4f  time=%.2f min" % (
            result["val_auc"], result["duration_min"]))
    except Exception as exc:
        row.update({"status": "failed",
                    "note": "%s: %s" % (type(exc).__name__, exc)})
        print("    FAILED: %s: %s" % (type(exc).__name__, exc))
    append_csv(out_path, row, FIELDS)


def print_summary(out_path):
    df = pd.read_csv(out_path, encoding="utf-8-sig")
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return
    print("\n=== 均值 ± 标准差（按 tag 分组）===")
    g = ok.groupby("tag")["val_auc"]
    summary = pd.DataFrame({"n": g.size(), "mean": g.mean(), "std": g.std(ddof=0)})
    print(summary.round(6).to_string())
    print("\n判断口径：只有「最优 mean − base mean」明显大于两者的 std，")
    print("才能写「调参带来提升」；否则改写成超参敏感性分析（见 docs/S1_ANALYSIS.md）。")


def main(argv=None):
    p = argparse.ArgumentParser(description="S2 grid / S3 seed repeats")
    p.add_argument("--mode", choices=["seeds", "grid"], default="seeds")
    p.add_argument("--results", default="results/runs.csv")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--grid", help="S2 组合，如 'lr=0.001,0.003;conv_width=32,64'")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--epochs", type=int, help="覆盖 epochs（冒烟用）")
    p.add_argument("--limit-train", type=int, help="冒烟用")
    p.add_argument("--limit-val", type=int, help="冒烟用")
    p.add_argument("--max-runs", type=int, help="冒烟时限制总组数")
    p.add_argument("--out", help="覆盖输出 CSV 路径")
    args = p.parse_args(argv)

    base = load_config(args.base)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_path = args.out or ("results/repeats.csv" if args.mode == "seeds"
                            else "results/grid.csv")
    jobs = []

    if args.mode == "seeds":
        if not os.path.isfile(args.results):
            print("ERROR: %s 不存在" % args.results)
            return 2
        df = pd.read_csv(args.results, encoding="utf-8-sig")
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        for tag, row in choose_seeds_runs(df, args.top):
            cfg = row_to_cfg(row, args.base)
            for seed in seeds:
                jobs.append((cfg, tag, row.get("run_id", "?"), seed,
                             row.get("changed_key", ""),
                             row.get("changed_value", "")))
        print("S3: %d configs x %d seeds = %d runs -> %s" % (
            len(choose_seeds_runs(df, args.top)), len(seeds), len(jobs), out_path))
    else:
        if not args.grid:
            print("ERROR: --mode grid 需要 --grid，如 "
                  "'lr=0.001,0.003;conv_width=32,64'")
            return 2
        grid = parse_grid(args.grid)
        keys = [k for k, _ in grid]
        for combo in itertools.product(*[vals for _, vals in grid]):
            cfg = dict(base)
            cfg.update(dict(zip(keys, combo)))
            tag = ",".join("%s=%s" % (k, v) for k, v in zip(keys, combo))
            jobs.append((cfg, tag, "grid", seeds[0],
                         "+".join(keys), str(list(combo))))
        print("S2: %d combos x %d seed -> %s" % (len(jobs), 1, out_path))

    if args.max_runs:
        jobs = jobs[:args.max_runs]
    for i, (cfg, tag, src, seed, ck, cv) in enumerate(jobs, 1):
        print("[%d/%d] %s seed=%s" % (i, len(jobs), tag, seed))
        run_one(cfg, tag, src, seed, args.mode, ck, cv, out_path,
                args.epochs, args.limit_train, args.limit_val)

    print("done:", out_path)
    if args.mode == "seeds":
        print_summary(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
