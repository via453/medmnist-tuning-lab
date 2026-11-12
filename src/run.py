"""单配置入口。

    python -m src.run --config configs/base.yaml
    python -m src.run --config configs/best.yaml --split test

默认：训练 + val 监控，结果追加到 results/runs.csv，权重存 outputs/。
--split test：加载已有权重，在 test 上评一次并覆盖写 results/test_final.csv。
             该文件只应有一行；重复运行会覆盖，请在报告里说明。
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import torch

from .data import get_loaders
from .evaluate import compute_metrics, predict
from .model import build_model
from .train import fit
from .utils import append_csv, ensure_dir, load_config, merge_config, short_git_sha

FIELDS = [
    "run_id", "timestamp", "seed", "git_commit", "status", "note",
    "changed_key", "changed_value",
    "dataset", "model", "lr", "optimizer", "batch_size", "weight_decay",
    "dropout", "use_bn", "augment", "class_weight", "scheduler",
    "conv_width", "pooling", "epochs_run", "duration_min", "params",
    "val_auc", "val_acc", "test_auc", "test_acc",
]

TEST_FIELDS = ["run_id", "timestamp", "seed", "git_commit", "config",
               "dataset", "test_auc", "test_acc", "test_f1", "confusion_matrix", "note"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MedMNIST tuning lab - single run")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--out", help="覆盖 runs.csv 输出路径（冒烟测试用）")
    p.add_argument("--ckpt", help="权重路径；默认 outputs/<配置名>.pth")
    p.add_argument("--no-ckpt", action="store_true", help="不保存权重（冒烟测试用）")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--run-id")
    p.add_argument("--note", default="")
    p.add_argument("--changed-key", default="")
    p.add_argument("--changed-value", default="")
    return p.parse_args(argv)


def _row_from(cfg, args, result, status="ok"):
    return {
        "run_id": args.run_id or os.path.splitext(os.path.basename(args.config))[0],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": cfg.get("seed", 0),
        "git_commit": short_git_sha(),
        "status": status,
        "note": args.note,
        "changed_key": args.changed_key,
        "changed_value": args.changed_value,
        "dataset": cfg.get("dataset"),
        "model": cfg.get("model"),
        "lr": cfg.get("lr"),
        "optimizer": cfg.get("optimizer"),
        "batch_size": cfg.get("batch_size"),
        "weight_decay": cfg.get("weight_decay"),
        "dropout": cfg.get("dropout"),
        "use_bn": cfg.get("use_bn"),
        "augment": cfg.get("augment"),
        "class_weight": cfg.get("class_weight"),
        "scheduler": cfg.get("scheduler"),
        "conv_width": cfg.get("conv_width"),
        "pooling": cfg.get("pooling"),
        "epochs_run": result.get("epochs_run"),
        "duration_min": result.get("duration_min"),
        "params": result.get("params"),
        "val_auc": round(result.get("val_auc", 0), 6),
        "val_acc": round(result.get("val_acc", 0), 6),
    }


def eval_test(cfg, args, ckpt_path):
    """只评一次 test。"""
    if not os.path.isfile(ckpt_path):
        print("ERROR: checkpoint not found: %s" % ckpt_path, file=sys.stderr)
        print("先运行 python -m src.run --config <配置> 生成权重。", file=sys.stderr)
        return 2
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(state.get("config", cfg))
    model.load_state_dict(state["model"])
    _, _, test_loader = get_loaders(cfg)
    probs, labels = predict(model, test_loader, torch.device("cpu"))
    m = compute_metrics(labels, probs)

    out_path = args.out or os.path.join(cfg.get("out_dir", "results"), "test_final.csv")
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=TEST_FIELDS)
        w.writeheader()
        w.writerow({
            "run_id": args.run_id or "final",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": cfg.get("seed", 0),
            "git_commit": short_git_sha(),
            "config": args.config,
            "dataset": cfg.get("dataset"),
            "test_auc": round(m["auc"], 6),
            "test_acc": round(m["acc"], 6),
            "test_f1": round(m["f1"], 6),
            "confusion_matrix": str(m["confusion_matrix"]),
            "note": args.note,
        })
    print("TEST (only once): auc=%.4f acc=%.4f f1=%.4f cm=%s" % (
        m["auc"], m["acc"], m["f1"], m["confusion_matrix"]))
    print("written:", out_path)
    return 0


def main(argv=None):
    args = parse_args(argv)
    cfg = merge_config(load_config(args.config), {
        "seed": args.seed,
        "epochs": args.epochs,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
    })
    ckpt_path = args.ckpt or os.path.join(
        cfg.get("ckpt_dir", "outputs"),
        os.path.splitext(os.path.basename(args.config))[0] + ".pth")

    if args.split == "test":
        return eval_test(cfg, args, ckpt_path)

    loaders = get_loaders(cfg)
    print("config=%s  train=%d  val=%d  device=%s" % (
        args.config, len(loaders[0].dataset), len(loaders[1].dataset),
        cfg.get("device", "cpu")))

    result = fit(cfg, loaders, run_id=args.run_id or "run",
                 ckpt_path=None if args.no_ckpt else ckpt_path)
    out_path = args.out or os.path.join(cfg.get("out_dir", "results"), "runs.csv")
    append_csv(out_path, _row_from(cfg, args, result), FIELDS)
    print("BEST val_auc=%.4f val_acc=%.4f epochs=%d params=%d time=%.2f min" % (
        result["val_auc"], result["val_acc"], result["epochs_run"],
        result["params"], result["duration_min"]))
    print("written:", out_path)
    if not args.no_ckpt:
        print("checkpoint:", ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
