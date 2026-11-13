"""由训练好的权重生成评估图件（验收三件套第三项）。

    python -m src.report --config configs/best.yaml --ckpt outputs/best.pth --split test

产出 figures/roc.png、confusion_matrix.png、samples.png 与 metrics.json。
图件全部由真实推理结果生成；--split test 只是评测，不会写 test_final.csv
（那一行仍由 src.run --split test 负责，避免重复评测）。
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .data import get_loaders
from .evaluate import compute_metrics, predict
from .model import build_model
from .utils import ensure_dir, load_config


def plot_roc(labels, probs, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, color="#2f6feb", lw=2,
            label="AUC = %.4f" % roc_auc)
    ax.plot([0, 1], [0, 1], "--", color="#9aa4b2", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "roc.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, float(roc_auc)


def plot_confusion(labels, probs, out_dir):
    cm = confusion_matrix(labels, (probs >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["normal", "pneumonia"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["normal", "pneumonia"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="#b42318" if cm[i, j] > cm.max() / 2 else "#1f2328")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    p = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, cm.tolist()


def plot_samples(model, loader, device, out_dir, n=16):
    x, y = next(iter(loader))
    x, y = x[:n], y.view(-1)[:n]
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[:, 1].numpy()
    img = x.numpy()[:, 0] * 0.5 + 0.5
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img[i], cmap="gray", vmin=0, vmax=1)
        ok = int(prob[i] >= 0.5) == int(y[i])
        ax.set_title("T%d P%d %.2f" % (int(y[i]), int(prob[i] >= 0.5), prob[i]),
                     fontsize=9, color="#1a7f37" if ok else "#b42318")
        ax.axis("off")
    fig.suptitle("Predictions on 16 samples (green=correct, red=wrong)")
    fig.tight_layout()
    p = os.path.join(out_dir, "samples.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_loss_curves(history_path, out_dir):
    if not history_path or not os.path.isfile(history_path):
        return None
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    ep = [h["epoch"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.8))
    ax1.plot(ep, [h["train_loss"] for h in hist], color="#2f6feb", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#2f6feb")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_auc"] for h in hist], color="#0f7b6c", label="val AUC")
    ax2.set_ylabel("val AUC", color="#0f7b6c")
    ax1.set_title("Training loss and validation AUC")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "loss_curves.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate evaluation figures from a checkpoint")
    p.add_argument("--config", default="configs/best.yaml")
    p.add_argument("--ckpt", default="outputs/best.pth")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--history", help="可选：训练历史 JSON（由 src.run 保存）")
    args = p.parse_args(argv)

    if not os.path.isfile(args.ckpt):
        print("ERROR: checkpoint not found: %s" % args.ckpt, file=sys.stderr)
        print("先运行 python -m src.run --config %s 生成权重。" % args.config,
              file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(state.get("config", cfg))
    model.load_state_dict(state["model"])
    model.eval()
    ensure_dir(args.out_dir)

    loaders = get_loaders(cfg)
    loader = loaders[2] if args.split == "test" else loaders[1]
    probs, labels = predict(model, loader, torch.device("cpu"))
    metrics = compute_metrics(labels, probs)

    roc_png, roc_auc = plot_roc(labels, probs, args.out_dir)
    cm_png, cm = plot_confusion(labels, probs, args.out_dir)
    sm_png = plot_samples(model, loader, torch.device("cpu"), args.out_dir)
    print("written:", roc_png)
    print("written:", cm_png)
    print("written:", sm_png)

    loss_png = plot_loss_curves(args.history, args.out_dir)
    if loss_png:
        print("written:", loss_png)
    else:
        print("note: no --history given, skip loss_curves.png")

    metrics.update({"split": args.split, "config": args.config,
                    "ckpt": args.ckpt, "roc_auc": roc_auc,
                    "confusion_matrix": cm, "n_samples": int(len(labels))})
    mp = os.path.join(args.out_dir, "metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("written:", mp)
    print("%s: auc=%.4f acc=%.4f f1=%.4f cm=%s" % (
        args.split, metrics["auc"], metrics["acc"], metrics["f1"], cm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
