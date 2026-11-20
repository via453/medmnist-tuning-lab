"""生成答辩用学术风格图件（中文轴标、低饱和配色、高分辨率）。

    python -m src.figures_academic

只读 val 划分与已有结果表，不触碰 test，不覆盖 figures/ 下的归档图；
输出到 figures/academic/。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .data import get_loaders
from .evaluate import predict
from .model import build_model
from .utils import ensure_dir, load_config

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#8C97A8"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.color"] = "#3A4453"
plt.rcParams["ytick.color"] = "#3A4453"
plt.rcParams["text.color"] = "#1F2733"

BLUE = "#2E5FA3"
RED = "#B04A3A"
GREEN = "#3F7A5E"
GRID = "#E3E7EE"


def _style(ax):
    ax.grid(color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_before_after(out_dir):
    """宽度 32 与 64 的验证集 AUC 对照。"""
    fig, ax = plt.subplots(figsize=(6.1, 4.15))
    xs = ["宽度 32（基线）", "宽度 64"]
    ys = [0.984400, 0.988956]
    bars = ax.bar(xs, ys, width=0.5, color=["#9DB8D9", "#4F82BE"],
                  edgecolor="#2E5FA3", linewidth=1.2)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, "%.4f" % v,
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color="#1F2733")
    ax.set_ylim(0.978, 0.992)
    ax.set_ylabel("验证集 AUC", fontsize=12, rotation=0, labelpad=22, va="center")
    ax.set_title("验证集 AUC：宽度 32 与 64（纵轴截取 0.978–0.992）", fontsize=12, pad=10)
    ax.tick_params(labelsize=11)
    _style(ax)
    fig.tight_layout()
    p = os.path.join(out_dir, "val_auc_bar.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("written:", p)


def fig_capacity(out_dir):
    """同参数量下三种结构在验证集上的 AUC。"""
    df = pd.read_csv("results/capacity_summary.csv", encoding="utf-8-sig")
    names = {"mlp": ("MLP", BLUE, "o"), "smallcnn": ("SmallCNN", RED, "s"),
             "vit": ("TinyViT", GREEN, "^")}
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for key, (label, color, marker) in names.items():
        sub = df[df["model"] == key].sort_values("params")
        ax.errorbar(sub["params"], sub["val_auc_mean"], yerr=sub["val_auc_std"],
                    fmt=marker + "-", color=color, lw=1.6, ms=6, capsize=3,
                    elinewidth=1.2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("可训练参数量（对数轴）", fontsize=11)
    ax.set_ylabel("验证集 AUC", fontsize=11, rotation=0, labelpad=20, va="center")
    ax.set_title("同参数量下的结构对比（3 个随机种子）", fontsize=12, pad=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    p = os.path.join(out_dir, "capacity.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("written:", p)


def fig_loss(out_dir, history="outputs/best_history.json"):
    """训练损失与验证集 AUC 随轮次的变化。"""
    hist = json.load(open(history, encoding="utf-8"))
    ep = [h["epoch"] for h in hist]
    tr = [h["train_loss"] for h in hist]
    va = [h["val_auc"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(5.4, 3.4))
    ax1.plot(ep, tr, color=BLUE, lw=2, label="训练损失")
    ax1.set_xlabel("训练轮次", fontsize=11)
    ax1.set_ylabel("训练损失", color=BLUE, fontsize=11, rotation=0, labelpad=42, va="center")
    ax1.tick_params(labelsize=9.5)
    ax1.grid(color=GRID, linewidth=0.9)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(ep, va, color=GREEN, lw=2, label="验证集 AUC")
    ax2.set_ylabel("验证集 AUC", color=GREEN, fontsize=11, rotation=0, labelpad=28, va="center")
    ax2.tick_params(labelsize=9.5)
    for side in ("top",):
        ax1.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)

    ax1.axvline(19, color="#8C97A8", lw=1, ls="--")
    ax1.set_title("训练损失与验证集 AUC（20 轮）", fontsize=12, pad=9)
    fig.tight_layout(pad=0.9)
    p = os.path.join(out_dir, "loss.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print("written:", p)


def fig_adv(out_dir):
    """干净训练与对抗训练在 FGSM 不同强度下的验证集表现。"""
    df = pd.read_csv("results/adv_curve.csv", encoding="utf-8-sig")
    eps = df["eps"].values
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.48))
    specs = [("AUC", "clean_auc_mean", "adv_auc_mean", "验证集 AUC"),
             ("ACC", "clean_acc_mean", "adv_acc_mean", "验证集 ACC")]
    for ax, (_, ck, ak, ylab) in zip(axes, specs):
        for mean_k, std_k, color, label in (
                (ck, ck.replace("_mean", "_std"), RED, "干净训练"),
                (ak, ak.replace("_mean", "_std"), BLUE, "对抗训练（ε=0.03）")):
            m = df[mean_k].values
            sd = df[std_k].values
            ax.plot(eps, m, "-o", color=color, lw=1.8, ms=4, label=label)
            ax.fill_between(eps, m - sd, m + sd, color=color, alpha=0.15, lw=0)
        ax.axvline(0.03, color="#8C97A8", lw=1, ls="--")
        ax.set_xlabel("攻击强度 ε", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.tick_params(labelsize=9.5)
        ax.set_ylim(-0.02, 1.03)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        _style(ax)
    fig.tight_layout(pad=1.0)
    p = os.path.join(out_dir, "adv.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("written:", p)


def _val_probs(cfg_path="configs/best.yaml", ckpt="outputs/best.pth"):
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    _, val_loader, _ = get_loaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model", state))
    return predict(model, val_loader, device)


def fig_roc(out_dir, labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    v = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(fpr, tpr, color=BLUE, lw=2.2, label="AUC = %.4f" % v)
    ax.plot([0, 1], [0, 1], ls="--", color="#A8B1BF", lw=1)
    ax.set_xlabel("假阳性率", fontsize=10)
    ax.set_ylabel("真阳性率", fontsize=10, rotation=0, labelpad=16, va="center")
    ax.set_title("验证集 ROC 曲线", fontsize=11, pad=8)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    p = os.path.join(out_dir, "roc.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("written:", p, "auc=%.4f" % v)


def fig_confusion(out_dir, labels, probs):
    pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, pred)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["正常", "肺炎"], fontsize=10)
    ax.set_yticks([0, 1], ["正常", "肺炎"], fontsize=10)
    ax.set_xlabel("预测", fontsize=10)
    ax.set_ylabel("真实", fontsize=10, rotation=0, labelpad=16, va="center")
    ax.set_title("验证集混淆矩阵（524 张）", fontsize=11, pad=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="#FFFFFF" if cm[i, j] > cm.max() / 2 else "#1F2733")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "confusion.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("written:", p)


def main():
    out_dir = "figures/academic"
    ensure_dir(out_dir)
    fig_before_after(out_dir)
    fig_capacity(out_dir)
    fig_loss(out_dir)
    fig_adv(out_dir)
    labels, probs = None, None
    try:
        probs, labels = _val_probs()
        fig_roc(out_dir, labels, probs)
        fig_confusion(out_dir, labels, probs)
    except Exception as exc:  # 缺权重时仍产出其余四张
        print("roc/confusion skipped:", exc)


if __name__ == "__main__":
    main()
