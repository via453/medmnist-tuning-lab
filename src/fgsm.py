"""C9 对抗攻击：FGSM 的 ε 曲线（笔记「五件动手的事」第 3 件）。

做法：x' = x + ε·sign(∇_x L)，梯度对**输入**求，裁剪回 [-1, 1]（本项目的归一化范围）。
只评 val，不碰 test。输出 results/fgsm.csv + figures/fgsm_curve.png。
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_loaders
from .evaluate import compute_metrics, predict
from .model import build_model
from .utils import ensure_dir


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["config"], ck.get("epoch")


@torch.enable_grad()
def fgsm_eval(model, loader, device, eps):
    """返回 (metrics, 平均扰动幅度)。eps=0 时即干净准确率。"""
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device).clone().detach().requires_grad_(True)
        y = y.view(-1).to(device).long()
        loss = F.cross_entropy(model(x), y)
        grad, = torch.autograd.grad(loss, x)
        x_adv = (x + eps * grad.sign()).clamp(-1.0, 1.0).detach()
        with torch.no_grad():
            p = torch.softmax(model(x_adv), 1)[:, 1]
        ys.append(y.cpu().numpy())
        ps.append(p.cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return compute_metrics(y_true, y_prob)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/best.pth")
    p.add_argument("--config", default="configs/best.yaml", help="仅用于取数据与 batch 设置")
    p.add_argument("--eps", default="0,0.01,0.02,0.03,0.05,0.08,0.1,0.2,0.3")
    p.add_argument("--out", default="results/fgsm.csv")
    p.add_argument("--fig", default="figures/fgsm_curve.png")
    a = p.parse_args(argv)

    from .utils import load_config
    cfg = load_config(a.config)
    device = torch.device(cfg.get("device", "cpu"))
    model, ck_cfg, epoch = load_model(a.ckpt, device)
    cfg.update({k: v for k, v in ck_cfg.items() if k in ("dataset", "size")})
    _, val_loader, _ = get_loaders(cfg)

    rows = []
    for eps in [float(x) for x in a.eps.split(",")]:
        m = fgsm_eval(model, val_loader, device, eps)
        rows.append({"eps": eps, "val_auc": m["auc"], "val_acc": m["acc"], "val_f1": m["f1"]})
        print("eps %.2f  auc %.4f  acc %.4f  f1 %.4f" % (eps, m["auc"], m["acc"], m["f1"]))

    ensure_dir("results")
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "val_auc", "val_acc", "val_f1"])
        w.writeheader()
        w.writerows(rows)
    print("written:", a.out, "| ckpt epoch:", epoch)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = [r["eps"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(e, [r["val_acc"] for r in rows], "o-", label="ACC")
        ax.plot(e, [r["val_auc"] for r in rows], "s-", label="AUC")
        ax.set_xlabel("FGSM epsilon")
        ax.set_ylabel("score")
        ax.set_title("FGSM attack on PneumoniaMNIST (val, 524)")
        ax.grid(alpha=.3)
        ax.legend()
        ensure_dir("figures")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=140)
        print("figure:", a.fig)
    except Exception as ex:  # 出图失败不影响 CSV
        print("plot skipped:", ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
