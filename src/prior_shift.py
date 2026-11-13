"""C11 的定量延伸：val→test 的落差里，有多少来自「阳性先验变化」？

方法（只用 val，不碰 test）：
1. 在完整 val（阳性 74.2%）上算 ACC / AUC / F1；
2. 随机下采样阳性样本，把 val 的阳性率压到与 test 相同的 62.5%，重复 200 次取均值；
3. 两次 ACC 之差 = 先验变化能解释的部分；剩余部分才是真正的分布偏移。

这样能把「模型坏了」和「阈值在新先验下偏了」分开，答辩时可直接引用。
"""
import argparse

import numpy as np
import torch

from .data import get_loaders
from .evaluate import compute_metrics, predict
from .model import build_model


def acc_at(y, p, thr=0.5):
    return float(((p >= thr).astype(int) == y).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/best.pth")
    ap.add_argument("--config", default="configs/best.yaml")
    ap.add_argument("--target-pos-rate", type=float, default=0.625,
                    help="test 的阳性率（624 张中 390 张）")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from .utils import load_config
    cfg = load_config(a.config)
    device = torch.device(cfg.get("device", "cpu"))
    ck = torch.load(a.ckpt, map_location=device)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    _, val_loader, _ = get_loaders(cfg)
    p, y = predict(model, val_loader, device)
    p = np.asarray(p).ravel()
    y = np.asarray(y).ravel().astype(int)

    full = compute_metrics(y, p)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    pos_rate = n_pos / len(y)
    print("val 全量: n=%d  阳性 %.1f%%  ACC %.4f  AUC %.4f  F1 %.4f" % (
        len(y), pos_rate * 100, full["acc"], full["auc"], full["f1"]))

    # 保持阴性全用、按目标先验抽阳性，使阳性率 = target
    r = a.target_pos_rate
    keep_pos = int(round(n_neg * r / (1 - r)))
    if keep_pos >= n_pos:
        print("目标先验高于 val 先验，无法用下采样模拟；跳过。")
        return 0
    rng = np.random.RandomState(a.seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    accs = []
    for _ in range(a.repeats):
        sub = np.concatenate([rng.choice(pos_idx, keep_pos, replace=False), neg_idx])
        accs.append(acc_at(y[sub], p[sub]))
    accs = np.array(accs)
    m = accs.mean()

    print("先验对齐后（阳性 %.1f%%，每次 %d 张）: ACC %.4f ± %.4f（%d 次自助）" % (
        r * 100, keep_pos + n_neg, m, accs.std(), a.repeats))
    print()
    print("--- 落差拆解（以 ACC 为例）---")
    print("val ACC                     : %.4f" % full["acc"])
    print("仅改先验后的 ACC            : %.4f  （Δ %+.4f）" % (m, m - full["acc"]))
    print("真实 test ACC               : 0.8349")
    print("先验解释的部分              : %.4f / 全部落差 %.4f = %.0f%%" % (
        m - full["acc"], 0.8349 - full["acc"],
        max(0.0, (m - full["acc"]) / (0.8349 - full["acc"])) * 100))
    print("剩余（真实分布偏移 + 阈值失配）: %+.4f" % (0.8349 - m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
