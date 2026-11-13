"""延伸实验 A：对抗训练 —— 把 ε=0.03 的坑填上一半。

做法（与 C9 的 FGSM 攻击严格同口径）：
    训练时每个 batch 按 adv_ratio 比例把样本替换成 FGSM(ε_train) 对抗样本，
    与干净样本一起算交叉熵；测试时重扫 ε 曲线，和「干净训练」的对照组比。

对照组不是复用 outputs/best.pth，而是在本脚本里用同样的 seed / 轮数重训一遍——
否则「训练集顺序、早停轮次」都会变成混杂因素。

    python -m src.adv_train                     # 3 seed × (clean, adv) = 6 次训练
    python -m src.adv_train --seeds 0 --epochs 3 --limit-train 512   # 冒烟

输出：results/adv_train.csv（每次训练一行）、results/adv_curve.csv（ε 曲线均值±std）、
      figures/fgsm_adv_compare.png。**只评 val，不碰 test。**
"""
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch

from .data import get_loaders
from .fgsm import fgsm_eval
from .train import fit
from .utils import count_params, ensure_dir, load_config, short_git_sha

TRAIN_FIELDS = ["run_id", "timestamp", "seed", "git_commit", "status", "note",
                "model", "conv_width", "adv_train_eps", "adv_train_ratio",
                "epochs_run", "duration_min", "params", "val_auc", "val_acc",
                "val_auc_adv003", "val_acc_adv003"]
CURVE_FIELDS = ["eps", "n_seed", "clean_auc_mean", "clean_auc_std",
                "clean_acc_mean", "clean_acc_std",
                "adv_auc_mean", "adv_auc_std", "adv_acc_mean", "adv_acc_std"]


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def main(argv=None):
    p = argparse.ArgumentParser(description="adversarial training vs clean training")
    p.add_argument("--config", default="configs/best.yaml")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--eps-train", type=float, default=0.03,
                   help="训练时混入的 FGSM ε（取攻击曲线已经崩掉的那个点）")
    p.add_argument("--ratio", type=float, default=0.5, help="batch 中对抗样本比例")
    p.add_argument("--eps", default="0,0.005,0.01,0.02,0.03,0.05,0.08,0.1,0.2,0.3")
    p.add_argument("--epochs", type=int, help="覆盖 epochs（冒烟用）")
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--out", default="results/adv_train.csv")
    p.add_argument("--curve-out", default="results/adv_curve.csv")
    p.add_argument("--fig", default="figures/fgsm_adv_compare.png")
    p.add_argument("--ckpt-dir", default="outputs")
    args = p.parse_args(argv)

    eps_list = [float(x) for x in args.eps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    base_cfg = load_config(args.config)
    device = torch.device(base_cfg.get("device", "cpu"))
    ensure_dir(args.ckpt_dir)

    train_rows, curves = [], {"clean": [], "adv": []}
    for seed in seeds:
        for tag, adv_eps in (("clean", 0.0), ("adv", args.eps_train)):
            cfg = dict(base_cfg)
            cfg.update({"seed": seed, "adv_train_eps": adv_eps,
                        "adv_train_ratio": args.ratio})
            if args.epochs:
                cfg["epochs"] = args.epochs
            if args.limit_train:
                cfg["limit_train"] = args.limit_train
            if args.limit_val:
                cfg["limit_val"] = args.limit_val
            run_id = "%s_w%s_s%d" % (tag, cfg.get("conv_width", "?"), seed)
            ckpt = os.path.join(args.ckpt_dir, "adv_%s_s%d.pth" % (tag, seed))
            print("[%s] seed=%d  adv_train_eps=%.3f" % (run_id, seed, adv_eps))

            row = {"run_id": run_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "seed": seed, "git_commit": short_git_sha(), "status": "ok",
                   "model": cfg.get("model", "smallcnn"),
                   "conv_width": cfg.get("conv_width"),
                   "adv_train_eps": adv_eps, "adv_train_ratio": args.ratio,
                   "note": "clean 对照组" if adv_eps == 0 else
                           "训练混入 FGSM eps=%.3f ratio=%.2f" % (adv_eps, args.ratio)}
            try:
                loaders = get_loaders(cfg)
                val_loader = loaders[1]
                res = fit(cfg, loaders, run_id=run_id, ckpt_path=ckpt, verbose=False)
                # 曲线用「val 最优轮」的权重，与主实验的选模口径一致
                ck = torch.load(ckpt, map_location=device)
                from .model import build_model
                model = build_model(ck["config"])
                model.load_state_dict(ck["model"])
                model.to(device).eval()

                curve = []
                for eps in eps_list:
                    m = fgsm_eval(model, val_loader, device, eps)
                    curve.append((eps, m["auc"], m["acc"]))
                curves[tag].append(curve)

                at = [c for c in curve if abs(c[0] - args.eps_train) < 1e-9]
                row.update({
                    "epochs_run": res["epochs_run"],
                    "duration_min": res["duration_min"],
                    "params": count_params(model),
                    "val_auc": round(curve[0][1], 6),
                    "val_acc": round(curve[0][2], 6),
                    "val_auc_adv003": round(at[0][1], 6) if at else "",
                    "val_acc_adv003": round(at[0][2], 6) if at else "",
                })
                print("    clean auc %.4f | eps=%.3f auc %.4f | %.2f min" % (
                    curve[0][1], args.eps_train,
                    at[0][1] if at else float("nan"), res["duration_min"]))
            except Exception as exc:
                row.update({"status": "failed",
                            "note": "%s: %s" % (type(exc).__name__, exc)})
                print("    FAILED: %s: %s" % (type(exc).__name__, exc))
            train_rows.append(row)

    write_csv(args.out, train_rows, TRAIN_FIELDS)

    # ---- ε 曲线：按 seed 求均值与 std（ddof=1）----
    curve_rows = []
    if curves["clean"] and curves["adv"]:
        for i, eps in enumerate(eps_list):
            r = {"eps": eps, "n_seed": len(curves["clean"])}
            for tag in ("clean", "adv"):
                aucs = [c[i][1] for c in curves[tag]]
                accs = [c[i][2] for c in curves[tag]]
                am, asd = mean_std(aucs)
                cm, csd = mean_std(accs)
                r.update({"%s_auc_mean" % tag: round(am, 6),
                          "%s_auc_std" % tag: round(asd, 6),
                          "%s_acc_mean" % tag: round(cm, 6),
                          "%s_acc_std" % tag: round(csd, 6)})
            curve_rows.append(r)
        write_csv(args.curve_out, curve_rows, CURVE_FIELDS)

        print("\n=== ε 曲线（val, n_seed=%d）===" % len(curves["clean"]))
        print("   eps   clean_auc   adv_auc   clean_acc   adv_acc")
        for r in curve_rows:
            print("  %.3f   %.4f     %.4f    %.4f      %.4f" % (
                r["eps"], r["clean_auc_mean"], r["adv_auc_mean"],
                r["clean_acc_mean"], r["adv_acc_mean"]))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            e = [r["eps"] for r in curve_rows]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, key, ylabel in ((axes[0], "auc", "val AUC"), (axes[1], "acc", "val ACC")):
                for tag, color, label in (("clean", "#c0392b", "clean training"),
                                          ("adv", "#2471a3", "adv. training (eps=%.3f)" % args.eps_train)):
                    m = np.array([r["%s_%s_mean" % (tag, key)] for r in curve_rows])
                    s = np.array([r["%s_%s_std" % (tag, key)] for r in curve_rows])
                    ax.plot(e, m, "o-", color=color, label=label, markersize=4)
                    ax.fill_between(e, m - s, m + s, color=color, alpha=0.15)
                ax.axvline(args.eps_train, color="gray", ls="--", lw=1)
                ax.set_xlabel("FGSM epsilon (eval)")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            axes[0].set_title("Robustness vs epsilon (PneumoniaMNIST val, %d seeds)"
                              % len(curves["clean"]))
            fig.tight_layout()
            ensure_dir(os.path.dirname(args.fig) or ".")
            fig.savefig(args.fig, dpi=140)
            print("figure:", args.fig)
        except Exception as exc:
            print("plot skipped:", exc)

    print("written:", args.out, "|", args.curve_out)
    print("提醒：曲线用 val 最优轮权重；test 仍未触碰。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
