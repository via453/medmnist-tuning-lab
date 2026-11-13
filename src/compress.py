"""C14 网络压缩：剪枝（+微调）与蒸馏（温度扫描）。

对应笔记「五件动手的事」第 4 件（剪枝 80% 微调 3 epoch），
以及 C14 里「温度 T 控制软标签平滑程度」那句——这里把 T 从 1 扫到 8 实测。
全部只评 val，不碰 test。
"""
import argparse
import copy
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from .data import get_loaders
from .evaluate import compute_metrics, predict
from .model import build_model
from .train import train_one_epoch
from .utils import count_params, ensure_dir, load_config, set_seed


def load_teacher(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["config"]


def eval_val(model, loader, device):
    p, y = predict(model, loader, device)
    return compute_metrics(y, p)


def prune_model(model, amount):
    """对所有权重矩阵做 L1 非结构化剪枝（BN 参数、bias 不动）。"""
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(mod, name="weight", amount=amount)


def sparsity(model):
    total = zero = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight.detach()
            total += w.numel()
            zero += int((w == 0).sum())
    return (zero / total if total else 0.0), total - zero


def finetune(model, train_loader, val_loader, device, epochs=3, lr=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        train_one_epoch(model, train_loader, crit, opt, device)
    model.eval()
    return eval_val(model, val_loader, device)


def train_student(cfg, train_loader, val_loader, teacher, device, T=1.0, alpha=0.5):
    """T<=1 且无 teacher（teacher=None）时是纯 CE 从头训练；否则做 KD。"""
    set_seed(int(cfg.get("seed", 0)))
    student = build_model(cfg).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=float(cfg.get("lr", 1e-3)))
    crit = nn.CrossEntropyLoss()
    epochs = int(cfg.get("epochs", 15))
    best, best_acc = -1.0, 0.0
    for _ in range(epochs):
        student.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.view(-1).long().to(device)
            logits = student(x)
            loss = crit(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                kd = F.kl_div(F.log_softmax(logits / T, 1),
                              F.softmax(t_logits / T, 1),
                              reduction="batchmean") * (T * T)
                loss = (1 - alpha) * loss + alpha * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = eval_val(student, val_loader, device)
        if m["auc"] > best:
            best, best_acc = m["auc"], m["acc"]
    return best, best_acc, student


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/best.pth")
    p.add_argument("--config", default="configs/best.yaml", help="用于取 data / batch / epochs")
    p.add_argument("--prune", default="0.5,0.8,0.9")
    p.add_argument("--prune-epochs", type=int, default=3)
    p.add_argument("--temps", default="1,4,8")
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--kd-epochs", type=int, default=15)
    p.add_argument("--out", default="results/compress.csv")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    device = torch.device(cfg.get("device", "cpu"))
    train_loader, val_loader, _ = get_loaders(cfg)
    teacher, t_cfg = load_teacher(a.ckpt, device)
    base = eval_val(teacher, val_loader, device)
    rows = []

    def record(kind, detail, m, model, t0, extra_zero=None):
        nz = extra_zero if extra_zero is not None else count_params(model)
        sp, nonzero = (extra_zero if extra_zero is not None else (0.0, nz))
        rows.append({"kind": kind, "detail": detail,
                     "val_auc": round(m["auc"], 6), "val_acc": round(m["acc"], 6),
                     "params_total": count_params(model), "params_nonzero": nonzero,
                     "sparsity": round(sp, 4), "duration_min": round((time.time() - t0) / 60, 3)})
        print("  %-10s %-14s auc %.4f  acc %.4f  nonzero %d/%d (sparse %.1f%%)  %.2f min" % (
            kind, detail, m["auc"], m["acc"], nonzero, count_params(model), sp * 100,
            (time.time() - t0) / 60))

    print("=== 基线：teacher (conv_width=64) ===")
    t0 = time.time()
    record("baseline", "w64", base, teacher, t0, (0.0, count_params(teacher)))

    print("=== 剪枝 + 微调 ===")
    for amount in [float(x) for x in a.prune.split(",")]:
        t0 = time.time()
        m = copy.deepcopy(teacher)
        prune_model(m, amount)
        sp, nonzero = sparsity(m)
        met = finetune(m, train_loader, val_loader, device, epochs=a.prune_epochs)
        record("prune", "%.0f%%+ft%d" % (amount * 100, a.prune_epochs), met, m, t0, (sp, nonzero))

    print("=== 蒸馏 / 从头训练（学生 conv_width=%d）===" % a.student_width)
    s_cfg = copy.deepcopy(t_cfg)
    s_cfg.update({"conv_width": a.student_width, "epochs": a.kd_epochs,
                  "lr": float(cfg.get("lr", 1e-3)), "seed": int(cfg.get("seed", 0))})
    s_ref = build_model(s_cfg)
    print("  学生参数量:", count_params(s_ref), "(teacher %d)" % count_params(teacher))

    t0 = time.time()
    auc, acc, stud = train_student(s_cfg, train_loader, val_loader, None, device)
    record("scratch", "w%d-CE" % a.student_width, {"auc": auc, "acc": acc}, stud, t0)

    for T in [float(x) for x in a.temps.split(",")]:
        t0 = time.time()
        auc, acc, stud = train_student(s_cfg, train_loader, val_loader, teacher, device, T=T)
        record("distill", "w%d-T%g" % (a.student_width, T), {"auc": auc, "acc": acc}, stud, t0)

    ensure_dir("results")
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "detail", "val_auc", "val_acc",
                                          "params_total", "params_nonzero",
                                          "sparsity", "duration_min"])
        w.writeheader()
        w.writerows(rows)
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
