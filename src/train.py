"""训练循环：交叉熵 + 优化器 + 学习率排程 + 早停。"""
import time

import torch
import torch.nn as nn

from .data import class_weights_from_loader
from .evaluate import compute_metrics, predict
from .model import build_model
from .utils import count_params, set_seed


def make_optimizer(model, cfg):
    name = cfg.get("optimizer", "adam").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=float(cfg.get("momentum", 0.9)),
                               weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def make_scheduler(optimizer, cfg, total_epochs):
    name = cfg.get("scheduler", "none")
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_epochs // 3), gamma=0.5)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs))
    return None


def make_criterion(cfg, train_loader, device):
    weight = None
    counts = None
    if cfg.get("class_weight") == "balanced":
        weight, counts = class_weights_from_loader(
            train_loader, n_classes=int(cfg.get("n_classes", 2)))
        weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weight, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    return criterion, counts


def fgsm_perturb(model, x, y, eps, criterion):
    """在输入上做单步 FGSM：x' = clamp(x + eps*sign(∇ₓL))，归一化范围 [-1,1]。

    制扰动时临时切 eval()：否则每步会多更新一次 BN 的 running stats，
    等于用对抗样本把 BN 统计量污染两遍（见 docs/PITFALLS.md 第 13 条）。
    """
    was_training = model.training
    model.eval()
    x_in = x.clone().detach().requires_grad_(True)
    loss = criterion(model(x_in), y)
    grad, = torch.autograd.grad(loss, x_in)
    x_adv = (x_in + eps * grad.sign()).clamp(-1.0, 1.0).detach()
    if was_training:
        model.train()
    return x_adv


def train_one_epoch(model, loader, criterion, optimizer, device,
                    adv_eps=0.0, adv_ratio=0.5):
    """一个 epoch。

    adv_eps > 0 时开启对抗训练：每个 batch 里按 adv_ratio 的比例把样本替换成
    FGSM 对抗样本，干净样本与对抗样本一起算交叉熵（1:1 即 adv_ratio=0.5）。
    """
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.view(-1).long().to(device)
        if adv_eps and adv_eps > 0:
            k = int(round(x.size(0) * float(adv_ratio)))
            k = max(1, min(k, x.size(0) - 1)) if x.size(0) > 1 else 0
            if k > 0:
                x_adv = fgsm_perturb(model, x[:k], y[:k], adv_eps, criterion)
                x = torch.cat([x[k:], x_adv], dim=0)
                y = torch.cat([y[k:], y[:k]], dim=0)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def fit(cfg, loaders, run_id="run", ckpt_path=None, verbose=True):
    """跑一组配置，返回含历史、最佳 val 指标与耗时的 dict。不写任何结果文件。"""
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(cfg.get("device", "cpu"))

    train_loader, val_loader, _ = loaders
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    epochs = int(cfg.get("epochs", 20))
    scheduler = make_scheduler(optimizer, cfg, epochs)
    criterion, counts = make_criterion(cfg, train_loader, device)

    base_lr = float(cfg.get("lr", 1e-3))
    warm = int(cfg.get("warmup_epochs", 0) or 0)
    patience = int(cfg.get("early_stop_patience", 5) or 0)
    adv_eps = float(cfg.get("adv_train_eps", 0.0) or 0.0)
    adv_ratio = float(cfg.get("adv_train_ratio", 0.5) or 0.5)

    history, best_auc, best_acc, best_epoch, bad = [], -1.0, 0.0, 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if warm > 0 and epoch <= warm:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * epoch / warm
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     adv_eps=adv_eps, adv_ratio=adv_ratio)
        probs, labels = predict(model, val_loader, device)
        metrics = compute_metrics(labels, probs)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_auc": metrics["auc"], "val_acc": metrics["acc"]})

        if metrics["auc"] > best_auc:
            best_auc, best_acc, best_epoch, bad = metrics["auc"], metrics["acc"], epoch, 0
            if ckpt_path:
                import os
                from .utils import ensure_dir
                ensure_dir(os.path.dirname(ckpt_path) or ".")
                torch.save({"model": model.state_dict(), "config": cfg,
                            "epoch": epoch, "val_auc": best_auc}, ckpt_path)
        else:
            bad += 1

        if scheduler is not None and (warm == 0 or epoch > warm):
            scheduler.step()
        if verbose:
            print("  epoch %2d/%d  loss %.4f  val_auc %.4f  val_acc %.4f" % (
                epoch, epochs, train_loss, metrics["auc"], metrics["acc"]))
        if patience and bad >= patience:
            if verbose:
                print("  early stop at epoch %d" % epoch)
            break

    return {
        "run_id": run_id,
        "history": history,
        "val_auc": best_auc,
        "val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "duration_min": round((time.time() - t0) / 60.0, 3),
        "params": count_params(model),
        "class_counts": counts,
        "adv_train_eps": adv_eps,
        "adv_train_ratio": adv_ratio,
        "model": model,
    }
