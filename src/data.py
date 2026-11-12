"""MedMNIST 数据加载：固定官方 train / val / test 划分，val 选配置、test 只评一次。"""
import numpy as np
import torch
from medmnist import BreastMNIST, PneumoniaMNIST, RetinaMNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

REGISTRY = {
    "pneumoniamnist": PneumoniaMNIST,
    "breastmnist": BreastMNIST,
    "retinamnist": RetinaMNIST,
}

# 与 medmnist 3.0.2 的 INFO 一致，便于冒烟时核对
EXPECTED_SPLITS = {
    "pneumoniamnist": (4708, 524, 624),
    "breastmnist": (546, 78, 156),
    "retinamnist": (1080, 120, 400),
}


def _normalize():
    return transforms.Normalize(mean=[0.5], std=[0.5])


def get_transforms(cfg, train=False):
    ops = [transforms.ToTensor(), _normalize()]
    if train:
        aug = cfg.get("augment", "flip")
        # 注意顺序：随机增广必须在 ToTensor 之前作用于 PIL 图
        pre = []
        if aug in ("flip", "flip_rot"):
            pre.append(transforms.RandomHorizontalFlip())
        if aug == "flip_rot":
            pre.append(transforms.RandomRotation(10))
        ops = pre + ops
    return transforms.Compose(ops)


def _dataset(cfg, split):
    name = cfg.get("dataset", "pneumoniamnist")
    if name not in REGISTRY:
        raise ValueError("unknown dataset: %s" % name)
    kwargs = dict(split=split, download=True,
                  transform=get_transforms(cfg, train=(split == "train")))
    if cfg.get("size") and cfg["size"] != 28:
        kwargs["size"] = cfg["size"]
    if cfg.get("data_root"):
        kwargs["root"] = cfg["data_root"]
    return REGISTRY[name](**kwargs)


def _limit(ds, n):
    if n is None:
        return ds
    n = min(int(n), len(ds))
    return Subset(ds, list(range(n)))


def get_loaders(cfg):
    """返回 (train, val, test) 三个 DataLoader。"""
    train_ds = _limit(_dataset(cfg, "train"), cfg.get("limit_train"))
    val_ds = _limit(_dataset(cfg, "val"), cfg.get("limit_val"))
    test_ds = _dataset(cfg, "test")

    common = dict(batch_size=cfg.get("batch_size", 64),
                  num_workers=cfg.get("num_workers", 0),
                  pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def class_weights_from_loader(loader, n_classes=2):
    """balanced 模式：按训练集中各类别样本数给交叉熵加权。"""
    counts = np.zeros(n_classes, dtype=np.float64)
    for _, y in loader:
        labels = y.view(-1).numpy().astype(int)
        for c in range(n_classes):
            counts[c] += (labels == c).sum()
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32), counts.astype(int).tolist()
