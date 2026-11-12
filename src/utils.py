"""通用工具：配置读取、随机种子、参数量统计。"""
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    """读取 YAML 配置，返回 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping: %s" % path)
    return cfg


def merge_config(base, override):
    """用命令行覆盖项更新配置（只覆盖显式给出的键）。"""
    out = dict(base)
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def set_seed(seed):
    """三处都要设，缺一处结果就不可复现（完整教程 3.2 的可复现性四件套）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def append_csv(path, row, fieldnames):
    """追加一行到 CSV；文件不存在时先写表头。失败组也要记，不许删。"""
    import csv

    ensure_dir(os.path.dirname(path) or ".")
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def short_git_sha():
    """把结果锁到具体代码版本；非 git 环境下返回 unknown。"""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"
