"""C10 可解释性：Grad-CAM（不依赖第三方库，纯 torch hook 实现）。

对最后一层卷积的输出梯度做加权求和，看模型「在看哪里」。
额外给一个可核对的数字：CAM 响应落在图像四周 4 像素边框的比例 ——
如果模型主要看边框（体位标记、黑边），那所有指标都要重新解释。
只读 val，不碰 test。
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from .data import get_loaders
from .model import build_model
from .utils import ensure_dir


class GradCAM:
    def __init__(self, model):
        target = None
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                target = mod
        if target is None:
            raise ValueError("模型里没有 Conv2d，Grad-CAM 不适用（MLP 无卷积层）")
        self.act = None
        self.grad = None
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.act = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def __call__(self, x, cls=1):
        self.act = self.grad = None
        model_out = self.model(x)
        model_out[:, cls].sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)          # 每个通道的权重
        cam = torch.relu((w * self.act).sum(1))[0]            # H, W
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def bind(self, model):
        self.model = model
        return self


def border_ratio(cam, margin=4):
    """边框响应占比。margin 以 cam 的像素为单位，返回 (占比, 边框面积占比)。"""
    h, w = cam.shape
    mask = np.ones((h, w), dtype=bool)
    mask[margin:h - margin, margin:w - margin] = False
    total = cam.sum() + 1e-8
    return float(cam[mask].sum() / total), float(mask.mean())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/best.pth")
    p.add_argument("--config", default="configs/best.yaml")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--fig", default="figures/gradcam.png")
    a = p.parse_args(argv)

    from .utils import load_config
    cfg = load_config(a.config)
    device = torch.device(cfg.get("device", "cpu"))
    ck = torch.load(a.ckpt, map_location=device)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(True)

    _, val_loader, _ = get_loaders(cfg)
    cam_engine = GradCAM(model).bind(model)

    imgs, cams, probs, labels = [], [], [], []
    for x, y in val_loader:
        for i in range(x.size(0)):
            if len(imgs) >= a.n:
                break
            xi = x[i:i + 1].to(device)
            cam = cam_engine(xi)
            # 最后一层卷积是 14x14，先上采样到 28x28，边框口径才与图像一致
            cam = torch.nn.functional.interpolate(
                torch.tensor(cam)[None, None], size=(28, 28),
                mode="bilinear", align_corners=False)[0, 0].numpy()
            with torch.no_grad():
                pr = torch.softmax(model(xi), 1)[0, 1].item()
            imgs.append(x[i, 0].numpy())
            cams.append(cam)
            probs.append(pr)
            labels.append(int(y[i].view(-1)[0]))
        if len(imgs) >= a.n:
            break

    ratios = [border_ratio(c)[0] for c in cams]
    area = border_ratio(cams[0])[1]
    print("样本 | 标签 | P(肺炎) | CAM 边框占比（边框面积占比 %.2f）" % area)
    for i, (l, pr, r) in enumerate(zip(labels, probs, ratios)):
        print("  %2d   %d     %.3f     %.3f" % (i, l, pr, r))
    print("边框占比 均值 %.3f；相对边框面积 %.2f 的富集倍数 %.2fx" % (
        float(np.mean(ratios)), area, float(np.mean(ratios)) / area))
    print("判读：富集倍数 ≈1 表示注意力在图像上大致均匀；>1.5 才算明显偏向边缘伪影")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(imgs)
        fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 4.0))
        for i in range(n):
            axes[0, i].imshow(imgs[i], cmap="gray")
            axes[0, i].set_title("y=%d p=%.2f" % (labels[i], probs[i]), fontsize=8)
            axes[0, i].axis("off")
            axes[1, i].imshow(imgs[i], cmap="gray")
            axes[1, i].imshow(cams[i], cmap="jet", alpha=0.55,
                              extent=(0, cams[i].shape[1], cams[i].shape[0], 0))
            axes[1, i].set_title("border %.2f" % ratios[i], fontsize=8)
            axes[1, i].axis("off")
        fig.suptitle("Grad-CAM on PneumoniaMNIST (val) · last conv layer", fontsize=10)
        fig.tight_layout()
        ensure_dir("figures")
        fig.savefig(a.fig, dpi=140)
        print("figure:", a.fig)
    except Exception as ex:
        print("plot skipped:", ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
