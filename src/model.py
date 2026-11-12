"""基础预测模型：MLP 与 SmallCNN。骨架来自李沐 D10 / D23，参数量可手算核对。"""
import torch
import torch.nn as nn


def _act(name):
    return nn.GELU() if name == "gelu" else nn.ReLU(inplace=True)


class MLP(nn.Module):
    """最低可训练基线：28x28 展平 -> 128 -> 2。"""

    def __init__(self, n_channels=1, n_classes=2, width=128, dropout=0.2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * 28 * 28, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """2~3 个 conv + BN + 全局平均池化 + FC。

    pooling='light' 时只做一次 2x2 池化；'none' 时完全不池化。
    C3 订正版指出：医学影像的微小结节会在池化中丢失，28x28 小图尤其不能堆池化。
    """

    def __init__(self, n_channels=1, n_classes=2, conv_width=32, use_bn=True,
                 activation="relu", pooling="light", dropout=0.2, **kwargs):
        super().__init__()
        w, w2 = conv_width, conv_width * 2

        def block(in_ch, out_ch, pool):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_act(activation))
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers

        layers = []
        layers += block(n_channels, w, pool=(pooling == "light"))
        layers += block(w, w2, pool=False)
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                   nn.Linear(w2, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {"mlp": MLP, "smallcnn": SmallCNN}


def build_model(cfg):
    name = cfg.get("model", "smallcnn")
    if name not in MODEL_REGISTRY:
        raise ValueError("unknown model: %s (choose from %s)" % (
            name, list(MODEL_REGISTRY)))
    return MODEL_REGISTRY[name](
        n_channels=cfg.get("n_channels", 1),
        n_classes=cfg.get("n_classes", 2),
        conv_width=cfg.get("conv_width", 32),
        use_bn=cfg.get("use_bn", True),
        activation=cfg.get("activation", "relu"),
        pooling=cfg.get("pooling", "light"),
        dropout=cfg.get("dropout", 0.2),
        width=128,
    )
