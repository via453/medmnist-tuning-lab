"""基础预测模型：MLP / SmallCNN / TinyViT。

MLP 与 SmallCNN 的骨架来自李沐 D10 / D23；TinyViT 对应笔记 C4 自注意力与 C5 Transformer。
三者参数量刻意放在同一量级，便于做「结构 vs 容量」的对照。
"""
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


class TinyViT(nn.Module):
    """微缩 ViT（C4 自注意力 + C5 Transformer 的工程版）。

    28x28 按 patch=4 切成 7x7=49 个 token，加 1 个 cls token；
    2 层 pre-norm TransformerEncoder（4 头、FFN 2 倍宽），末尾取 cls token 分类。
    参数量刻意压到与 conv_width=64 的 SmallCNN 同一量级，
    用来回答「笔记里的注意力放到 28x28 小图上到底行不行」。
    """

    def __init__(self, n_channels=1, n_classes=2, img=28, patch=4, dim=64,
                 depth=2, heads=4, dropout=0.1, **kwargs):
        super().__init__()
        n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(n_channels, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)          # B, n, dim
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


MODEL_REGISTRY = {"mlp": MLP, "smallcnn": SmallCNN, "vit": TinyViT}


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
        width=int(cfg.get("width", 128)),   # MLP 容量旋钮，供容量对齐实验使用
        img=int(cfg.get("size", 28) or 28),
        patch=int(cfg.get("patch", 4)),
        dim=int(cfg.get("dim", 64)),
        depth=int(cfg.get("depth", 2)),
        heads=int(cfg.get("heads", 4)),
    )
