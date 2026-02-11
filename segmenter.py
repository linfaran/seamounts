import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)  # B,D,H',W'
        B, D, H, W = x.shape
        N = H*W
        x = x.flatten(2).transpose(1, 2)  # B,N,D
        return x, (H, W)

class TransformerEncoder(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4, depth=6, drop=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, int(dim*mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(int(dim*mlp_ratio), dim),
                )
            ]))
    def forward(self, x):
        for ln1, attn, ln2, mlp in self.layers:
            h = x
            x = ln1(x)
            x, _ = attn(x, x, x, need_weights=False)
            x = x + h
            h = x
            x = ln2(x)
            x = mlp(x) + h
        return x

class SegmenterMini(nn.Module):
    """
    简化 Segmenter：
    - ViT 编码得到 patch tokens: B,N,D
    - 为每个类别设置一个 class token（可学习），通过注意力/MLP 映射到 mask logits
    - 最后把 patch mask 重排到 H'×W'，再上采样到原图
    * 这里实现的是“mask transformer”极简观点：用线性头生成每类的 per-token logits
    """
    def __init__(self, n_channels=3, n_classes=19, embed_dim=256, patch=16, depth=6):
        super().__init__()
        self.patch = patch
        self.embed = PatchEmbed(n_channels, embed_dim, patch)
        self.encoder = TransformerEncoder(embed_dim, num_heads=8, depth=depth)
        # 类别原型（可选，不强制用注意力，这里简单线性头）
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        tokens, (h, w) = self.embed(x)     # B,N,D
        z = self.encoder(tokens)           # B,N,D
        logits_tok = self.head(z)          # B,N,C
        logits = logits_tok.transpose(1,2).reshape(x.size(0), -1, h, w)  # B,C,H',W'
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits
