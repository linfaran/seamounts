import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- ViT 编码器（简化） --------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        # B, C, H, W -> B, N, D
        x = self.proj(x)
        B, D, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2)  # B,N,D
        return x, (H, W)

class TransformerEncoder(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4.0, depth=4, drop=0.0):
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
            # Self-attn
            h = x
            x = ln1(x)
            x, _ = attn(x, x, x, need_weights=False)
            x = x + h
            # MLP
            h = x
            x = ln2(x)
            x = mlp(x) + h
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p='same'):
        super().__init__()
        if p == 'same':
            p = (k-1)//2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c+skip_c, out_c, 3, 1, 'same')
        self.conv2 = ConvBNReLU(out_c, out_c, 3, 1, 'same')
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)

class TransUNet(nn.Module):
    """
    轻量 TransUNet：
    - ViT 编码器提取全局表示（patch=16）
    - 将token还原回 feature map 后，走 UNet 解码器并与浅层CNN特征融合
    """
    def __init__(self, n_channels=3, n_classes=19, embed_dim=256, patch=16, vit_depth=4):
        super().__init__()
        # 浅层CNN作为 skip features
        self.enc1 = nn.Sequential(ConvBNReLU(n_channels, 32), ConvBNReLU(32, 32))
        self.pool1 = nn.MaxPool2d(2)  # /2
        self.enc2 = nn.Sequential(ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.pool2 = nn.MaxPool2d(2)  # /4

        # ViT 编码器（对 /4 的特征再做 patch-embedding）
        self.pre_proj = ConvBNReLU(64, 64, 3, 1, 'same')
        self.patch_embed = PatchEmbed(64, embed_dim, patch=patch)  # >> 下采样更多
        self.pos_emb = None  # 简化：不显式pos，或运行时按N创建（省略）
        self.encoder = TransformerEncoder(dim=embed_dim, depth=vit_depth, num_heads=8)

        # token -> fmap
        self.post_proj = nn.Conv2d(embed_dim, 256, 1)

        # 解码器
        self.up2 = UpBlock(256, 64, 128)  # 与 enc2 融合（/4）
        self.up1 = UpBlock(128, 32, 64)   # 与 enc1 融合（/2）
        self.head = nn.Sequential(
            ConvBNReLU(64, 64),
            nn.Conv2d(64, n_classes, 1)
        )

    def forward(self, x):
        H, W = x.shape[2:]

        # 浅层特征
        c1 = self.enc1(x)                 # /1
        p1 = self.pool1(c1)               # /2
        c2 = self.enc2(p1)                # /2
        p2 = self.pool2(c2)               # /4

        # ViT 编码
        pre = self.pre_proj(p2)           # /4
        tokens, (h, w) = self.patch_embed(pre)  # 进一步下采样
        z = self.encoder(tokens)          # B, N, D

        # 还原为 feature map（/4 * 额外patch缩小倍数）
        z = z.transpose(1, 2).reshape(x.size(0), -1, h, w)
        z = self.post_proj(z)

        # 解码融合
        d2 = self.up2(z, c2)              # /4
        d1 = self.up1(d2, c1)             # /2
        out = F.interpolate(self.head(d1), size=(H, W), mode='bilinear', align_corners=False)
        return out
