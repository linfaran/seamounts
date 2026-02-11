import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TinyBackbone(nn.Module):
    def __init__(self, in_ch=3, out_ch=256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 3, 2),    # /2
            ConvBNReLU(64, 64, 3, 1),
            nn.MaxPool2d(2),                # /4
            ConvBNReLU(64, 128, 3, 2),      # /8
            ConvBNReLU(128, out_ch, 3, 1),
        )
        self.out_channels = out_ch
    def forward(self, x):
        return self.net(x)  # B, D, H', W'

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4., drop=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim),
        )
    def forward(self, q, k, v):
        # Cross-Attention: q queries vs. encoder tokens
        h = q
        q = self.ln1(q)
        q, _ = self.cross_attn(q, k, v, need_weights=False)
        q = q + h
        # Self-Attention among queries
        h = q
        q = self.ln2(q)
        q, _ = self.self_attn(q, q, q, need_weights=False)
        q = q + h
        # MLP
        h = q
        q = self.ln3(q)
        q = self.mlp(q) + h
        return q

class TransformerDecoder(nn.Module):
    def __init__(self, dim=256, layers=3, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(dim, num_heads) for _ in range(layers)])
    def forward(self, q, kv):
        for layer in self.layers:
            q = layer(q, kv, kv)
        return q

class Mask2FormerMini(nn.Module):
    """
    极简 Mask2Former 风格：
    - Backbone: Tiny CNN -> F (B,D,H',W')
    - Flatten F -> tokens (B,N,D) 供 Decoder cross-attn
    - Learnable query embeddings -> Decoder 输出 Q (B,Nq,D)
    - mask_embed = Linear(Q) -> B,Nq,D
    - masks = einsum(mask_embed, F) -> B,Nq,H',W'
    - class_logits = Linear(Q) -> B,Nq,C
    - 将 Nq 个查询掩码按类别权重聚合为 C 通道 logits 返回（教学用）
    """
    def __init__(self, n_channels=3, n_classes=19, dim=256, num_queries=100, dec_layers=3):
        super().__init__()
        self.backbone = TinyBackbone(n_channels, out_ch=dim)
        self.query_embed = nn.Embedding(num_queries, dim)
        self.decoder = TransformerDecoder(dim=dim, layers=dec_layers, num_heads=8)
        self.class_head = nn.Linear(dim, n_classes)
        self.mask_head = nn.Linear(dim, dim)  # 生成 mask embedding 用于与 F 点积

    def forward(self, x):
        B, C, H, W = x.shape
        Fmap = self.backbone(x)                    # B,D,h,w
        D, h, w = Fmap.shape[1], Fmap.shape[2], Fmap.shape[3]
        tokens = Fmap.flatten(2).transpose(1, 2)   # B, N=h*w, D

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # B, Nq, D
        Q = self.decoder(queries, tokens)          # B, Nq, D

        class_logits = self.class_head(Q)          # B, Nq, C
        mask_embed   = self.mask_head(Q)           # B, Nq, D

        # 逐像素内积得到查询掩码（B, Nq, h, w）
        Fnorm = Fmap                               # B,D,h,w
        masks = torch.einsum('bqd,bdhw->bqhw', mask_embed, Fnorm)

        # 把 Nq 查询掩码聚合成 C 通道 logits（按类别打分做加权和：softmax over queries）
        prob_q = F.softmax(class_logits, dim=1)    # B, Nq, C（跨 query softmax 也可；此处按 Nq 做 softmax 更合理）
        prob_q = F.softmax(class_logits.transpose(1,2), dim=2).transpose(1,2)  # B,Nq,C 变成 B,Nq,C（对每类在Nq上softmax）
        # 聚合：对每个类 c，按查询的权重对 mask 求和
        masks = masks.unsqueeze(2)                 # B, Nq, 1, h, w
        prob_q = prob_q.unsqueeze(-1).unsqueeze(-1) # B, Nq, C, 1, 1
        seg_logits = (masks * prob_q).sum(dim=1)   # B, C, h, w

        seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
        return seg_logits
