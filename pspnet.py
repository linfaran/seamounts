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

class SimpleBackbone(nn.Module):
    """ 轻量 encoder，输出 stride=8 高层特征 """
    def __init__(self, in_ch=3, feat=128):
        super().__init__()
        self.layer = nn.Sequential(
            ConvBNReLU(in_ch, 64, 3, 2),  # /2
            ConvBNReLU(64, 64, 3, 1),
            nn.MaxPool2d(2),              # /4
            ConvBNReLU(64, 128, 3, 2),    # /8
            ConvBNReLU(128, feat, 3, 1),
        )
        self.out_channels = feat
    def forward(self, x):
        return self.layer(x)

class PPM(nn.Module):
    """ Pyramid Pooling Module """
    def __init__(self, in_c, out_c, bins=(1,2,3,6)):
        super().__init__()
        stages = []
        for b in bins:
            if b == 1:
                # 关键改动：1x1 池化这一路不再用 BN，避免 batch=1 时报错
                stages.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_c, out_c, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True)
                ))
            else:
                stages.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(b),
                    ConvBNReLU(in_c, out_c, k=1, s=1, p=0)
                ))
        self.stages = nn.ModuleList(stages)
        self.project = ConvBNReLU(in_c + out_c*len(bins), out_c, k=3, s=1, p='same')

    def forward(self, x):
        H, W = x.shape[2:]
        priors = [x]
        for s in self.stages:
            y = s(x)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            priors.append(y)
        x = torch.cat(priors, dim=1)
        return self.project(x)

class PSPNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=19, feat=128, ppm_out=128, bins=(1,2,3,6)):
        super().__init__()
        self.backbone = SimpleBackbone(n_channels, feat=feat)  # /8
        self.ppm = PPM(self.backbone.out_channels, ppm_out, bins=bins)
        self.fuse = nn.Sequential(
            ConvBNReLU(ppm_out, 64, 3, 1, 'same'),
            nn.Dropout2d(0.1)
        )
        self.cls = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        H, W = x.shape[2:]
        f = self.backbone(x)          # /8
        f = self.ppm(f)               # /8
        f = self.fuse(f)
        out = self.cls(f)             # /8
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
