import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 小工具层 -----------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p='same', d=1):
        super().__init__()
        if p == 'same':
            p = ((k-1)//2)*d
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    # 轻量ResNet Bottleneck (C->C/4->C/4->4C)
    expansion = 4
    def __init__(self, in_c, mid_c, s=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, mid_c, k=1, s=1, p=0)
        self.conv2 = ConvBNReLU(mid_c, mid_c, k=3, s=s, p='same', d=dilation)
        self.conv3 = nn.Conv2d(mid_c, mid_c * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_c * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNetMini(nn.Module):
    """
    非官方简化 ResNet-50 风格：
    输出两个特征：
      - low: stage1 输出 (用于 Deeplabv3+ 的低层解码器)
      - high: 最后一个stage输出 (ASPP输入)
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 2, p=3),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)   # C1 输出 256
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2) # C2 输出 512
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2) # C3 输出 1024
        # 使用空洞代替下采样：输出 stride 16
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=1, dilation=2) # C4 输出 2048
        self.out_channels_low  = 256
        self.out_channels_high = 2048
    def _make_layer(self, in_c, mid_c, blocks, stride=1, dilation=1):
        downsample = None
        out_c = mid_c * Bottleneck.expansion
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        layers = [Bottleneck(in_c, mid_c, s=stride, downsample=downsample, dilation=dilation)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_c, mid_c, s=1, downsample=None, dilation=dilation))
        return nn.Sequential(*layers)
    def forward(self, x):
        x  = self.stem(x)        # /4
        c1 = self.layer1(x)      # /4
        c2 = self.layer2(c1)     # /8
        c3 = self.layer3(c2)     # /16
        c4 = self.layer4(c3)     # /16 (空洞)
        low  = c1                # /4, 256ch
        high = c4                # /16, 2048ch
        return low, high

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rates=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1
        self.branches.append(ConvBNReLU(in_c, out_c, k=1, s=1, p=0))
        # dilated convs
        for r in rates[1:]:
            self.branches.append(ConvBNReLU(in_c, out_c, k=3, s=1, p='same', d=r))
        # image pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.project = ConvBNReLU(out_c*(len(rates)+1), out_c, k=1, s=1, p=0)
    def forward(self, x):
        size = x.shape[2:]
        feats = [b(x) for b in self.branches]
        imgp = self.image_pool(x)
        imgp = F.interpolate(imgp, size=size, mode='bilinear', align_corners=False)
        feats.append(imgp)
        x = torch.cat(feats, dim=1)
        return self.project(x)

class DeepLabV3Plus(nn.Module):
    """
    轻量 DeepLabv3+：
    - Backbone: ResNetMini（输出stride=16高层特征 + stride=4低层特征）
    - ASPP on high
    - 低层1x1降通道 + 上采样融合
    """
    def __init__(self, n_channels=3, n_classes=19, aspp_out=256, low_ch=48):
        super().__init__()
        self.backbone = ResNetMini(n_channels)
        self.aspp = ASPP(self.backbone.out_channels_high, aspp_out)
        self.low_proj = ConvBNReLU(self.backbone.out_channels_low, low_ch, k=1, s=1, p=0)
        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_out+low_ch, 256, 3, 1, 'same'),
            ConvBNReLU(256, 256, 3, 1, 'same')
        )
        self.classifier = nn.Conv2d(256, n_classes, 1)
    def forward(self, x):
        H, W = x.shape[2:]
        low, high = self.backbone(x)             # low:/4, high:/16
        high = self.aspp(high)                   # /16
        high_up = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        low_p = self.low_proj(low)
        dec = self.decoder(torch.cat([high_up, low_p], dim=1)) # /4
        out = self.classifier(dec)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
