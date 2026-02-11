""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .scconv import *


class ScConv_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ScConv_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 原始UNet组件
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 添加ScConv模块
        self.sc_conv_bottleneck = ScConv(1024 // factor)  # 瓶颈层
        self.sc_conv_x4 = ScConv(512)  # 跳跃连接x4
        self.sc_conv_x3 = ScConv(256)  # 跳跃连接x3
        self.sc_conv_x2 = ScConv(128)  # 跳跃连接x2
        self.sc_conv_x1 = ScConv(64)  # 跳跃连接x1

        # 解码器部分保持不变
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 在瓶颈层应用ScConv
        x5 = self.sc_conv_bottleneck(x5)

        # 解码器（在跳跃连接处应用ScConv）
        x = self.up1(x5, self.sc_conv_x4(x4))  # x4经过ScConv
        x = self.up2(x, self.sc_conv_x3(x3))  # x3经过ScConv
        x = self.up3(x, self.sc_conv_x2(x2))  # x2经过ScConv
        x = self.up4(x, self.sc_conv_x1(x1))  # x1经过ScConv
        logits = self.outc(x)
        return logits

    # 检查点方法保持不变
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# # ----------------------------
# # 定义使用 ScConv 的卷积块
# # ----------------------------
# class ScConvBlock(nn.Module):
#     """
#     用1个普通的3x3卷积调整通道数，然后用ScConv模块进一步处理，最后加上ReLU激活。
#     """
#
#     def __init__(self, in_channels, out_channels, padding=1):
#         super(ScConvBlock, self).__init__()
#         self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False)
#         self.sc_conv = ScConv(out_channels)  # ScConv 要求输入通道数等于op_channel
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv_in(x)
#         x = self.sc_conv(x)
#         x = self.relu(x)
#         return x
#
#
# class DoubleScConv(nn.Module):
#     """
#     替换传统的 DoubleConv 模块，由两个连续的 ScConvBlock 组成。
#     """
#
#     def __init__(self, in_channels, out_channels):
#         super(DoubleScConv, self).__init__()
#         self.block1 = ScConvBlock(in_channels, out_channels)
#         self.block2 = ScConvBlock(out_channels, out_channels)
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         return x
#
#
# # ----------------------------
# # 定义下采样模块
# # ----------------------------
# class Down(nn.Module):
#     """
#     下采样模块：先通过 2x2 最大池化，再用 DoubleScConv 进行特征提取。
#     """
#
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.pool = nn.MaxPool2d(2)
#         self.conv = DoubleScConv(in_channels, out_channels)
#
#     def forward(self, x):
#         x = self.pool(x)
#         x = self.conv(x)
#         return x
#
#
# # ----------------------------
# # 定义上采样模块
# # ----------------------------
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             # 修改此处，接收in_channels，输出in_channels//2
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         # 拼接后的通道数为 (in_channels//2 + in_channels//2) = in_channels
#         self.conv = DoubleScConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # 利用padding调整尺寸（如果需要）
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x
#
#
# # ----------------------------
# # 定义输出卷积
# # ----------------------------
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         # 输出层一般使用1x1卷积，这里可以保留原始卷积
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
# # ----------------------------
# # 完整的 ScConv_UNet（全面替换卷积模块版本）
# # ----------------------------
# class ScConv_UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(ScConv_UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         # 编码器部分
#         self.inc = DoubleScConv(n_channels, 64)  # 初始卷积
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#
#         # 瓶颈层：这里我们可以用一个 ScConv 模块对下采样结果进行进一步加工
#         self.sc_conv_bottleneck = ScConv(1024 // factor)
#
#         # 解码器部分（上采样模块）
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         # 编码器
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#
#         # 在瓶颈层应用 ScConv
#         x5 = self.sc_conv_bottleneck(x5)
#
#         # 解码器（跳跃连接直接使用编码器的特征，不需要额外ScConv，因为编码器中的已用ScConv构造）
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#
#         logits = self.outc(x)
#         return logits
#
#     # 如有需要，可以继续添加 checkpointing 方法等辅助函数



