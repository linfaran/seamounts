import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Conv_Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class dens_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(dens_block, self).__init__()  # 这三个相同吗？？？？
        self.conv1 = Conv_Block(ch_in, ch_out)
        self.conv2 = Conv_Block(ch_out + ch_in, ch_out)
        self.conv3 = Conv_Block(ch_out * 2 + ch_in, ch_out)

    def forward(self, input_tensor):
        x1 = self.conv1(input_tensor)
        add1 = torch.cat([x1, input_tensor], dim=1)
        x2 = self.conv2(add1)
        add2 = torch.cat([x1, input_tensor, x2], dim=1)
        x3 = self.conv3(add2)
        return x3


class Conv2D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DenseU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=2):
        super(DenseU_Net, self).__init__()
        self.n_channels = img_ch
        self.n_classes = output_ch
        self.bilinear = False
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0 = nn.Conv2d(img_ch, 32, kernel_size=7, padding=3, stride=1)
        self.Conv1 = dens_block(ch_in=32, ch_out=64)
        self.Conv2 = dens_block(ch_in=64, ch_out=64)
        self.Conv3 = dens_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        # center
        self.Conv5_1 = Conv2D(ch_in=256, ch_out=512)
        self.Conv5_2 = Conv2D(ch_in=512, ch_out=512)
        self.Drop5 = nn.Dropout(0.5)

        self.Up6 = up_conv(512, 512)
        self.add6 = torch.cat
        self.up6 = dens_block(512 + 256, 256)

        self.Up7 = up_conv(256, 256)
        self.add7 = torch.cat
        self.up7 = dens_block(256 + 128, 128)

        self.Up8 = up_conv(128, 128)
        self.add8 = torch.cat
        self.up8 = dens_block(128 + 64, 64)

        self.Up9 = up_conv(64, 64)
        self.add9 = torch.cat
        self.up9 = dens_block(64 + 64, 64)

        self.conv10_1 = nn.Conv2d(64, 32, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
        self.conv10_2 = nn.Conv2d(32, output_ch, 3, 1, 1)

    def forward(self, x):
        x = self.Conv0(x)  # 256
        down1 = self.Conv1(x)  # 256
        pool1 = self.Maxpool(down1)  # 128
        down2 = self.Conv2(pool1)  # 128
        pool2 = self.Maxpool(down2)  # 64
        down3 = self.Conv3(pool2)  # 64
        pool3 = self.Maxpool(down3)  # 32
        down4 = self.Conv4(pool3)  # 32
        pool4 = self.Maxpool(down4)  # 16
        conv5 = self.Conv5_1(pool4)  # 16
        conv5 = self.Conv5_2(conv5)  # 16
        drop5 = self.Drop5(conv5)  # 16

        up6 = self.Up6(drop5)  # 32
        # print(up6.shape)
        # print(down4.shape)
        add6 = self.add6([down4, up6], dim=1)
        up6 = self.up6(add6)

        up7 = self.Up7(up6)  # 64
        add7 = self.add7([down3, up7], dim=1)
        up7 = self.up7(add7)

        up8 = self.Up8(up7)  # 128
        add8 = self.add8([down2, up8], dim=1)
        up8 = self.up8(add8)

        up9 = self.Up9(up8)  # 256
        add9 = self.add9([down1, up9], dim=1)
        up9 = self.up9(add9)

        conv10 = self.conv10_1(up9)
        conv10 = self.relu(conv10)
        conv10 = self.conv10_2(conv10)

        return conv10
