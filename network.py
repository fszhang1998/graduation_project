import torch
import torch.nn as nn
from layers import *


def conv5x5(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ELU, inplace=True))

def conv1x1(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 1, stride, 0, activation_fn=partial(nn.ELU, inplace=True))

def deconv5x5(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,
                  activation_fn=partial(nn.ELU, inplace=True))


def resblock(in_channels):
    """Resblock without BN and the last activation
    """
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=3, stride=1, use_batchnorm=False,
                      activation_fn=partial(nn.ELU, inplace=True), last_activation_fn=None)


class EBlock(nn.Module):
    def __init__(self, out_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.resblock_stack(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.deconv = deconv5x5(in_channels, out_channels, stride, output_padding)

        self.attention = Attention(out_channels, out_channels, 1)
    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        y=self.attention(x)
        return x*y


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, 3, 3, 1, 1, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x

class Attention(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(type(self), self).__init__()
        self.conv=conv1x1(in_channels,out_channels,stride)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class SRNDeblurNet(nn.Module):
    """SRN-DeblurNet
    examples:
        net = SRNDeblurNet()
        y = net( x1 , x2 , x3）#x3 is the coarsest image while x1 is the finest image
    """

    def __init__(self,upsample_fn=partial(torch.nn.functional.upsample, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.upsample_fn = upsample_fn
        self.conv1_1 = conv5x5(3+3, 32, 1)
        self.inblock = EBlock(32)
        self.conv1_2 = conv5x5(32, 64, 2)
        self.eblock1 = EBlock(64)
        self.conv1_3 = conv5x5(64, 128, 2)
        self.eblock2 = EBlock(128)

        self.dblock1 = DBlock(128, 64, 2, 1)
        self.dblock2 = DBlock(64, 32, 2, 1)
        self.outblock = OutBlock(32)

        self.conv2_1 = conv5x5(3 + 3, 32, 1)
        self.conv2_2 = conv5x5(32, 64, 2)
        self.conv2_3 = conv5x5(64, 128, 2)

        self.conv3_1 = conv5x5(3 + 3, 32, 1)
        self.conv3_2 = conv5x5(32, 64, 2)
        self.conv3_3 = conv5x5(64, 128, 2)
 
        self.attention1_1 = Attention(32, 32, 1)
        self.attention1_2 = Attention(64, 64, 1)
        self.attention1_3 = Attention(128, 128, 1)

        self.attention2_1 = Attention(32, 32, 1)
        self.attention2_2 = Attention(64, 64, 1)
        self.attention2_3 = Attention(128, 128, 1)

        self.attention3_1 = Attention(32, 32, 1)
        self.attention3_2 = Attention(64, 64, 1)
        self.attention3_3 = Attention(128, 128, 1)
        self.input_padding = None
        if xavier_init_all:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    torch.nn.init.xavier_normal_(m.weight)
                    # torch.nn.init.kaiming_normal_(m.weight)
                    # print(name)

    def forward_step1(self, x):
        input=self.conv1_1(x)
        e32 = self.inblock(input*self.attention1_1(input))
        conv_e32=self.conv1_2(e32)
        e64 = self.eblock1(conv_e32*self.attention1_2(conv_e32))
        conv_e64=self.conv1_3(e64)
        e128 = self.eblock2(conv_e64*self.attention1_3(conv_e64))

        d64 = self.dblock1(e128)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        return d3

    def forward_step2(self, x):
        input = self.conv2_1(x)
        e32 = self.inblock(input*self.attention2_1(input))
        conv_e32=self.conv2_2(e32)
        e64 = self.eblock1(conv_e32*self.attention2_2(conv_e32))
        conv_e64=self.conv2_3(e64)
        e128 = self.eblock2(conv_e64*self.attention2_3(conv_e64))

        d64 = self.dblock1(e128)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        return d3

    def forward_step3(self, x):
        input = self.conv3_1(x)
        e32 = self.inblock(input * self.attention3_1(input))
        conv_e32=self.conv3_2(e32)
        e64 = self.eblock1(conv_e32*self.attention3_2(conv_e32))
        conv_e64=self.conv3_3(e64)
        e128 = self.eblock2(conv_e64*self.attention3_3(conv_e64))
        d64 = self.dblock1(e128)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        return d3
    def forward(self, b1, b2, b3):
        if self.input_padding is None or self.input_padding.shape != b3.shape:
            self.input_padding = b3

        i3 = self.forward_step1(torch.cat([b3.detach(), self.input_padding], 1))

        i2 = self.forward_step2(torch.cat([b2.detach(), self.upsample_fn(i3, scale_factor=2)], 1))

        i1 = self.forward_step3(torch.cat([b1.detach(), self.upsample_fn(i2, scale_factor=2)], 1))


        return i1, i2, i3



