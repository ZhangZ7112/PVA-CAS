# -*- coding: utf-8 -*-

#导入库
from torch import nn, cat
import torch.nn.functional as F
import torch

class ProtoConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ProtoConv,self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        return torch.exp(- (torch.sum(torch.mul(x-self.weight,x-self.weight),dim=1, keepdim=True)) )

class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpooling = nn.MaxPool3d(2)

    def forward(self, x):
        x0 = self.maxpooling(x)
        x1 = self.relu1(self.gn1(self.conv1(x0)))
        y = self.relu2(self.gn2(self.conv2(x1)))
        return  y


class EnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(EnBlock, self).__init__()
        self.maxpooling = nn.MaxPool3d(2)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.maxpooling(x)
        x1 = self.relu1(self.gn1(self.conv1(x)))
        y = self.relu2(self.gn2(self.conv2(x1)))
        y = self.dropout(y)
        return y


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)


    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):        #x+relu2(bn2(conv2(relu1(bn1(conv1(x))))))
        x_mid = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(x_mid)))
        return y+x

class UNet(nn.Module):
    def __init__(self, in_channels, base_channels, num_classes=1):
        super(UNet, self).__init__()

        self.InitBlock = InitBlock(in_channels=in_channels, out_channels=base_channels)
        self.EnBlock1 = EnBlock(in_channels=base_channels, out_channels=base_channels * 2 , dropout_rate=0)
        self.EnBlock2 = EnBlock(in_channels=base_channels * 2, out_channels=base_channels * 4, dropout_rate=0)
        self.EnBlock3 = EnBlock(in_channels=base_channels * 4, out_channels=base_channels * 8, dropout_rate=0.3)
        self.EnBlock4 = EnBlock(in_channels=base_channels * 8, out_channels=base_channels * 16, dropout_rate=0.3)

        self.DeUp1 = DeUp_Cat(in_channels=base_channels * 16, out_channels=base_channels * 8)
        self.DeBlock1 = DeBlock(in_channels=base_channels * 8, out_channels=base_channels * 8)

        self.DeUp2 = DeUp_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock2 = DeBlock(in_channels=base_channels * 4, out_channels=base_channels * 4)

        self.DeUp3 = DeUp_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock3 = DeBlock(in_channels=base_channels * 2, out_channels=base_channels * 2)

        self.DeUp4 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels * 1)
        self.DeBlock4 = DeBlock(in_channels=base_channels * 1, out_channels=base_channels * 1)

        self.upsample =  nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dropout = nn.Dropout(0.3)
        self.endconv = nn.Conv3d(in_channels=base_channels * 1, out_channels=num_classes, kernel_size=1)
        if num_classes == 1:
            self.decision_function = nn.Sigmoid()
        else:
            self.decision_function = nn.Softmax(dim=1)

    def encode(self, x):
        x_0 = self.InitBlock(x)  # out_channels=16
        x_1 = self.EnBlock1(x_0)  # out_channels=32
        x_2 = self.EnBlock2(x_1)  # out_channels=64
        x_3 = self.EnBlock3(x_2)  # out_channels=128
        x_4 = self.EnBlock4(x_3)  # out_channels=256

        return x_0, x_1, x_2, x_3, x_4

    def decode(self, x_0, x_1, x_2, x_3, x_4):

        y1 = self.DeUp1(x_4, x_3)
        y1 = self.DeBlock1(y1)

        y2 = self.DeUp2(y1, x_2)
        y2 = self.DeBlock2(y2)

        y3 = self.DeUp3(y2, x_1)
        y3 = self.DeBlock3(y3)

        y4 = self.DeUp4(y3, x_0)
        y4 = self.DeBlock4(y4)

        y4 = self.dropout(y4)
        y = self.endconv(y4)
        y = self.decision_function(y)

        y = self.upsample(y)

        return y

    def forward(self, x):
        x_0, x_1, x_2, x_3, x_4 = self.encode(x)
        y = self.decode(x_0, x_1, x_2, x_3, x_4)
        return y


class Proto_UNet(nn.Module):
    def __init__(self, in_channels, base_channels, num_classes=1):
        super(Proto_UNet, self).__init__()

        self.InitBlock = InitBlock(in_channels=in_channels, out_channels=base_channels)
        self.EnBlock1 = EnBlock(in_channels=base_channels, out_channels=base_channels * 2 , dropout_rate=0)
        self.EnBlock2 = EnBlock(in_channels=base_channels * 2, out_channels=base_channels * 4, dropout_rate=0)
        self.EnBlock3 = EnBlock(in_channels=base_channels * 4, out_channels=base_channels * 8, dropout_rate=0.3)
        self.EnBlock4 = EnBlock(in_channels=base_channels * 8, out_channels=base_channels * 16, dropout_rate=0.3)

        self.DeUp1 = DeUp_Cat(in_channels=base_channels * 16, out_channels=base_channels * 8)
        self.DeBlock1 = DeBlock(in_channels=base_channels * 8, out_channels=base_channels * 8)

        self.DeUp2 = DeUp_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock2 = DeBlock(in_channels=base_channels * 4, out_channels=base_channels * 4)

        self.DeUp3 = DeUp_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock3 = DeBlock(in_channels=base_channels * 2, out_channels=base_channels * 2)

        self.DeUp4 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels * 1)
        self.DeBlock4 = DeBlock(in_channels=base_channels * 1, out_channels=base_channels * 1)

        self.upsample =  nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dropout = nn.Dropout(0.3)
        self.endconv = nn.Conv3d(in_channels=base_channels * 1, out_channels=num_classes, kernel_size=1)
        self.pconv = ProtoConv(in_channels=base_channels * 1, out_channels=num_classes)
        if num_classes == 1:
            self.decision_function = nn.Sigmoid()
        else:
            self.decision_function = nn.Softmax(dim=1)

    def encode(self, x):
        x_0 = self.InitBlock(x)  # out_channels=16
        x_1 = self.EnBlock1(x_0)  # out_channels=32
        x_2 = self.EnBlock2(x_1)  # out_channels=64
        x_3 = self.EnBlock3(x_2)  # out_channels=128
        x_4 = self.EnBlock4(x_3)  # out_channels=256

        return x_0, x_1, x_2, x_3, x_4

    def decode(self, x_0, x_1, x_2, x_3, x_4, start_pconv=False):

        y1 = self.DeUp1(x_4, x_3)
        y1 = self.DeBlock1(y1)

        y2 = self.DeUp2(y1, x_2)
        y2 = self.DeBlock2(y2)

        y3 = self.DeUp3(y2, x_1)
        y3 = self.DeBlock3(y3)

        y4 = self.DeUp4(y3, x_0)
        y4 = self.DeBlock4(y4)

        y4 = self.dropout(y4)

        if start_pconv==False:
            y = self.endconv(y4)
            y = self.decision_function(y)
            y = self.upsample(y)
        else:  #start_pconv==True --> activate ProtoConv, and freeze the other components
            y = self.pconv(y4)
            y = self.upsample(y)

        return y

    def forward(self, x, start_pconv=False):
        x_0, x_1, x_2, x_3, x_4 = self.encode(x)
        y = self.decode(x_0, x_1, x_2, x_3, x_4, start_pconv)
        return y


class UNet_try(nn.Module):
    def __init__(self, in_channels, base_channels, num_classes=1):
        super(UNet_try, self).__init__()

        self.InitBlock = InitBlock(in_channels=in_channels, out_channels=base_channels)
        self.EnBlock1 = EnBlock(in_channels=base_channels, out_channels=base_channels * 2 , dropout_rate=0)
        self.EnBlock2 = EnBlock(in_channels=base_channels * 2, out_channels=base_channels * 4, dropout_rate=0)
        self.EnBlock3 = EnBlock(in_channels=base_channels * 4, out_channels=base_channels * 8, dropout_rate=0.3)
        self.EnBlock4 = EnBlock(in_channels=base_channels * 8, out_channels=base_channels * 16, dropout_rate=0.3)

        self.DeUp1 = DeUp_Cat(in_channels=base_channels * 16, out_channels=base_channels * 8)
        self.DeBlock1 = DeBlock(in_channels=base_channels * 8, out_channels=base_channels * 8)

        self.DeUp2 = DeUp_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock2 = DeBlock(in_channels=base_channels * 4, out_channels=base_channels * 4)

        self.DeUp3 = DeUp_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock3 = DeBlock(in_channels=base_channels * 2, out_channels=base_channels * 2)

        self.DeUp4 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels * 1)
        self.DeBlock4 = DeBlock(in_channels=base_channels * 1, out_channels=base_channels * 1)

        self.upsample =  nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dropout = nn.Dropout(0.3)
        self.endconv = nn.Conv3d(in_channels=base_channels * 1, out_channels=num_classes, kernel_size=1)
        if num_classes == 1:
            self.decision_function = nn.Sigmoid()
        else:
            self.decision_function = nn.Softmax(dim=1)

    def encode(self, x):
        x_0 = self.InitBlock(x)  # out_channels=16
        x_1 = self.EnBlock1(x_0)  # out_channels=32
        x_2 = self.EnBlock2(x_1)  # out_channels=64
        x_3 = self.EnBlock3(x_2)  # out_channels=128
        x_4 = self.EnBlock4(x_3)  # out_channels=256

        return x_0, x_1, x_2, x_3, x_4

    def decode(self, x_0, x_1, x_2, x_3, x_4):

        y1 = self.DeUp1(x_4, x_3)
        y1 = self.DeBlock1(y1)

        y2 = self.DeUp2(y1, x_2)
        y2 = self.DeBlock2(y2)

        y3 = self.DeUp3(y2, x_1)
        y3 = self.DeBlock3(y3)

        y4 = self.DeUp4(y3, x_0)
        y4 = self.DeBlock4(y4)

        y4 = self.dropout(y4)

        y = self.decision_function(y4)

        map = y.data.cpu().numpy()

        return map

    def forward(self, x):
        x_0, x_1, x_2, x_3, x_4 = self.encode(x)
        y = self.decode(x_0, x_1, x_2, x_3, x_4)
        return y



if __name__ == '__main__':
    # with torch.no_grad():
    #     import os
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #     cuda0 = torch.device('cuda:0')
    #     x = torch.rand((1, 1, 16, 32, 32), device=cuda0)
    #     model = UNet_light(1,16,1)
    #     model.cuda()
    #     x = model(x)
    #     print(x.shape)
    p = ProtoConv(16,1)
    print(p.weight.shape)

