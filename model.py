#coding:UTF-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size,
                                    stride=stride,padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)
    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        if out[0].equal(torch.max(out[0], out[1])):
            output = torch.cat((out[0]/2,torch.max(out[1], out[2])/2), dim=0)
        else:
            output = torch.cat((out[1]/2,torch.max(out[0]/2, out[2])/2), dim=0)
        output1 = torch.chunk(output, 2, 0)
        return torch.max(output1[0], output1[1])
class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out
class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x
class reslightenedcnn(nn.Module):
    def __init__(self, block, layers, num_classes=157995):
        super(reslightenedcnn, self).__init__()
        self.conv1 = mfm(1, 96, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#pool1
        self.block1 = self._make_layer(block, layers[0], 96, 96)
        self.group1 = group(96, 192, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)#pool2
        self.block2 = self._make_layer(block, layers[1], 192, 192)
        self.group2 = group(192, 384, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#pool3
        self.block3 = self._make_layer(block, layers[2], 384, 384)
        self.group3 = group(384, 256, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 256, 256)
        self.group4 = group(256, 256, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)#pool4
        self.fc1 = mfm(9216, 512, type=0)
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)
        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)
        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        fc1 = self.fc1(x)
        return fc1
def FaceCNN(**kwargs):
    model = reslightenedcnn(resblock, [1, 2, 3, 4], **kwargs)
    return model