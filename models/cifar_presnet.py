'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
from quantizer.condconv import Dynamic_conv2d

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion=1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion=4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes*4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        
        raw0 = None
        if self.downsample is not None:
            residual, raw0 = self.downsample(out)

        out, raw1 = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out, raw2 = self.conv2(out)

        out += residual
        return out, [raw0, raw1, raw2]


# class PreActBottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(PreActBottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.bn1(x)
#         out = self.relu(out)

#         if self.downsample is not None:
#             #residual = self.downsample(out)
#             residual, y = self.downsample[0](out)
#             # residual = self.downsample[1](residual)

#         out,y = self.conv1(out)

#         out = self.bn2(out)
#         out = self.relu(out)
#         out,y = self.conv2(out)

#         out = self.bn3(out)
#         out = self.relu(out)
#         out,y = self.conv3(out)

#         out += residual

#         return out


# class ResNet_Cifar(nn.Module):

#     def __init__(self, block, layers, num_classes=10):
#         super(ResNet_Cifar, self).__init__()
#         self.inplanes = 16
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(block, 16, layers[0])
#         self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
#         self.avgpool = nn.AvgPool2d(8, stride=1)
#         self.fc = nn.Linear(64 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion)
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x,y = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x,y = self.layer1(x)
#         x,y = self.layer2(x)
#         x,y = self.layer3(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x,y = self.fc(x)

#         return x


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, standard_forward=False):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.layers = layers
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)
        self.standard_forward = standard_forward

        # assertion
        if num_classes == 100:
            assert sum([5,5,5]) == sum(self.layers)
        if num_classes == 10:
            assert sum([3,3,3]) == sum(self.layers)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Dynamic_conv2d):
                m.update_temperature()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.standard_forward:
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


        # ==============================================

        x = self.conv1(x)

        x, raw1 = self.layer1[0](x)
        x, raw2 = self.layer1[1](x)
        x, raw3 = self.layer1[2](x)

        if sum(self.layers) == sum([5,5,5]):
            x, raw4 = self.layer1[3](x)
            x, raw5 = self.layer1[4](x)

        x, raw6 = self.layer2[0](x)
        x, raw7 = self.layer2[1](x)
        x, raw8 = self.layer2[2](x)

        if sum(self.layers) == sum([5,5,5]):
            x, raw9 = self.layer2[3](x)
            x, raw10 = self.layer2[4](x)

        x, raw11 = self.layer3[0](x)
        x, raw12 = self.layer3[1](x)
        x, raw13 = self.layer3[2](x)

        if sum(self.layers) == sum([5,5,5]):
            x, raw14 = self.layer3[3](x)
            x, raw15 = self.layer3[4](x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        
        if sum(self.layers) == sum([5,5,5]):
            return x, [*raw1, *raw2, *raw3, *raw4, *raw5, *raw6, *raw7, *raw8, *raw9, *raw10, *raw11, *raw12, *raw13, *raw14, *raw15]

        if sum(self.layers) == sum([3,3,3]):
            return x, [*raw1, *raw2, *raw3, *raw6, *raw7, *raw8, *raw11, *raw12, *raw13]

# def resnet20_cifar(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
#     return model


# def resnet32_cifar(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
#     return model


# def resnet44_cifar(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
#     return model


# def resnet56_cifar(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
#     return model


# def resnet110_cifar(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
#     return model


# def resnet1202_cifar(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
#     return model


# def resnet164_cifar(**kwargs):
#     model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
#     return model


# def resnet1001_cifar(**kwargs):
#     model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
#     return model

def preact_resnet20_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [3, 3, 3], **kwargs)
    return model

def preact_resnet32_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [5, 5, 5], **kwargs)
    return model

# def preact_resnet110_cifar(**kwargs):
#     model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
#     return model


# def preact_resnet164_cifar(**kwargs):
#     model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
#     return model


# def preact_resnet1001_cifar(**kwargs):
#     model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
#     return model


if __name__ == '__main__':
    net = resnet20_cifar()
    y = net(torch.randn(1, 3, 64, 64))
    print(net)
    print(y.size())
