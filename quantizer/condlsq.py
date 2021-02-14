import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad
    
def gradient_approximation(x, temperature):
    y = torch.nn.functional.one_hot(torch.argmax(x, dim=1), num_classes=x.size(1)) * 1.0
    y_grad  = F.softmax(x/temperature, 1)
    return (y- y_grad).detach() + y_grad

class attention2d(nn.Module):
    def __init__(self, in_channels, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_channels!=3:
            hidden_planes = int(in_channels*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_channels, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return gradient_approximation(x, self.temperature), x


class LSQQuantizer(t.nn.Module):
    def __init__(self, bit, is_activation=False):
        super(LSQQuantizer,self).__init__()

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.bit = bit
        self.is_activation = is_activation
        self.register_buffer('init_state', torch.zeros(1))        
        
        if is_activation:
            self.Qn = 0
            self.Qp = 2 ** self.bit - 1
        else:
            self.Qn = -2 ** (self.bit - 1)
            self.Qp = 2 ** (self.bit - 1) - 1

    def forward(self, x):
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.detach().abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)
            print (self.__class__.__name__, "Initializing step-size value ...")
        
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        _alpha = grad_scale(self.alpha, g)
        x_q = round_pass((x / _alpha).clamp(self.Qn, self.Qp)) * _alpha
        return x_q

    def __repr__(self):
        return "LSQQuantizer (bit=%s, is_activation=%s)" % (self.bit, self.is_activation)



class Dynamic_LSQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=4, ratio=0.25, K=None,temperature=34, init_weight=True):


        super(Dynamic_LSQConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.quan_w = LSQQuantizer(bit=bit, is_activation=False)
        self.quan_a = LSQQuantizer(bit=bit, is_activation=True)
        self.bit = bit
        self.attention = attention2d(in_channels, ratio, out_channels, temperature)


    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            softmax_attention, raw_attention = self.attention(x)
            w_q = self.quan_w(self.weight)
            w_q = w_q * softmax_attention.view(-1, 1, 1, 1)
            return F.conv2d(self.quan_a(x), , self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
    def update_temperature(self):
        self.attention.updata_temperature()


