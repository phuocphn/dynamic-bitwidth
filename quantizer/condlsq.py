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


class ActivationAtentionQuantizer(torch.nn.Module):
    def __init__(self, bit=2, is_activation=True):
        super(ActivationAtentionQuantizer,self).__init__()

        self.alpha = nn.Parameter(torch.randn(1))
        self.bit = bit #list(np.array(range(K)) + 2)
        # self.K = K
        self.is_activation = is_activation
        self.register_buffer('init_state', torch.zeros(1))
                
        # # if is_activation:
        # # Qns = [0 for bit in [bit] * 4 ]
        # self.register_buffer('Qns', torch.tensor(Qns, requires_grad=False).view(-1, 1))

        # Qps = [2** bit -1 for bit in np.array(range(K)) + 2]
        # self.register_buffer('Qps', torch.tensor(Qps, requires_grad=False).view(-1, 1))
        # # else:
        # #     print ("Un-expected behaviour")


        if self.is_activation:
            self.Qn = 0
            self.Qp = 2 ** self.bit - 1
        else:
            self.Qn = -2 ** (self.bit - 1)
            self.Qp = 2 ** (self.bit - 1) - 1



    def forward(self, x, attention):
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.detach().abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)
            print (self.__class__.__name__, "Initializing step-size value ...")
        
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        _alpha = grad_scale(self.alpha, g)
        x_q = round_pass((x / _alpha).clamp(self.Qn, self.Qp)) * _alpha
        return x_q
        
        # runtime_Qns = torch.mm(attention, self.Qns*1.0)
        # runtime_Qps = torch.mm(attention, self.Qps*1.0)
        # runtime_alphas = torch.mm(attention, self.alpha)

        # g = 1.0 / ( (x.numel() / x.size(0)) * runtime_Qps) ** 0.5
        # _alpha = grad_scale(runtime_alphas, g)
        # clipped = torch.max(torch.min(x/_alpha.view(-1, 1, 1,1), runtime_Qps.view(-1, 1,1,1)), runtime_Qns.view(-1,1,1,1))
        # clipped = x/ _alpha.view(-1, 1,1,1)
        # x_q = round_pass(clipped) * _alpha.view(-1,1,1,1)
        # return x_q

    def __repr__(self):
        return self.__class__.__name__ + " (bit=%s, is_activation=%s)" % (self.bit, True)


class WeightAtentionQuantizer(torch.nn.Module):
    def __init__(self, bit=None, K=4):
        super(WeightAtentionQuantizer,self).__init__()

        self.alpha = nn.Parameter(torch.randn(K, 1))
        self.bit = [bit] * K
        self.K = K
        self.register_buffer('init_state', torch.zeros(1))
                

        Qns = [-2 ** (_bit- 1) for _bit in [bit] * K ]
        self.register_buffer('Qns', torch.tensor(Qns, requires_grad=False).view(-1, 1))

        Qps = [2 ** (_bit-1) -1 for _bit in [bit] * K]
        self.register_buffer('Qps', torch.tensor(Qps, requires_grad=False).view(-1, 1))

    def forward(self, x):
        if self.training and self.init_state == 0:
            self.alpha.data.copy_( (2* x.detach().abs().mean() / (self.Qps)**0.5))
            self.init_state.fill_(1)
            print (self.__class__.__name__, "Initializing step-size value ...")
        
        g = 1.0 / (x.numel() * self.Qps) ** 0.5
        _alpha = grad_scale(self.alpha, g)
        clipped = torch.max(torch.min(x/_alpha, self.Qps), self.Qns)
        x_q = round_pass(clipped) * _alpha
        return x_q

    def __repr__(self):
        return self.__class__.__name__ + " (bit=%s, is_activation=%s)" % (self.bit, False)




class Dynamic_LSQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True, bit=2):
        super(Dynamic_LSQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1)
        assert in_channels%groups==0
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size[0]
        # self.stride = stride[0]
        # self.padding = padding[0]
        self.dilation = dilation
        self.groups = groups
        #self.bias = bias
        self.K = K
        self.bit = bit 
        self.attention = attention2d(in_channels, ratio, K, temperature)
        self.quan_w = WeightAtentionQuantizer(bit=bit, K=K)
        self.quan_a = ActivationAtentionQuantizer(bit=bit)

        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size[0], kernel_size[0]), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_channels))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO åˆå§‹åŒ–
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#
        softmax_attention, raw_attention = self.attention(x)
        batch_size, in_channels, height, width = x.size()
        if self.bit !=32:
            x = self.quan_a(x, softmax_attention)
        x = x.view(1, -1, height, width)
        
        weight = self.weight.view(self.K, -1)

        if self.bit !=32:
            weight = self.quan_w(weight)


        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        return [output, raw_attention]



