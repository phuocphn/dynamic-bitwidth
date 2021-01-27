import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from munch import munchify


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



FLAGS  = {
    'bits_list': [8,6,5,4], 
    'switchbn': True, 
    'weight_only':False, 
    'switch_alpha': False,
    'rescale_conv': False,
    'clamp': True,
    'rescale': True

}
FLAGS = munchify(FLAGS)


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
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return gradient_approximation(x, self.temperature), x


class q_k(Function):
    """
        This is the quantization module.
        The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    @staticmethod
    def forward(ctx, input, bit, scheme='original'):
        assert bit > 0
        assert torch.all(input >= 0) and torch.all(input <= 1)
        if scheme == 'original':
            a = (1 << bit) - 1
            res = torch.round(a * input)
            res.div_(a)
        elif scheme == 'modified':
            a = 1 << bit
            res = torch.floor(a * input)
            res.clamp_(max=a - 1).div_(a)
        else:
            raise NotImplementedError
        assert torch.all(res >= 0) and torch.all(res <= 1)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class ADoReFaQuantizer(torch.nn.Module):
    def __init__(self, bit=None): #, K=4, is_activation=False):
        super(ADoReFaQuantizer,self).__init__()

        # self.alpha = nn.Parameter(torch.randn(K, 1))
        if getattr(FLAGS, 'switch_alpha', False):
            self.alpha = nn.Parameter(torch.ones(len(FLAGS.bits_list)) * 8.0)
        else:
            self.alpha = nn.Parameter(torch.tensor(8.0))

        self.bit = bit

        self.double_side = False
        self.quant = q_k.apply
        self.act_quant_scheme = getattr(FLAGS, 'act_quant_scheme', 'original')

        # self.K = K
        # self.is_activation = is_activation
        # self.register_buffer('init_state', torch.zeros(1))
        # if is_activation:
        #     Qns = [0 for bit in np.array(range(K)) + 2 ]
        #     self.register_buffer('Qns', torch.tensor(Qns, requires_grad=False).view(-1, 1))

        #     Qps = [2** bit -1 for bit in np.array(range(K)) + 2]
        #     self.register_buffer('Qps', torch.tensor(Qps, requires_grad=False).view(-1, 1))
        # else:
        #     print ("Un-expected behaviour")

    def forward(self, x, attention):
        if self.bit < 32:
            if getattr(FLAGS, 'switch_alpha', False):
                print ("Un-expected behavior ..")
                exit()
                # if bita in FLAGS.bits_list:
                #     idx_alpha = FLAGS.bits_list.index(bita)
                # else:
                #     idx_alpha = 0
                # alpha = torch.abs(self.alpha[idx_alpha])
            else:
                alpha = torch.abs(self.alpha)
            if self.double_side:
                # input_val = torch.where(input > -alpha, input, -alpha)
                pass
            else:
                # input_val = torch.relu(input)
                input_val = x




            input_val = torch.where(input_val < alpha, input_val, alpha)
            # if bita < 32 and not self.weight_only:
            if self.bit < 32:
                input_val.div_(alpha)
                if self.double_side:
                    print ("Un-expected behavior ..")
                    exit()
                    # input_val.add_(1.0)
                    # input_val.div_(2.0)
                input_val = self.quant(input_val, self.bit, self.act_quant_scheme)
                if self.double_side:
                    pass
                    # input_val.mul_(2.0)
                    # input_val.sub_(1.0)
                input_val.mul_(alpha)
        else:
            input_val = input
        return input



        # if self.training and self.init_state == 0:
        #     self.alpha.data.copy_( (2* x.detach().abs().mean() / (self.Qps * 1.0)**0.5))
        #     self.init_state.fill_(1)
        #     print (self.__class__.__name__, "Initializing step-size value ...")
        
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
        return self.__class__.__name__ + " (bit=%s, is_activation=%s)" % (self.bit, self.is_activation)



class WDoReFaQuantizer(torch.nn.Module):
    def __init__(self, bit=None, K=4, is_activation=False):
        super(WDoReFaQuantizer,self).__init__()
        self.weight_quant_scheme = getattr(FLAGS, 'weight_quant_scheme', 'modified')
        self.act_quant_scheme = getattr(FLAGS, 'act_quant_scheme', 'original')
        self.bit = bit
        self.quant = q_k.apply
        self.K=K

    def forward(self, x):
        outputs = []
        
        for bit in np.arange(self.K) + 2:
            xq = torch.tanh(x) / torch.max(torch.abs(torch.tanh(x)))
            xq.add_(1.0)
            xq.div_(2.0)
            xq = self.quant(xq, bit, self.weight_quant_scheme)
            xq.mul_(2.0)
            xq.sub_(1.0)
            outputs.append(xq)

        return torch.stack(outputs, dim=0)



        # if self.bit < 32:
        #     x = torch.tanh(x) / torch.max(torch.abs(torch.tanh(x)))
        #     x.add_(1.0)
        #     x.div_(2.0)
        #     x = self.quant(x, self.bit, self.weight_quant_scheme)
        #     x.mul_(2.0)
        #     x.sub_(1.0)

        #     if getattr(FLAGS, 'rescale_conv', False):
        #         print ("Un-expected path....", "if getattr(FLAGS, 'rescale_conv', False):")
        #         exit()
        # else:
        #     print ("Un-expected path....", "32bit ???")
        #     exit()

        #     # no quantizing but only clamping
        #     # if getattr(FLAGS, 'clamp', True):
        #     #     x = torch.tanh(x) / torch.max(torch.abs(torch.tanh(x)))
        #     # else:
        #     #     x = x * 1.0
        #     # if getattr(FLAGS, 'rescale_conv', False):
        #     #     print ("Un-expected path....", "if getattr(FLAGS, 'rescale_conv', False):")
        #     #     exit()

        # return x

    def __repr__(self):
        return self.__class__.__name__ + "weight_quant_scheme={}, act_quant_scheme={}, bit={}".format(self.weight_quant_scheme, self.act_quant_scheme, self.bit)
        # return self.__class__.__name__ + " (bit=%s, is_activation=%s)" % (self.bit, self.is_activation)




class DynamicDRFConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(DynamicDRFConv2d, self).__init__()
        assert in_channels%groups==0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0]
        self.stride = stride[0]
        self.padding = padding[0]
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_channels, ratio, K, temperature)
        self.quan_w = WDoReFaQuantizer(bit=None, K=K, is_activation=False)
        self.quan_a = ADoReFaQuantizer(bit=4) #, K=K, is_activation=True)

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
        # x = self.quan_a(x, softmax_attention)
        x = x.view(1, -1, height, width)
        
        weight = self.weight.view(self.K, -1)
        weight = self.quan_w(weight)


        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
            
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        return [output, raw_attention]
