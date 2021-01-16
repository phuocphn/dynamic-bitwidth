import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return gradient_approximation(x, self.temperature), x



class LSQQuantizer(torch.nn.Module):
    def __init__(self, bit, K, is_activation=False):
        super(LSQQuantizer,self).__init__()

        self.alpha = nn.Parameter(torch.randn(K, 1))
        self.bit = bit
        self.K = K
        self.is_activation = is_activation
        self.register_buffer('init_state', torch.zeros(1))
                
        if is_activation:
            self.Qn = 0
            self.Qp = 2 ** self.bit - 1
            self.Qp_2bit = 2** 2 -1
            self.Qp_3bit = 2** 3 -1
            self.Qp_4bit = 2** 4 -1
            # self.Qp_5bit = 2** 5 -1 
        else:
            self.Qn = -2 ** (self.bit - 1)
            self.Qn_2bit = -2 ** (2 - 1)
            self.Qn_3bit = -2 ** (3 - 1)
            self.Qn_4bit = -2 ** (4 - 1)
            # self.Qn_5bit = -2 ** (5 - 1)


            self.Qp = 2 ** (self.bit - 1) - 1
            self.Qp_2bit = 2 ** (2-1) -1
            self.Qp_3bit = 2 ** (3-1) -1
            self.Qp_4bit = 2 ** (4-1) -1
            # self.Qp_5bit = 2 ** (5-1) -1 


    def forward(self, x):
        if self.training and self.init_state == 0:
            self.alpha[0, 0].data.copy_((2* x.detach().abs().mean() / math.sqrt(self.Qp_2bit)))
            self.alpha[1, 0].data.copy_((2* x.detach().abs().mean() / math.sqrt(self.Qp_3bit)))
            self.alpha[2, 0].data.copy_((2* x.detach().abs().mean() / math.sqrt(self.Qp_4bit)))
            # self.alpha[3, 0].data.copy_((2* x.detach().abs().mean() / math.sqrt(self.Qp_5bit)))
            self.init_state.fill_(1)
            print (self.__class__.__name__, "Initializing step-size value ...")
        
        _alpha = self.alpha
        _alpha[0, 0].data.copy_(grad_scale(_alpha[0, 0], 1.0 / math.sqrt(x.size(1) * self.Qp_2bit)))
        _alpha[1, 0].data.copy_(grad_scale(_alpha[1, 0], 1.0 / math.sqrt(x.size(1) * self.Qp_3bit)))
        _alpha[2, 0].data.copy_(grad_scale(_alpha[2, 0], 1.0 / math.sqrt(x.size(1) * self.Qp_4bit)))
        # _alpha[3, 0].data.copy_(grad_scale(_alpha[3, 0], 1.0 / math.sqrt(x.size(1) * self.Qp_5bit)))


        # g = 1.0 / math.sqrt(x.numel() * Qp)
        # _alpha = grad_scale(self.alpha, g)
        _div = x / _alpha
        _div[0, 0] =  _div[0,0].clone().clamp(self.Qn_2bit, self.Qp_2bit)
        _div[1, 0] =  _div[1,0].clone().clamp(self.Qn_3bit, self.Qp_3bit)
        _div[2, 0] =  _div[2,0].clone().clamp(self.Qn_4bit, self.Qp_4bit)
        # _div[3, 0] =  _div[3,0].clone().clamp(self.Qn_5bit, self.Qp_5bit)

        x_q = round_pass(_div) * _alpha
        return x_q

    def __repr__(self):
        return "LSQQuantizer (bit=%s, is_activation=%s)" % (self.bit, self.is_activation)

class Dynamic_LSQConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_LSQConv2d, self).__init__()
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
        self.lsq_fn = LSQQuantizer(bit=4, K=K, is_activation=False)
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
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)
        weight = self.lsq_fn(weight)


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
