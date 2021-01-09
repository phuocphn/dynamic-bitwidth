import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# class attention1d(nn.Module):
#     def __init__(self, in_channels, ratios, K, temperature, init_weight=True):
#         super(attention1d, self).__init__()
#         assert temperature%3==1
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         if in_channels!=3:
#             hidden_planes = int(in_channels*ratios)+1
#         else:
#             hidden_planes = K
#         self.fc1 = nn.Conv1d(in_channels, hidden_planes, 1, bias=False)
#         # self.bn = nn.BatchNorm2d(hidden_planes)
#         self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
#         self.temperature = temperature
#         if init_weight:
#             self._initialize_weights()


#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m ,nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def updata_temperature(self):
#         if self.temperature!=1:
#             self.temperature -=3
#             print('Change temperature to:', str(self.temperature))


#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x).view(x.size(0), -1)
#         return F.softmax(x/self.temperature, 1)


# class Dynamic_conv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
#         super(Dynamic_conv1d, self).__init__()
#         assert in_channels%groups==0
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.K = K
#         self.attention = attention1d(in_channels, ratio, K, temperature)

#         self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size), requires_grad=True)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(K, out_channels))
#         else:
#             self.bias = None
#         if init_weight:
#             self._initialize_weights()

#         #TODO åˆå§‹åŒ–
#     def _initialize_weights(self):
#         for i in range(self.K):
#             nn.init.kaiming_uniform_(self.weight[i])


#     def update_temperature(self):
#         self.attention.updata_temperature()

#     def forward(self, x):#å°†batchè§†ä½œç»´åº¦å˜é‡ï¼Œè¿›è¡Œç»„å·ç§¯ï¼Œå› ä¸ºç»„å·ç§¯çš„æƒé‡æ˜¯ä¸åŒçš„ï¼ŒåŠ¨æ€å·ç§¯çš„æƒé‡ä¹Ÿæ˜¯ä¸åŒçš„
#         softmax_attention = self.attention(x)
#         batch_size, in_channels, height = x.size()
#         x = x.view(1, -1, height, )# å˜åŒ–æˆä¸€ä¸ªç»´åº¦è¿›è¡Œç»„å·ç§¯
#         weight = self.weight.view(self.K, -1)

#         # åŠ¨æ€å·ç§¯çš„æƒé‡çš„ç”Ÿæˆï¼Œ ç”Ÿæˆçš„æ˜¯batch_sizeä¸ªå·ç§¯å‚æ•°ï¼ˆæ¯ä¸ªå‚æ•°ä¸åŒï¼‰
#         aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_channels, self.kernel_size,)
#         if self.bias is not None:
#             aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
#             output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups*batch_size)
#         else:
#             output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups * batch_size)

#         output = output.view(batch_size, self.out_channels, output.size(-1))
#         return output

# class WTA(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return torch.nn.functional.one_hot(torch.argmax(input, dim=1), num_classes=input.size(1)) * 1.0

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         input, = ctx.saved_tensors
#         softmax_out = torch.nn.functional.softmax(input, dim=1) 
#         softmax_deriv = softmax_out * (1- softmax_out)
#         grad_input = grad_output.clone()
#         grad_input = grad_input * softmax_deriv
#         return grad_input

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
        # self.wta = WTA.apply
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
        # return F.softmax(x/self.temperature, 1)
        # return torch.nn.functional.one_hot(torch.argmax(x, dim=1))
        # return self.wta(x)
        return gradient_approximation(x, self.temperature)



def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad




class LSQQuantizer(torch.nn.Module):
    def __init__(self, bit, K, is_activation=False):
        super(LSQQuantizer,self).__init__()

        self.alpha = nn.Parameter(torch.Tensor(K, 1))
        self.bit = bit
        self.K = K
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
            self.alpha.data.copy_( (2 * x.detach().abs().mean(dim=1) / math.sqrt(self.Qp)).view(self.K, 1) )             
            self.init_state.fill_(1)
            print (self.__class__.__name__, "Initializing step-size value ...")
        
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        _alpha = grad_scale(self.alpha, g)
        x_q = round_pass((x / _alpha).clamp(self.Qn, self.Qp)) * _alpha
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

    def forward(self, x):#å°†batchè§†ä½œç»´åº¦å˜é‡ï¼Œè¿›è¡Œç»„å·ç§¯ï¼Œå› ä¸ºç»„å·ç§¯çš„æƒé‡æ˜¯ä¸åŒçš„ï¼ŒåŠ¨æ€å·ç§¯çš„æƒé‡ä¹Ÿæ˜¯ä¸åŒçš„
        softmax_attention = self.attention(x)
        batch_size, in_channels, height, width = x.size()
        x = x.view(1, -1, height, width)# å˜åŒ–æˆä¸€ä¸ªç»´åº¦è¿›è¡Œç»„å·ç§¯
        weight = self.weight.view(self.K, -1)
        weight = self.lsq_fn(weight)
        # åŠ¨æ€å·ç§¯çš„æƒé‡çš„ç”Ÿæˆï¼Œ ç”Ÿæˆçš„æ˜¯batch_sizeä¸ªå·ç§¯å‚æ•°ï¼ˆæ¯ä¸ªå‚æ•°ä¸åŒï¼‰


        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        return output


# class attention3d(nn.Module):
#     def __init__(self, in_channels, ratios, K, temperature):
#         super(attention3d, self).__init__()
#         assert temperature%3==1
#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         if in_channels != 3:
#             hidden_planes = int(in_channels * ratios)+1
#         else:
#             hidden_planes = K
#         self.fc1 = nn.Conv3d(in_channels, hidden_planes, 1, bias=False)
#         self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
#         self.temperature = temperature

#     def updata_temperature(self):
#         if self.temperature!=1:
#             self.temperature -=3
#             print('Change temperature to:', str(self.temperature))

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x).view(x.size(0), -1)
#         return F.softmax(x / self.temperature, 1)

# class Dynamic_conv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):
#         super(Dynamic_conv3d, self).__init__()
#         assert in_channels%groups==0
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.K = K
#         self.attention = attention3d(in_channels, ratio, K, temperature)

#         self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(K, out_channels))
#         else:
#             self.bias = None


#         #TODO åˆå§‹åŒ–
#         # nn.init.kaiming_uniform_(self.weight, )

#     def update_temperature(self):
#         self.attention.updata_temperature()

#     def forward(self, x):#å°†batchè§†ä½œç»´åº¦å˜é‡ï¼Œè¿›è¡Œç»„å·ç§¯ï¼Œå› ä¸ºç»„å·ç§¯çš„æƒé‡æ˜¯ä¸åŒçš„ï¼ŒåŠ¨æ€å·ç§¯çš„æƒé‡ä¹Ÿæ˜¯ä¸åŒçš„
#         softmax_attention = self.attention(x)
#         batch_size, in_channels, depth, height, width = x.size()
#         x = x.view(1, -1, depth, height, width)# å˜åŒ–æˆä¸€ä¸ªç»´åº¦è¿›è¡Œç»„å·ç§¯
#         weight = self.weight.view(self.K, -1)

#         # åŠ¨æ€å·ç§¯çš„æƒé‡çš„ç”Ÿæˆï¼Œ ç”Ÿæˆçš„æ˜¯batch_sizeä¸ªå·ç§¯å‚æ•°ï¼ˆæ¯ä¸ªå‚æ•°ä¸åŒï¼‰
#         aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
#         if self.bias is not None:
#             aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
#             output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups*batch_size)
#         else:
#             output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups * batch_size)

#         output = output.view(batch_size, self.out_channels, output.size(-3), output.size(-2), output.size(-1))
#         return output