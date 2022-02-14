import math
import time
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# quantize for weights and activations
class Quantizer(Function):
    '''
    take a real value x in alpha*[0,1] or alpha*[-1,1]
    output a discrete-valued x in alpha*{0, 1/(2^k-1), ..., (2^k-1)/(2^k-1)} or likeness
    where k is nbit
    '''
    @staticmethod
    def forward(ctx, input, nbit, alpha=None, offset=None):
        ctx.alpha = alpha
        ctx.offset = offset
        scale = (2 ** nbit - 1) if alpha is None else (2 ** nbit - 1) / alpha
        ctx.scale = scale
        return torch.round(input * scale) / scale if offset is None \
                else (torch.round(input * scale) + torch.round(offset)) / scale
#         if alpha is None:
#             scale = 2 ** nbit - 1
#             ctx.scale = scale
#             if offset is None:                
#                 return torch.round(input * scale) / scale
#             else:
#                 return (torch.round(input * scale) + offset) / scale
#         else:
#             scale = (2 ** nbit - 1) / alpha
#             if offset is None:                
#                 return torch.round(input * scale) / scale
#             else:
#                 ctx.save_for_backward(input, scale)
#                 return (torch.round(input * scale) + offset) / scale

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.offset is None:
            return grad_output, None, None, None
        else:
            return grad_output, None, None, torch.sum(grad_output) / ctx.scale


def quantize(input, nbit, alpha=None, offset=None):
    return Quantizer.apply(input, nbit, alpha, offset)


# standard sign with STE
class Signer(Function):
    '''
    take a real value x
    output sign(x)
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign(input):
    return Signer.apply(input)


# sign in xnor-net for weights
class Xnor(Function):
    '''
    take a real value x
    output sign(x_c) * E(|x_c|)
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input), dim=[1,2,3], keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def xnor(input):
    return Xnor.apply(input)


# sign in dorefa-net for weights
class ScaleSigner(Function):
    '''
    take a real value x
    output sign(x) * E(|x|)
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    
def scale_sign(input):
    return ScaleSigner.apply(input)

def dorefa_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w

def wrpn_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = quantize(torch.clamp(w, -1, 1), nbit_w - 1)
    return w

def xnor_w(w, nbit_w=1, *args, **kwargs):
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in XNOR-Net.')
    return xnor(w)
            
def bireal_w(w, nbit_w=1, *args, **kwargs):
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in Bi-Real-Net.')
    return sign(w) * torch.mean(torch.abs(w.clone().detach()))


# dorefa quantize for activations
def dorefa_a(input, nbit_a, *args, **kwargs):
    return quantize(torch.clamp(input, 0, 1.0), nbit_a, *args, **kwargs)

# PACT quantize for activations
def pact_a(input, nbit_a, alpha, *args, **kwargs):
    x = 0.5*(torch.abs(input)-torch.abs(input-alpha)+alpha)
    return quantize(x, nbit_a, alpha, *args, **kwargs)

# bi-real sign for activations
class BirealActivation(Function):
    '''
    take a real value x
    output sign(x)
    '''
    @staticmethod
    def forward(ctx, input, nbit_a=1):
        ctx.save_for_backward(input)
        return input.clamp(-1, 1).sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = (2 + 2 * input) * input.lt(0).float() + (2 - 2 * input) * input.ge(0).float()
        grad_input = torch.clamp(grad_input, 0)
        grad_input *= grad_output
        return grad_input, None

    
def bireal_a(input, nbit_a=1, *args, **kwargs):
    return BirealActivation.apply(input)


class QuantConv(nn.Conv2d):
    # general QuantConv for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels        
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_custome_parameters()
        self.quant_config()
        
    def quant_config(self, quant_name_w='dorefa', quant_name_a='dorefa', nbit_w=1, nbit_a=1, has_offset=False):
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w, 'pact': dorefa_w, 'wrpn': wrpn_w, 'xnor': xnor_w, 'bireal': bireal_w}
        name_a_dict = {'dorefa': dorefa_a, 'pact': pact_a, 'wrpn': dorefa_a, 'xnor': dorefa_a, 'bireal': bireal_a}
        self.quant_w = name_w_dict[quant_name_w]
        self.quant_a = name_a_dict[quant_name_a]
        
        if quant_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_a', None)
        if quant_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_w', None)
        if has_offset:
            self.offset = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('offset', None)
#         print(quant_name_w, quant_name_a, nbit_w, nbit_a)
        
        if self.alpha_a is not None:
            nn.init.constant_(self.alpha_a, 10)
        if self.alpha_w is not None:
            nn.init.constant_(self.alpha_w, 10)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input):
        # 0-bit: identity mapping
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.stride == 2 or self.stride == (2, 2):
                x = F.pad(input[:, :, ::2, ::2], (0, 0, 0, 0, diff_channels//2, diff_channels-diff_channels//2), 'constant', 0)
                return x
            else:
                x = F.pad(input, (0, 0, 0, 0, diff_channels//2, diff_channels-diff_channels//2), 'constant', 0)
                return x
        # w quan
        if self.nbit_w < 32:
            w = self.quant_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quant_a(input, self.nbit_a, self.alpha_a)
        else:
            x = F.relu(input)
        
        x = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
        return x
        