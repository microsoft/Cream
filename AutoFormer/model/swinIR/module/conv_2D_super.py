import torch.nn as nn
from typing import Union, Tuple
import torch.nn.functional as F

class Conv2DSuper(nn.Module):
    def __init__(self,
                in_channels: int, out_channels: int,
                kernel_size: int,
                stride: Union[int, Tuple],
                padding: Union[int, Tuple, str],
                *args,
                **kwargs
                ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            *args,
            **kwargs
        )
        # This check is assuming we want to preserve output size when we sample later
        # And that we don't want to crop images to be able to do so.
        # One other way to implement this if we want to preserve output size but allow cropping is:
        # Create a mask to multiply with the weight before passing on to F.conv2d.
        # So instead of sampling a smaller weight, we keep its size (kernel_size-wise) but set outer edges to 0.
        if padding < kernel_size // 2:
            raise ValueError(f"Padding {padding} too small for kernel_size: {kernel_size}. Won't allow decreasing further.")

        self.super_in_channels = in_channels
        self.super_out_channels = out_channels
        self.super_kernel_size = kernel_size
        self.super_stride = stride
        self.super_padding = padding

    def set_sample_config(self, in_channels=None, out_channels=None, kernel_size=None):
        # conv.weight is of shape (output_channels, input_channels/group, kernel_size[0], kernel_size[1])
        self.sample_out_channels = self.conv.weight.shape[0] if out_channels is None else out_channels
        self.sample_in_channels = self.conv.weight.shape[1] if in_channels is None else in_channels
        self.sample_kernel_size = self.conv.weight.shape[2] if kernel_size is None else kernel_size

        # We want to pick out the middle part
        # So we figure out k_i, the point from where to begin the slicing
        k_i = (self.super_kernel_size - self.sample_kernel_size) // 2

        self.conv_sample_weight = self.conv.weight[:self.sample_out_channels,
                                                   :self.sample_in_channels,
                                                   k_i: k_i + self.sample_kernel_size,
                                                   k_i: k_i + self.sample_kernel_size]
        self.conv_sample_bias = self.conv.bias[:self.sample_out_channels, ...]

        self.sample_padding = self.super_padding - k_i

    def forward(self, x):
        return F.conv2d(x,
                        self.conv_sample_weight, self.conv_sample_bias,
                        stride=self.super_stride,
                        padding=self.sample_padding,
                        )

    def calc_sampled_param_num(self):
        return self.conv_sample_weight.numel + self.conv_sample_bias.numel

