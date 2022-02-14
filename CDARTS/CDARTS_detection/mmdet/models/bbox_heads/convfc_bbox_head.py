import torch.nn as nn

from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule

from .auto_head.build_head import build_search_head


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.relu = nn.ReLU()
        if in_channel != out_channel:
            self.downsample = nn.Conv2d(in_channel, out_channel, 1, bias=False)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                nn.SyncBatchNorm(in_channel),
                nn.Conv2d(in_channel, out_channel, 1, bias=False),
                nn.SyncBatchNorm(out_channel)
            )
        else:
            self.downsample = nn.Sequential()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // 4, 1, bias=False),
                nn.SyncBatchNorm(in_channel // 4),
                nn.Conv2d(in_channel // 4, in_channel // 4, 3, padding=1, bias=False),
                nn.SyncBatchNorm(in_channel // 4),
                nn.Conv2d(in_channel // 4, out_channel, 1, bias=False),
                nn.SyncBatchNorm(out_channel)
            )

        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x):
        out = self.conv(x)
        short_cut = self.downsample(x)
        out = self.relu(out + short_cut)
        return out


# For toy experiments
class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, kernel_size, dilation=1, groups=1):
        super(MBBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels =out_channels 
        self.stride = stride
        self.groups = groups
        mid_channels = in_channels * expansion
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0, dilation=1, bias=False, groups=groups),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, groups=mid_channels),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0, dilation=1, bias=False, groups=groups),
            nn.SyncBatchNorm(out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.in_channels == self.out_channels and self.stride == 1:
            out = out + x
        return out


@HEADS.register_module
class ConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 convs_kernel=3,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 search_head=None,
                 toy_replace=None,
                 bottle_first='conv',     
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs >= 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.convs_kernel = convs_kernel
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.bottle_first = bottle_first
        self.SearchHead = build_search_head(search_head)
        self.toy_replace = toy_replace # for toy experiments replace

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True, toy_replace)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch_2head(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch_2head(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        if self.SearchHead is not None and self.num_shared_fcs == 0:
            self.cls_last_dim = self.SearchHead.last_dim
            self.reg_last_dim = self.SearchHead.last_dim

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False,
                            toy_replace=None):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                if toy_replace is not None and i == toy_replace.get('stage', 30):
                    if toy_replace.get('block', 'res') == 'ir':
                        branch_convs.append(
                            MBBlock(conv_in_channels, self.conv_out_channels, 1, 1, 
                                toy_replace.get('conv_kernel'), dilation=toy_replace.get('dilation'), groups=1))
                    else:
                        branch_convs.append(
                            ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            toy_replace.get('conv_kernel'),
                            padding=(toy_replace.get('conv_kernel')-1) * toy_replace.get('dilation') // 2,
                            dilation=toy_replace.get('dilation'),
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            bottle_first=self.bottle_first))
                else:
                    branch_convs.append(
                        ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            self.convs_kernel,
                            padding=(self.convs_kernel-1) // 2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            bottle_first=self.bottle_first))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def _add_conv_fc_branch_2head(self,
                                 num_branch_convs,
                                 num_branch_fcs,
                                 in_channels):
        """convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ResidualBlock(conv_in_channels, self.conv_out_channels)
                ) 
            last_layer_dim = self.conv_out_channels

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim * self.roi_feat_size * self.roi_feat_size if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        loss_latency = None
        if self.SearchHead is not None:
            x, loss_latency = self.SearchHead(x)
        
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, loss_latency


@HEADS.register_module
class SharedFCBBoxHead(ConvFCBBoxHead):
    def __init__(self, num_convs=0, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        super(SharedFCBBoxHead, self).__init__(
            num_shared_convs=num_convs,
            num_shared_fcs=num_fcs,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)