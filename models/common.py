from matplotlib.colors import Normalize
import torch.nn as nn
import torch
from torchvision import transforms
import math
import random
import torchvision
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2),
        16: (20, 16, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)
        return out


import torch
import torch.nn as nn



class FCM(nn.Module):
    def __init__(self, dp_feats, rgb_feats, scale):
        super(FCM, self).__init__()

        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dp_feats + rgb_feats, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.spatial_att_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid()
        )
        self.total_down = DenseProjection(dp_feats + rgb_feats, dp_feats + rgb_feats, scale, up=False, bottleneck=False)

    def forward(self, depth, rgb):

        dp_h = self.dp_up(depth)
        combined = torch.cat([dp_h, rgb], dim=1)
        fusion_map = self.fusion_conv(combined)

        att_avg = torch.mean(fusion_map, dim=1, keepdim=True)
        att_max, _ = torch.max(fusion_map, dim=1, keepdim=True)

        attention = self.spatial_att_conv(torch.cat([att_avg, att_max], dim=1))

        rgb_weighted = rgb * attention

        total = torch.cat([dp_h, rgb_weighted], dim=1)


        out = self.total_down(total)

        return out

class CLC(nn.Module):

    def __init__(self, in_ch, out_ch=None, k=3, negative_slope=0.1):                                                                                                                                                                     # 微信公众号:CV缝合救星
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),                                                                                                                                                                     # 微信公众号:CV缝合救星
        )

    def forward(self, x):
        return self.net(x)

def _choose_gn_groups(C: int) -> int:

    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1

class DNRU(nn.Module):
                                                                                                                                                                  # 微信公众号:CV缝合救星
    def __init__(self, channels, up_scale=1):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        self.gn = nn.GroupNorm(_choose_gn_groups(channels), channels)                                                                                                                                                                     # 微信公众号:CV缝合救星
        self.relu = nn.ReLU(inplace=True)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.relu(x)
        if self.up_scale and self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        return x


def vector_to_grid(x_vec):

    B, C, _, _ = x_vec.shape
    Hc = int(math.floor(math.sqrt(C)))
    Wc = int(math.ceil(C / Hc))
    pad = Hc * Wc - C
    if pad > 0:
        x_vec = F.pad(x_vec.view(B, C), (0, pad))                                                                                                                                                                     # 微信公众号:CV缝合救星
        C_ = C + pad
    else:
        x_vec = x_vec.view(B, C)
        C_ = C
    grid = x_vec.view(B, 1, Hc, Wc)
    return grid, (Hc, Wc, C, pad)


def grid_to_vector(grid, meta):

    Hc, Wc, C, pad = meta
    B = grid.size(0)
    vec = grid.view(B, Hc * Wc)
    if pad > 0:
        vec = vec[:, :C]
    return vec.view(B, C, 1, 1)



class SMM(nn.Module):
    def __init__(self, channels, negative_slope=0.1, up_scale=1):                                                                                                                                                                     # 微信公众号:CV缝合救星
        super().__init__()
        self.channels = channels


        self.clc3 = CLC(channels, channels, k=3, negative_slope=negative_slope)                                                                                                                                                                     # 微信公众号:CV缝合救星


        self.clc1_amp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )
        self.clc1_pha = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )


        self.dnru = DNRU(channels, up_scale=up_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, "channels mismatch"


        feat = self.clc3(x)


        chan_desc = F.adaptive_avg_pool2d(feat, 1)

        grid, meta = vector_to_grid(chan_desc)
        spec = torch.fft.fft2(grid)                                                                                                                                                                                   # 微信公众号:CV缝合救星
        amp = torch.abs(spec)
        pha = torch.angle(spec)


        amp = amp * self.clc1_amp(amp)
        pha = pha * self.clc1_pha(pha)


        spec_new = torch.polar(amp, pha)


        grid_ifft = torch.fft.ifft2(spec_new).real
        weight_vec = grid_to_vector(grid_ifft, meta)
        weight = torch.sigmoid(weight_vec)


        y = feat * weight
        out = y + x

        out = self.dnru(out)
        return out