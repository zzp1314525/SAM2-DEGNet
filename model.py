import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log



class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride,
                      padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EFPM(nn.Module):
    def __init__(self, in_channel, out_channel, need_relu=True):
        super(EFPM, self).__init__()
        self.need_relu = need_relu
        self.relu = nn.ReLU(True)

        # 多尺度分支
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        # 计算权重 k 的跨尺度自注意力
        self.attn = nn.Sequential(
            nn.Conv2d(4 * out_channel, 4, kernel_size=1),  # 生成 (B, 4, H, W) 的 k
            nn.Softmax(dim=1)  # 在通道维度归一化
        )

        # 逐通道融合
        self.alpha = nn.Parameter(torch.ones(4))  # 可学习参数
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # 拼接所有分支
        x_cat = torch.cat((x0, x1, x2, x3), dim=1)

        # 计算自注意力权重 k
        k = self.attn(x_cat)  # (B, 4, H, W)

        # 按照权重融合每个分支
        x_fused = (
            self.alpha[0] * k[:, 0:1] * x0 +
            self.alpha[1] * k[:, 1:2] * x1 +
            self.alpha[2] * k[:, 2:3] * x2 +
            self.alpha[3] * k[:, 3:4] * x3
        )

        # 逐点加上残差连接，并进行 ReLU
        if self.need_relu:
            x_out = self.relu(x_fused + self.conv_res(x))
        else:
            x_out = x_fused + self.conv_res(x)
        return x_out


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class GEA(nn.Module):
    def __init__(self):
        super(GEA, self).__init__()
        self.reduce1 = Conv1x1(144, 256)
        self.reduce4 = Conv1x1(1152, 256)
        self.attention = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.block = nn.Sequential(
            ConvBNR(1024, 256, 3),
            ConvBNR(256, 128, 3),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x4, x1):
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)

        gradient_x1 = self.compute_image_gradient(x1)
        gradient_x4 = self.compute_image_gradient(x4)

        x1_with_gradient = torch.cat((x1, gradient_x1), dim=1)
        x4_with_gradient = torch.cat((x4, gradient_x4), dim=1)
        x4_with_gradient = F.interpolate(
            x4_with_gradient, x1_with_gradient.size()[2:], mode='bilinear', align_corners=False)

        attention_weights = self.attention(torch.cat((x4_with_gradient, x1_with_gradient), dim=1))
        x1_weighted = x1_with_gradient * attention_weights[:, :, :x1.size(2), :x1.size(3)]
        x4_weighted = x4_with_gradient * attention_weights[:, :, :x1.size(2), :x1.size(3)]
        out = torch.cat((x4_weighted, x1_weighted), dim=1)
        out = self.block(out)

        return out

    def compute_image_gradient(self, x):
        x = torch.autograd.Variable(x, requires_grad=True)
        gradient_x = torch.autograd.grad(outputs=x.sum(), inputs=x, create_graph=True)[0]
        gradient_x = torch.abs(gradient_x)
        return gradient_x

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(512, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff

class DEOM(nn.Module):
    def __init__(self, channel):
        super(DEOM, self).__init__()
        self.fom = EFPM(1, 1)
        self.sigmoid_coarse = nn.Sigmoid()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k,
                                padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, c, edge_att, coarse_att):
        if c.size() != edge_att.size():
            edge_att = F.interpolate(
                edge_att, c.size()[2:], mode='bilinear', align_corners=False)
        if c.size() != coarse_att.size():
            coarse_att = F.interpolate(
                coarse_att, c.size()[2:], mode='bilinear', align_corners=False)
        coarse_att1 = self.fom(coarse_att)
        coarse_att2 = self.sigmoid_coarse(coarse_att1)
        edge_pred = make_laplace(coarse_att, 1)
        edge_pred_feature = c * edge_pred
        coarse_feature = c * coarse_att2
        edge_feature = c * edge_att
        x = torch.cat([edge_feature, coarse_feature, edge_pred_feature], dim=1)
        x = self.fusion_conv(x)
        x = x + coarse_feature + edge_feature
        attention_map = self.attention(x)
        x = x * attention_map
        out = x + c
        out = self.conv2d(out)
        wei = self.avg_pool(out)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)
                          ).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        out = out * wei

        return out + c


class CrossAttention(nn.Module):
    """跨层交叉注意力特征融合"""
    def __init__(self, in_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
        x: 浅层特征 (B, C, H, W)
        y: 深层特征 (B, C, H', W')
        """
        B, C, H, W = x.shape

        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        y_flat = y.view(B, C, -1).permute(0, 2, 1)  # (B, M, C)

        Q = self.query(x_flat)  # (B, N, C)
        K = self.key(y_flat)  # (B, M, C)
        V = self.value(y_flat)  # (B, M, C)

        attn = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (C ** 0.5))  # (B, N, M)
        z = torch.bmm(attn, V)  # (B, N, C)
        z = z.permute(0, 2, 1).view(B, C, H, W)  # 变回 (B, C, H, W)

        return x + z  # 残差连接，增强特征


class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

class GatedConv2dWithActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

class TFD(nn.Module):
    def __init__(self, in_channels):
        super(TFD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.gatedconv = GatedConv2dWithActivation(in_channels * 2, in_channels, kernel_size=3, stride=1,
                                                   padding=1, dilation=1, groups=1, bias=True, batch_norm=True,
                                                   activation=torch.nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feature_map, perior_repeat):
        assert (feature_map.shape == perior_repeat.shape), "feature_map and prior_repeat have different shape"
        uj = perior_repeat
        uj_conv = self.conv(uj)
        uj_1 = uj_conv + uj
        uj_i_feature = torch.cat([uj_1, feature_map], 1)
        uj_2 = uj_1 + self.gatedconv(uj_i_feature) - 3 * uj_conv
        return uj_2


class ODE(nn.Module):
    def __init__(self, in_channels):
        super(ODE, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, feature_map):
        # 计算RK4的四个增量
        k1 = self.F1(feature_map)
        k2 = self.F2(feature_map + k1 / 2)
        k3 = self.F3(feature_map + k2 / 2)
        k4 = self.F4(feature_map + k3)

        # 组合四个增量
        out = feature_map + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return out


class CEAD(nn.Module):
    def __init__(self, channel):
        super(CEAD, self).__init__()
        self.conv1 = EFPM(3 * channel, channel, False)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)

        self.attn_x2y = CrossAttention(channel)  # x 关注 y
        self.attn_y2x = CrossAttention(channel)  # y 关注 x
        self.alpha = nn.Conv2d(channel * 2, 1, kernel_size=1) #计算融合权重

        self.conv3 = nn.Conv2d(channel * 2, channel, kernel_size=1)
        self.ode = ODE(channel)
        self.out_B = nn.Sequential(
            BasicDeConv2d(channel, channel // 2, kernel_size=3, stride=2, padding=1, out_padding=1),
            BasicConv2d(channel // 2, channel // 4, kernel_size=3, padding=1),
            nn.Conv2d(channel // 4, 1, kernel_size=3, padding=1)
        )
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1)

    def forward(self, x, y):
        x_to_y = self.attn_x2y(x, y)
        y_to_x = self.attn_y2x(y, x)
        alpha = torch.sigmoid(self.alpha(torch.cat([x_to_y, y_to_x], dim=1)))  # 计算融合权重
        F_fusion = alpha * x_to_y + (1 - alpha) *  y_to_x # 加权融合

        yt = self.conv3(torch.cat([x, y], dim=1))
        ode_out = self.ode(yt)
        bound = self.out_B(ode_out)
        bound = self.edge_enhance(bound)

        cat2 = torch.cat([F_fusion, ode_out], dim=1)  # 2,128,48,48
        F_fusion = self.conv4(cat2)

        b = self.conv1(torch.cat((x, y, F_fusion), 1))
        ret = self.conv2(F_fusion + b)
        return ret, bound

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class CPM(nn.Module):
    def __init__(self, channel):
        super(CPM, self).__init__()
        self.fia1_ = CEAD(channel)
        self.fia2_ = CEAD(channel)
        self.fia3_ = CEAD(channel)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(channel, channel, 1)
        self.conv2 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x2a, _ = self.fia1_(x2, self.upsample(x1))
        x3a, _ = self.fia2_(x3, self.upsample(x2a))
        x4a, _ = self.fia3_(x4, self.upsample(x3a))
        x = self.conv1(x4a)
        ret = self.conv2(x)
        return ret, x


