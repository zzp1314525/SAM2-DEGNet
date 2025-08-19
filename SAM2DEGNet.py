import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GEA, CPM, EFPM, DEOM, CEAD
from sam2.build_sam import build_sam2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

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
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
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
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class SAM2DEGNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2DEGNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.gea = GEA()
        self.UNM = CPM(64)
        self.efpm1 = EFPM(144, 64)
        self.efpm2 = EFPM(288, 64)
        self.efpm3 = EFPM(576, 64)
        self.efpm4 = EFPM(1152, 64)

        self.igm1 = DEOM(64)
        self.igm2 = DEOM(64)
        self.igm3 = DEOM(64)
        self.igm4 = DEOM(64)

        self.fiac = CEAD(64)
        self.fia1 = CEAD(64)
        self.fia2 = CEAD(64)
        self.fia3 = CEAD(64)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(64, 1, 1)
        self.predictor3 = nn.Conv2d(64, 1, 1)
        self.predictor4 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        edge = self.gea(x4, x1)
        edge_att = torch.sigmoid(edge)

        x1, x2, x3, x4 = self.efpm1(x1), self.efpm2(x2), self.efpm3(x3), self.efpm4(x4)
        coarse_att, coarse_feature = self.UNM(x4, x3, x2, x1)

        x4a = self.igm4(x4, edge_att, coarse_att)
        xcou = F.interpolate(coarse_feature, size=x4a.size()[
                                                  2:], mode='bilinear', align_corners=False)
        xc4, e4 = self.fiac(x4a, xcou)
        o4 = self.predictor4(xc4)

        x3a = self.igm3(x3, edge_att, o4)
        x4au = F.interpolate(xc4, size=x3a.size()[
                                       2:], mode='bilinear', align_corners=False)
        x34, e3 = self.fia1(x3a, x4au)
        o3 = self.predictor3(x34)

        x2a = self.igm2(x2, edge_att, o3)
        x34u = F.interpolate(x34, size=x2a.size()[
                                       2:], mode='bilinear', align_corners=False)
        x234, e2 = self.fia2(x2a, x34u)
        o2 = self.predictor2(x234)

        x1a = self.igm1(x1, edge_att, o2)
        x234u = F.interpolate(x234, size=x1a.size()[
                                         2:], mode='bilinear', align_corners=False)
        x1234, e1 = self.fia3(x1a, x234u)
        o1 = self.predictor1(x1234)

        o3 = F.interpolate(o3, scale_factor=16,
                           mode='bilinear', align_corners=False)
        o2 = F.interpolate(o2, scale_factor=8,
                           mode='bilinear', align_corners=False)
        o1 = F.interpolate(o1, scale_factor=4,
                           mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4,
                           mode='bilinear', align_corners=False)

        oc = F.interpolate(coarse_att, scale_factor=4,
                           mode='bilinear', align_corners=False)
        o4 = F.interpolate(o4, scale_factor=32,
                           mode='bilinear', align_corners=False)
        e4 = F.interpolate(e4, size=x.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        e3 = F.interpolate(e3, size=x.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        e2 = F.interpolate(e2, size=x.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        e1 = F.interpolate(e1, size=x.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384


        return o3, o2, o1, oe, oc, o4, e4, e3, e2, e1


if __name__ == "__main__":
        model = SAM2DEGNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        o3, o2, o1, oe, oc, o4, e4, e3, e2, e1 = model(x)
        print(o3.shape, o2.shape, o1.shape,o3.shape, oe.shape, oc.shape, o4.shape, e4.shape, e3.shape, e2.shape, e1.shape)