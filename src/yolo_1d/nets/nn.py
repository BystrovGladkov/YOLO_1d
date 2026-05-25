import math

import torch
import torch.nn as nn

from yolo_1d.utils.util import make_anchors

SIGNAL_LENGTH = 49152

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(nn.Module): 
    """
        Базовый слой свёртки с BatchNorm и Функцией активации
    """
    
    def __init__(self, channels_in, channels_out, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        # Одномерная свёртка. 
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=False, groups=g)
        # Слой нормализации по батчу.
        self.bn = nn.BatchNorm1d(channels_out, eps=0.001, momentum=0.03)
        # Гладкое прилижение ReLU
        self.act = activation# nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x):
        return self.act(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch * 2, out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool1d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())

    def forward(self, x):
        b, c, l = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, l)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, l) + self.conv1(v.reshape(b, c, l))
        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.SiLU()),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


class DarkNet(torch.nn.Module):
    def __init__(self, input_channels, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []

        # p1/2
        # 50000
        self.p1.append(Conv(input_channels, width[0], torch.nn.SiLU(), 3, 2, 1))      # 24576, 3
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), 3, 2, 1))      # 12288, 7
        self.p1.append(Conv(width[1], width[2], torch.nn.SiLU(), 3, 2, 1))     # 6144, 19
        self.p1.append(CSP( width[2], width[2], depth[0], csp[0], r=2))                               # 6144, ? 40
        self.p1.append(Conv(width[2], width[3], torch.nn.SiLU(), 3, 2, 1))    # 3072  ? 80
        self.p1.append(Conv(width[3], width[4], torch.nn.SiLU(), 3, 2, 1))    # 1536  ? 200
        self.p1.append(CSP( width[4], width[4], depth[1], csp[0], r=2))                               # 1536  ? 300 
        
        # p2/4
        self.p2.append(Conv(width[4], width[5], torch.nn.SiLU(), 3, 2, 1))    # 768, 700
        self.p2.append(Conv(width[5], width[6], torch.nn.SiLU(), 3, 2, 1))    # 384, 1500
        self.p2.append(CSP( width[6], width[6], depth[2], csp[0], r=2))                               # 384, 3000
        # p3/8
        self.p3.append(Conv(width[6], width[7], torch.nn.SiLU(), 3, 2, 1))    # 195, 8000
        self.p3.append(CSP( width[7], width[7], depth[3], csp[0], r=2))                               # 195, 12000
        self.p3.append(SPP(width[7], width[7]))
        self.p3.append(PSA(width[7], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        return p1, p2, p3


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up1 = torch.nn.Upsample(scale_factor=2)
        self.up2 = torch.nn.Upsample(scale_factor=4)
        self.h1 = CSP(width[6] + width[7], width[7], depth[4], csp[1], r=2)
        self.h2 = CSP(width[4] + width[7], width[7], depth[5], csp[1], r=2)
        self.h3 = nn.Sequential(Conv(width[7], width[7], torch.nn.SiLU(), 3, 2, 1), 
                                Conv(width[7], width[7], torch.nn.SiLU(), 3, 2, 1))
        self.h4 = CSP(width[7] * 2, width[7], depth[6], csp[1], r=2)
        self.h5 = Conv(width[7], width[7], torch.nn.SiLU(), 3, 2, 1)
        self.h6 = CSP(width[7] * 2, width[7], depth[7], csp[2], r=2)

    def forward(self, x):
        p3, p4, p5 = x
        
        p4 = self.h1(torch.cat(tensors=[self.up1(p5), p4], dim=1))
        
        p3 = self.h2(torch.cat(tensors=[self.up2(p4), p3], dim=1))
        
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        # ch * stride_of_head  = максимальный размер предсказываемого объекта
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 2, self.ch, a).transpose(2, 1) # Теперь предсказывается только начало и конец
        return self.conv(x.softmax(1)).view(b, 2, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 2  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.validate = False

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
                                                           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
                                                           torch.nn.Conv1d(box, out_channels=2 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                           Conv(x, cls, torch.nn.SiLU()),
                                                           Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                           Conv(cls, cls, torch.nn.SiLU()),
                                                           torch.nn.Conv1d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        y = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = y.split(split_size=(2 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        
        y = torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)
        
        return (x, y) if self.validate else y

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (SIGNAL_LENGTH / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width=None, depth=None, csp=None, num_classes=3, input_channels=15):
        super().__init__()
        self.net = DarkNet(input_channels, width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)
        
        # calculate biases
        img_dummy = torch.zeros(1, input_channels, SIGNAL_LENGTH)
        self.head = Head(num_classes, (width[7], width[7], width[7]))
        self.head.stride = torch.tensor([SIGNAL_LENGTH / x.shape[-1] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        if x.shape[-1] != SIGNAL_LENGTH:
            x = torch.nn.functional.interpolate(x, SIGNAL_LENGTH, mode='linear')
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def set_validate(self, value: bool):
        """Sets model state to get raw (for loss computation) and decoded boxes (for metrics evaluation)."""
        self.head.validate = value
    
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_classes: int = 80):
    csp = [False, False, True]
    depth = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    width = [32, 32, 32, 64, 128, 128, 128, 128]
    return YOLO(width, depth, csp, num_classes, 30)


def yolo_v11_s(num_classes: int = 80):
    csp = [True, True, True]
    depth = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    width = [32, 32, 64, 96, 128, 128, 192, 192]
    return YOLO(width, depth, csp, num_classes, 30)


def yolo_v11_m(num_classes: int = 80):
    csp = [True, True, True]
    depth = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    width = [32, 32, 64, 96, 128, 192, 192, 256]
    return YOLO(width, depth, csp, num_classes, 30)


def yolo_v11_l(num_classes: int = 80):
    csp = [True, True, True]
    depth = [1, 1, 1, 1, 1, 2, 2, 2, 3]
    width = [32, 64, 64, 96, 128, 192, 256, 512]
    return YOLO(width, depth, csp, num_classes, 30)

def yolo_v11_xl(num_classes: int = 80):
    csp = [True, True, True]
    depth = [1, 3, 3, 3, 3, 3, 4, 4, 5]
    width = [32, 64, 64, 96, 128, 192, 256, 512]
    return YOLO(width, depth, csp, num_classes, 30)

def get_model(size:str, num_classes:int):    
    if size == "nano":
        return yolo_v11_n(num_classes)
    if size == "small":
        return yolo_v11_s(num_classes)
    if size == "medium":
        return yolo_v11_m(num_classes)
    if size == "large":
        return yolo_v11_l(num_classes)
    if size == "xlarge":
        return yolo_v11_xl(num_classes)