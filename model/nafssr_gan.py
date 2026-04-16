import torch
import torch.nn as nn
import torch.nn.functional as F

"""
NAFSSR-like Generator (NAFNet family for SR/restoration, 2022+) 
- Attention-based CNN without self-attention; uses SimpleGate + SCA (simplified channel attention)
- UNet-style encoder–decoder with NAFBlocks
- Output activation: Tanh
- No upscaling (1x); suitable for paired restoration pipelines used in this repo

References (conceptual): NAFNet/NAFSSR (Chen et al., ECCV 2022)
This is a lightweight approximation adhering to the project’s conventions.
"""


def _get_norm(norm: str, ch: int):
    if norm == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, ch)
    else:
        return nn.BatchNorm2d(ch)


def xavier_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """Simplified Channel Attention via global pooling -> 1x1 -> sigmoid."""
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        y = self.act(y)
        return x * y


class NAFBlock(nn.Module):
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.sg = SimpleGate()
        self.eca = SCA(dim)
        self.pw2 = nn.Conv2d(dim, dim, 1)
        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.dw(x)
        x = self.pw1(x)
        x = self.sg(x)
        x = self.eca(x)
        x = self.pw2(x)
        x = self.drop1(x)
        y = identity + self.beta * x

        x2 = self.norm2(y)
        x2 = self.pw1(x2)
        x2 = self.sg(x2)
        x2 = self.pw2(x2)
        x2 = self.drop2(x2)
        out = y + self.gamma * x2
        return out


class Down(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.block = NAFBlock(dim_in)
        self.down = nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.block(x)
        return self.down(x)


class Up(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1)
        self.fuse = nn.Conv2d(dim_out * 2, dim_out, 1)
        self.block = NAFBlock(dim_out)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.fuse(x)
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=6, dropout_rate=0.0, norm="batch"):
        super().__init__()
        base = 64
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, base, 3),
            _get_norm(norm, base),
            nn.ReLU(inplace=True)
        )
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)

        blocks = []
        for _ in range(max(1, n_residual_blocks)):
            blocks.append(NAFBlock(base * 8, drop=dropout_rate))
        self.bottleneck = nn.Sequential(*blocks)

        self.up3 = Up(base * 8, base * 4)
        self.up2 = Up(base * 4, base * 2)
        self.up1 = Up(base * 2, base)

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base, output_nc, 3),
            nn.Tanh()
        )

        self.apply(xavier_init)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(s3)
        u2 = self.up3(b, s2)
        u1 = self.up2(u2, s1)
        u0 = self.up1(u1, s0)
        out = self.head(u0)
        return out
