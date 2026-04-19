import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Restormer-UNet-like Generator (2022–2024 transformer family for image restoration)
- Uses MDTA (multi-DConv head transposed attention) and GDFN blocks
- UNet-style encoder–decoder with skip connections
- Output activation: Tanh
- No upscaling (1x). Suitable for paired img2img tasks in this repo.

Note: This is a compact implementation inspired by Restormer (Zamir et al., CVPR 2022), adapted to
minimize external dependencies while following project conventions.
"""


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


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention (simplified).
    Operates in channel space with depthwise conv on QKV, as per Restormer idea.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to heads
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=3)
        k = F.normalize(k, dim=3)
        attn = (q.transpose(-2, -1) @ k) * self.temperature  # (b, heads, HW, HW)
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2, -1))  # (b, heads, HW, C_head)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        out = self.project(out)
        return out


class GDFN(nn.Module):
    """Gated-DConv Feed-Forward Network (simplified)."""
    def __init__(self, dim, expand=2.0, drop=0.0):
        super().__init__()
        hidden = int(dim * expand)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=True)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=True)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=True)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = self.drop(x)
        return x


class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expand=2.0, drop=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expand=ffn_expand, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Down(nn.Module):
    def __init__(self, dim_in, dim_out, depth=1, num_heads=4, drop=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[RestormerBlock(dim_in, num_heads, drop=drop) for _ in range(depth)])
        self.down = nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.blocks(x)
        return self.down(x)


class Up(nn.Module):
    def __init__(self, dim_in, dim_out, depth=1, num_heads=4, drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1)
        self.fuse = nn.Conv2d(dim_out * 2, dim_out, 1)
        self.blocks = nn.Sequential(*[RestormerBlock(dim_out, num_heads, drop=drop) for _ in range(depth)])

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.fuse(x)
        return self.blocks(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=6, dropout_rate=0.0, norm="batch"):
        super().__init__()
        # Map n_residual_blocks to per-stage depths (keep small by default)
        base = 64
        depth_each = max(1, n_residual_blocks // 3)
        heads = 4

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, base, 3),
        )

        self.down1 = Down(base, base * 2, depth=depth_each, num_heads=heads, drop=dropout_rate)
        self.down2 = Down(base * 2, base * 4, depth=depth_each, num_heads=heads, drop=dropout_rate)
        self.down3 = Down(base * 4, base * 8, depth=depth_each, num_heads=heads, drop=dropout_rate)

        self.bottleneck = nn.Sequential(*[RestormerBlock(base * 8, num_heads=heads, drop=dropout_rate) for _ in range(depth_each)])

        self.up3 = Up(base * 8, base * 4, depth=depth_each, num_heads=heads, drop=dropout_rate)
        self.up2 = Up(base * 4, base * 2, depth=depth_each, num_heads=heads, drop=dropout_rate)
        self.up1 = Up(base * 2, base, depth=depth_each, num_heads=heads, drop=dropout_rate)

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
