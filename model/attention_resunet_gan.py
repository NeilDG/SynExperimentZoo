import torch
import torch.nn as nn
import torch.nn.functional as F

"""
AttentionResUNet Generator (2023-2026 style)
- CNN U-Net backbone with modern attention blocks (Large Kernel Attention + ECA)
- Follows project conventions: expose class Generator with (input_nc, output_nc, n_residual_blocks/num_blocks, dropout_rate, norm)
- Output activation: Tanh (per user preference)
- No upscaling (1x image-to-image), intended for paired translation / restoration

Design notes
- LKA (large-kernel attention) approximated via depthwise conv k=5 + dilated depthwise conv k=7,d=3 + pointwise conv producing an attention map
- ECA channel attention for lightweight channel reweighting
- UNet-style skip connections with attention at each stage
"""


def _get_norm(norm: str, ch: int):
    if norm == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    elif norm == "layer":
        # simple per-channel LayerNorm proxy using GroupNorm with 1 group
        return nn.GroupNorm(1, ch)
    else:
        return nn.BatchNorm2d(ch)


def xavier_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ECA(nn.Module):
    """Efficient Channel Attention (lightweight SOTA channel attn)."""
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: B,C,H,W
        y = self.avg(x)  # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(1, 2))  # B,1,C -> conv1d along C
        y = y.transpose(1, 2).unsqueeze(-1)  # B,C,1,1
        y = self.sig(y)
        return x * y


class LKA(nn.Module):
    """Large Kernel Attention from VAN (depthwise conv + dilated depthwise conv + pw conv)."""
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=7, padding=9, dilation=3, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        u = x
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.pw(x)
        return u * self.sig(x)


class AttnConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="batch", dropout=0.0, use_lka=True, use_eca=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = _get_norm(norm, out_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = _get_norm(norm, out_ch)
        self.act2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_res = (in_ch == out_ch)
        self.attn = nn.Sequential(*(filter(None, [LKA(out_ch) if use_lka else None, ECA(out_ch) if use_eca else None])))

    def forward(self, x):
        identity = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.drop(x)
        if len(self.attn) > 0:
            x = self.attn(x)
        if self.use_res:
            x = x + identity
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm="batch", dropout=0.0):
        super().__init__()
        self.block = AttnConvBlock(in_ch, out_ch, norm, dropout)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.block(x)
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm="batch", dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=0)
        self.block = AttnConvBlock(in_ch=out_ch*2, out_ch=out_ch, norm=norm, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed for odd shapes
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=6, dropout_rate=0.0, norm="batch"):
        super().__init__()
        # Map num_blocks to depth: 4 stages encoder/decoder typical; n_residual_blocks controls bottleneck depth
        width = 64
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, width, 3),
            _get_norm(norm, width),
            nn.ReLU(inplace=True)
        )

        self.down1 = Down(width, width*2, norm, dropout_rate)
        self.down2 = Down(width*2, width*4, norm, dropout_rate)
        self.down3 = Down(width*4, width*8, norm, dropout_rate)

        blocks = []
        for _ in range(max(1, n_residual_blocks)):
            blocks.append(AttnConvBlock(width*8, width*8, norm, dropout_rate, use_lka=True, use_eca=True))
        self.bottleneck = nn.Sequential(*blocks)

        self.up3 = Up(width*8, width*4, norm, dropout_rate)
        self.up2 = Up(width*4, width*2, norm, dropout_rate)
        self.up1 = Up(width*2, width, norm, dropout_rate)

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(width, output_nc, 3),
            nn.Tanh()
        )

        self.apply(xavier_weights_init)

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
