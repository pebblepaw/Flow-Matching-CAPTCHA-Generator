"""
Rectified Flow Model for CAPTCHA Generation
U-Net architecture with attention for high-quality image generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for flow time parameter"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, H * W)  # B, C, HW
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        
        # Attention
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        
        return x + self.proj(h)


class ResBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class DownBlock(nn.Module):
    """Downsampling block with residual connections"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2, use_attention=False):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb):
        for resblock in self.resblocks:
            x = resblock(x, time_emb)
        x = self.attention(x)
        return self.downsample(x), x


class UpBlock(nn.Module):
    """Upsampling block with residual connections and skip connections"""
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, num_layers=2, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # First resblock handles concatenated channels (upsampled + skip)
        # Then convert to out_channels for subsequent layers
        self.resblocks = nn.ModuleList([
            ResBlock(in_channels + skip_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for resblock in self.resblocks:
            x = resblock(x, time_emb)
        x = self.attention(x)
        return x


class UNetFlowModel(nn.Module):
    """
    U-Net based Rectified Flow model for CAPTCHA generation
    Input: (B, 3, 80, 640) - CAPTCHA images
    Output: (B, 3, 80, 640) - Velocity field for flow matching
    """
    def __init__(
        self,
        img_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.1
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels * mult for mult in channel_multipliers]
        
        for i in range(len(channel_multipliers)):
            in_ch = base_channels if i == 0 else channels[i-1]
            out_ch = channels[i]
            use_attention = i >= len(channel_multipliers) - 2  # Attention in last 2 layers
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_emb_dim, num_res_blocks, use_attention)
            )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], channels[-1], time_emb_dim),
            AttentionBlock(channels[-1]),
            ResBlock(channels[-1], channels[-1], time_emb_dim)
        )
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        for i in range(len(channel_multipliers)):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i]  # Skip connection from corresponding encoder level
            out_ch = reversed_channels[i+1] if i < len(channel_multipliers) - 1 else base_channels
            use_attention = i < 2  # Attention in first 2 layers
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, time_emb_dim, num_res_blocks, use_attention)
            )
        
        # Final layers
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )
    
    def forward(self, x, time):
        """
        Args:
            x: (B, C, H, W) - Input image at time t
            time: (B,) - Flow time parameter in [0, 1]
        Returns:
            (B, C, H, W) - Predicted velocity field
        """
        # Time embedding
        time_emb = self.time_mlp(time)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Downsampling with skip connections
        skips = []
        for down in self.down_blocks:
            x, skip = down(x, time_emb)
            skips.append(skip)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        
        # Upsampling with skip connections
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, time_emb)
        
        # Final output
        return self.final(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = UNetFlowModel(
        img_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 80, 640)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
