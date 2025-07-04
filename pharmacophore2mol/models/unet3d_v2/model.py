"""
UNet3D Model
Originally based of a normal UNET
Implemented a bunch of mods (residual blocks, attention blocks, etc.) as seen in https://nn.labml.ai/diffusion/ddpm/unet.html
Converted to 3D UNet, and also kept the time embeddings for diffusion optional
Also made some mods, namely activations, as seen on voxmol https://github.com/Genentech/voxmol/blob/main/voxmol/models/unet3d.py
"""


from tqdm import tqdm
import math
from typing import Optional, Tuple, Union, List
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = nn.SiLU()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for group normalization
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = nn.SiLU()

        self.dropout = nn.Dropout(dropout) #TODO droput doesn't go well with batchnorm (what about groupnorm?), only use in the end (of the block is acceptable? check this)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, depth, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h)))) #TODO: is this ordering right?

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for group normalization
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, depth, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, depth, height, width = x.shape
        # Flatten spatial dimensions (depth, height, width) into a sequence
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)  # [B, D*H*W, C]
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, depth, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool, n_groups: int = 32):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, n_groups=n_groups)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool, n_groups: int = 32):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels, n_groups=n_groups)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, n_groups: int = 32):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, n_groups=n_groups)
        self.attn = AttentionBlock(n_channels, n_groups=n_groups)
        # self.attn = nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, n_groups=n_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet3DV2(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, in_channels: int = 3,
                    out_channels: int = 8,
                    n_internal_channels: int = 64,
                    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                    is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                    n_blocks: int = 2,
                    n_groups: int = 8):
        """
        * `input_channels` is the number of channels in the input.
        * `output_channels` is the number of channels in the output.
        * `n_internal_channels` is the base number of channels into which the input is projected initially and that serves as the reference for the width of the rest of the network, being used along the multipliers in `ch_mults` to determine the number of channels at each resolution.
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution. WARNING: there's an attention block not controlled by this parameter in the middle bottleneck of the UNet
        * `n_blocks` is the number of `Up/DownBlocks` at each resolution (number of blocks before reducing or increasing size)
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html). Default is 8, 1 is LayerNorm
        """
        super().__init__()
        self.n_groups = n_groups
        assert len(ch_mults) == len(is_attn), "The number of channel multipliers and attention flags must match"
        assert n_internal_channels % n_groups == 0, "The number of internal channels must be divisible by the number of groups for group normalization"

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv3d(in_channels, n_internal_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_internal_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        current_channels = prev_channels = n_internal_channels #i know the names are somewhat confusing, but the in and out are just for internals, not 
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            current_channels = prev_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(prev_channels, current_channels, n_internal_channels * 4, is_attn[i], n_groups=self.n_groups))
                prev_channels = current_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(prev_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(current_channels, n_internal_channels * 4, n_groups=self.n_groups)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        prev_channels = current_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            current_channels = prev_channels
            for _ in range(n_blocks):
                up.append(UpBlock(prev_channels, current_channels, n_internal_channels * 4, is_attn[i], n_groups=self.n_groups))
            # Final block to reduce the number of channels
            current_channels = prev_channels // ch_mults[i]
            up.append(UpBlock(prev_channels, current_channels, n_internal_channels * 4, is_attn[i], n_groups=self.n_groups))
            prev_channels = current_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(prev_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(self.n_groups, n_internal_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv3d(prev_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """



        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

if __name__ == "__main__":
    # Example usage #TODO: check why trainable param count is so low
    bs = 2
    in_c = 5
    out_c = 7
    d = h = w = 32
    input_x = torch.randn(bs, in_c, d, h, w)  # Batch size of 2, 5 channels, 32x32x32 grid
    input_t = torch.randint(0, 1000, (bs,))  # Random timesteps for the batch
    print(input_t)
    model = UNet3DV2(in_channels=in_c, out_channels=out_c, n_internal_channels=64, ch_mults=(1, 2, 2, 4), is_attn=(False, False, True, True), n_blocks=2)
    print(f"Input shape: {input_x.shape}")  # Should be (2, 5, 32, 32)
    output = model(input_x, input_t)
    print(f"Output shape: {output.shape}")  # Should be (2, 8, 32, 32)
    print("trainable parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
