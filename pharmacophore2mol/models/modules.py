from functools import partial
import math

import torch
import torch.nn as nn

class NormActConv3d(nn.Module):
    """
    A convolution block consisting of GroupNorm -> SiLU -> Conv3D.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        return self.conv(x)


class ResNetV2Block(nn.Module):
    """
    A ResNet block consisting of two NormActConv3d layers with a skip connection.
    Includes timestep embedding injection.
    """
    def __init__(self, in_channels, out_channels, emb_channels, groups=8):
        super().__init__()
        self.conv1 = NormActConv3d(in_channels, out_channels, groups=groups)
        
        self.emb_proj = TimeProjection(emb_channels, out_channels)
        
        self.conv2 = NormActConv3d(out_channels, out_channels, groups=groups)

        self.shortcut = OptionalProjection(in_channels, out_channels)

    def forward(self, x, emb):
        identity = self.shortcut(x)
        out = self.conv1(x)
        
        local_emb = self.emb_proj(emb)
        out = out + local_emb[:, :, None, None, None]
        
        out = self.conv2(out)
        return out + identity


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
        
        # Second linear layer and act
        self.lin2 = TimeProjection(self.n_channels, self.n_channels)

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
        emb = self.lin1(emb)
        emb = self.lin2(emb)

        #
        return emb
    
class TimeProjection(nn.Module):
    """
    A module that projects the time embedding to a specified number of channels.
    """
    def __init__(self, emb_channels, out_channels):
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Linear(emb_channels, out_channels)

    def forward(self, t_emb):
        return self.proj(self.act(t_emb))
    

class OptionalProjection(nn.Module):
    """
    A module that optionally applies a linear projection to the input tensor.
    If the input and output channels are the same, it returns the input tensor unchanged.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)


class MLP(nn.Module):
    """
    A fully connected MLP block with LayerNorm and SiLU activation.
    """
    def __init__(self, in_features, out_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """
    A 3D UNet architecture for processing voxelized molecular data.
    """
    def __init__(self,
                in_channels,
                out_channels,
                channels=[64, 128, 256, 512],
                t_emb_channels=128,
                groups=8,
                n_res_blocks=2):
        
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.n_res_blocks = n_res_blocks
        self.channels = channels # + channels[len(channels)-2::-1]
        # self.init_proj = NormActConv3d(in_channels, channels[0], kernel_size=1, stride=1, padding=0, groups=self.groups)
        # self.final_proj = NormActConv3d(channels[0], out_channels, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.t_emb = TimeEmbedding(t_emb_channels)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        #downs
        for i in range(len(channels)-1):
            current_channels = self.channels[i]
            down_block = nn.ModuleList()
            for _ in range(n_res_blocks):
                down_block.append(ResNetV2Block(self.channels[i], self.channels[i], t_emb_channels, groups=self.groups))
            
            self.downs.append(down_block)
                
                
        for i in range(len(channels)-1, 0, -1):
            for _ in range(n_res_blocks):
                self.ups.append(ResNetV2Block(self.channels[i], self.channels[i], t_emb_channels, groups=self.groups))
            self.resize.append(nn.ConvTranspose3d(self.channels[i], self.channels[i-1], kernel_size=4, stride=2, padding=0))

    

    def forward(self, x, t):
        #global time embedding
        t_emb = self.t_emb(t)
        #down path
        for d in self.downs:
            x = d(x, t_emb)



if __name__ == "__main__":
    unet = UNet3D(in_channels=4, out_channels=4)
    print(unet)
