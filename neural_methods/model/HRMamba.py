import math
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM


# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def output_head(dim: int):
    """
    Creates a head for the output layer of a model.

    Args:
        dim (int): The input dimension of the head.
        num_classes (int): The number of output classes.

    Returns:
        nn.Sequential: The output head module.
    """
    """    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )
"""
    return nn.Sequential(
        nn.Conv1d(dim, 1, 1, stride=1, padding=0),
    )

class PPG_Linear(nn.Module):
    def __init__(self, T, hidden_dim, bias=True):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, T, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, T))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.T):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.T):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class VisionEncoderMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        heads (int): The number of heads in the multi-head attention mechanism.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.


    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
            dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,

    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor): # 640,127,256
        # x is of shape [batch_size, seq_len, dim]
        b, s, d = x.shape

        # Skip connection
        skip = x    # b,320,256

        # Normalization
        x = self.norm(x) # b,320,256

        # Split x into x1 and x2 with linears
        z1 = self.proj(x) # b,320,256
        x1 = self.proj(x) # b,320,256

        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s") # b,256,320
        forward_conv_output = self.forward_conv1d(x1_rearranged) # b,256,320
        forward_conv_output = rearrange(
            forward_conv_output, "b d s -> b s d"
        ) # b,320,256
        x1_ssm = self.ssm(forward_conv_output)

        # backward conv x2
        x2_rearranged = rearrange(x1, "b s d -> b d s") # b,256,320
        x2 = self.backward_conv1d(x2_rearranged)    # b,256,320
        x2 = rearrange(x2, "b d s -> b s d")    # b,320,256

        # Backward ssm
        x2 = self.ssm(x2)   # b,320,256

        # Activation
        z = self.activation(z1)

        # matmul with z + backward ssm
        x2 = x2 * z

        # Matmul with z and x1
        x1 = x1_ssm * z

        # Add both matmuls
        x = x1 + x2

        # Add skip connection
        return x + skip


class Vim(nn.Module):
    """
    Vision Mamba (Vim) model implementation.

    Args:
        dim (int): Dimension of the model.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dt_rank (int, optional): Rank of the dynamic tensor. Defaults to 32.
        dim_inner (int, optional): Inner dimension of the model. Defaults to None.
        d_state (int, optional): State dimension of the model. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to None.
        image_size (int, optional): Size of the input image. Defaults to 224.
        patch_size (int, optional): Size of the image patch. Defaults to 16.
        channels (int, optional): Number of image channels. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        depth (int, optional): Number of encoder layers. Defaults to 12.

    Attributes:
        dim (int): Dimension of the model.
        heads (int): Number of attention heads.
        dt_rank (int): Rank of the dynamic tensor.
        dim_inner (int): Inner dimension of the model.
        d_state (int): State dimension of the model.
        num_classes (int): Number of output classes.
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        channels (int): Number of image channels.
        dropout (float): Dropout rate.
        depth (int): Number of encoder layers.
        to_patch_embedding (nn.Sequential): Sequential module for patch embedding.
        dropout (nn.Dropout): Dropout module.
        cls_token (nn.Parameter): Class token parameter.
        to_latent (nn.Identity): Identity module for latent representation.
        layers (nn.ModuleList): List of encoder layers.
        output_head (output_head): Output head module.

    """

    def __init__(
        self,
        dim: int =256,
        heads: int = 8,
        dt_rank: int = 32,
        dim_inner: int = 256,
        d_state: int = 256,
        num_classes: int = 1,
        image_size: int = 128,
        patch_size: int = 16,
        channels: int = 3,
        dropout: float = 0.1,
        depth: int = 12,
        T: int = 160,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.dropout = dropout
        self.depth = depth
        self.T = T

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        self.blender = nn.Conv2d(3, 1, 1)
        self.Stem0 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.Stem1 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.Stem2 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        '''
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, (1, 5, 5), stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        '''

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_height,
            ),
            nn.Linear(patch_dim, dim),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Latent
        self.to_latent = nn.Identity()

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                VisionEncoderMambaBlock(
                    dim=dim,
                    heads=heads,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )

        # Output head
        self.output_head = output_head(dim)

        self.dim_linear = nn.Linear(2*self.T, self.T)
        self.ppg_linear = PPG_Linear(self.T, self.dim, bias=True)


    def forward(self, x: torch.Tensor): # 4,160,3,128,128
        # Patch embedding
        b, t, c, h, w = x.shape     # 1,3,160,128,128   7864320
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, c, h, w)
        x = self.blender(x)
        x = x.squeeze(1)
        x = x.reshape(b, t, h, w)   # b,160,128,128

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)   # b,160,16,16

        x = x.reshape(b, t, -1)
        #print(f"Patch embedding: {x.shape}")

        # x = self.Stem0(x)           # 1,64,160,64,64    41943040
        # x = self.Stem1(x)           # 1,128,160,32,32   20971520
        # x = self.Stem2(x)           # 1,256,160,16,16   10485760
        # x = rearrange(x, 'b c t h w -> (b t) c h w') # 160,256,16,16
        # x = self.to_patch_embedding(x) # 640,64,256
        # print(f"Patch embedding: {x.shape}")

        # Temporal difference
        x_diff = x[:, 1:] - x[:, :-1] # b,159,256
        x = torch.cat((x, x_diff), dim=1) # b,160+159,256

        # Shape
        b, n, _ = x.shape

        # Cls tokens
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b) # 640,1,256
        #print(f"Cls tokens: {cls_tokens.shape}")

        # Concatenate
        x = torch.cat((cls_tokens, x), dim=1)   # b,320,256

        # Dropout
        x = self.dropout(x)
        #print(x.shape)

        # Forward pass with the layers
        for layer in self.layers:
            x = layer(x)
            #print(f"Layer: {x.shape}")

        # Latent
        x = self.to_latent(x)

        # Output head with the cls tokens
        rPPG = x.permute(0, 2, 1)
        rPPG = self.dim_linear(rPPG)
        rPPG = rPPG.permute(0, 2, 1)
        rPPG = self.ppg_linear(rPPG)
        return rPPG


