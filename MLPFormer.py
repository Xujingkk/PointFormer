import torch
import torch.nn as nn
from einops import rearrange
import einops
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
# from thop import clever_format
# from thop import profile
from model.segformer import *
import torch.utils.checkpoint as checkpoint
from model.AFE import AFE
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class LocalAttention(nn.Module):
    def __init__(self, dim, resolution, heads, local_size):
        super(LocalAttention, self).__init__()
        assert dim % heads == 0
        self.dim = dim
        self.resolution = resolution
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.local_H, self.local_W = local_size[0], local_size[1]

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def img2local(self, img):
        B, N, C = img.shape
        H, W = self.resolution
        # print("self.local_H", H, self.local_H)
        img = img.transpose(-2, -1).contiguous().view(B, C, H, W)
        img_reshape = img.view(B, C, H // self.local_H, self.local_H,
                               W // self.local_W, self.local_W)
        img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1) \
            .contiguous().reshape(-1, self.local_H * self.local_W, C)
        x = img_perm.reshape(-1, self.local_H * self.local_W,
                             self.heads, C // self.heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        x = x.view(B, C, H // self.local_H, self.local_H, W // self.local_W, self.local_W)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, self.local_H, self.local_W)  # B', C, H', W'

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, self.heads, C // self.heads,
                            self.local_H * self.local_W).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.heads, C // self.heads,
                      self.local_H * self.local_W).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]

        H, W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.img2local(q)
        k = self.img2local(k)
        # print(q.shape)
        v, lepe = self.get_lepe(v, self.get_v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.local_H * self.local_W, C)  # B head N N @ B head N C

        x = x.view(B, H // self.local_H, W // self.local_W, self.local_H, self.local_W, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        x = x.view(B, -1, C)
        return x


class Cross_LocalAttention(nn.Module):
    def __init__(self, dim, resolution, heads, local_size):
        super(Cross_LocalAttention, self).__init__()
        assert dim % heads == 0
        self.dim = dim
        self.resolution = resolution
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.local_H, self.local_W = local_size[0], local_size[1]

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def img2local(self, img):
        B, N, C = img.shape
        H, W = self.resolution
        img = img.transpose(-2, -1).contiguous().view(B, C, H, W)
        img_reshape = img.view(B, C, H // self.local_H, self.local_H,
                               W // self.local_W, self.local_W)
        img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1) \
            .contiguous().reshape(-1, self.local_H * self.local_W, C)
        x = img_perm.reshape(-1, self.local_H * self.local_W,
                             self.heads, C // self.heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        x = x.view(B, C, H // self.local_H, self.local_H, W // self.local_W, self.local_W)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, self.local_H, self.local_W)  # B', C, H', W'

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, self.heads, C // self.heads,
                            self.local_H * self.local_W).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.heads, C // self.heads,
                      self.local_H * self.local_W).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def points_nms(self, heat, kernel=3):
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def forward(self, qkv, cross_q=None):
        q, k, v = qkv[0], qkv[1], qkv[2]

        H, W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.img2local(q)
        k = self.img2local(k)

        if cross_q is not None:
            B_, head1, N1, C1 = cross_q.shape
            cross_q = cross_q.permute(0, 2, 1, 3)
            cross_q = cross_q.reshape(B_, N1, head1 * C1).permute(0, 2, 1)
            cross_q = cross_q.reshape(B_, head1 * C1, int(N1 ** 0.5), int(N1 ** 0.5))
            # print("cross_q0", cross_q.shape)
            cross_q = self.points_nms(cross_q.sigmoid(), kernel=3)
            # print("cross_q1", cross_q.shape)
            model_q = AFE(gate_channels=head1 * C1)
            model_q = model_q.to(device)
            cross_q = model_q(cross_q)
            # print("cross_q2", cross_q.shape)
            B_, C2, H, W = cross_q.shape
            cross_q = cross_q.reshape(B_, head1, C2 // head1, H * W).permute(0, 1, 3, 2)
            # print("cross_q", q.shape, cross_q.shape)
            # print("222222222222")
            q = q + cross_q
        # print(q.shape)
        v, lepe = self.get_lepe(v, self.get_v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.local_H * self.local_W, C)  # B head N N @ B head N C

        x = x.view(B, H // self.local_H, W // self.local_W, self.local_H, self.local_W, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        x = x.view(B, -1, C)
        return x, q


class SelfCrossAttention(nn.Module):
    def __init__(self, dim, resolution, heads, window_size, cross_size: tuple):
        super(SelfCrossAttention, self).__init__()
        self.dim = dim
        self.resolution = resolution
        self.heads = heads
        self.window_size = window_size
        self.cross_size = cross_size

        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        # self.get_lepe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def img2cross(self, img):
        B, N, C = img.shape
        H = W = self.resolution
        img = img.reshape(B, H, W, C)
        img_reshape = img.reshape(B, H // self.cross_size[0], self.cross_size[0],
                                  W // self.cross_size[1], self.cross_size[1],
                                  self.heads, C // self.heads)
        img_perm = img_reshape.permute(0, 5, 1, 3, 2, 4, 6).contiguous() \
            .reshape(-1, H // self.cross_size[0], W // self.cross_size[1],
                     self.cross_size[0] * self.cross_size[1], C // self.heads)

        return img_perm

    def img2window(self, img):
        B, N, C = img.shape
        H = W = self.resolution
        img = img.reshape(B, H, W, C)
        img_reshape = img.reshape(B, H // self.window_size, self.window_size,
                                  W // self.window_size, self.window_size,
                                  self.heads, C // self.heads)
        img_perm = img_reshape.permute(0, 5, 1, 3, 2, 4, 6).contiguous() \
            .reshape(-1, H // self.window_size, W // self.window_size,
                     self.window_size * self.window_size, C // self.heads)
        return img_perm

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]

        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.img2window(q)
        k = self.img2cross(k)
        v = self.img2cross(v)

        print(q.shape, k.shape, v.shape)
        # k = k.repeat_interleave(self.cross_size[1] // self.window_size, 2) \
        #     .repeat_interleave(self.cross_size[0] // self.window_size, 1)
        # v = v.repeat_interleave(self.cross_size[1] // self.window_size, 2) \
        #     .repeat_interleave(self.cross_size[0] // self.window_size, 1)
        # print(q.shape, k.shape, v.shape)

        q = q * self.scale
        attn = torch.einsum('bhwnc, bqkmc -> bhwnm', q, k)

        print(attn.shape)
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        x = (attn @ v)
        x = x.reshape(B, self.heads, H // self.window_size, W // self.window_size,
                      self.window_size, self.window_size, C // self.heads) \
            .permute(0, 2, 4, 3, 5, 1, 6).reshape(B, H, W, C).reshape(B, -1, C)
        # print(x.shape)
        return x


class LocalConv(nn.Module):
    def __init__(self, dim, resolution):
        super(LocalConv, self).__init__()
        self.resolution = resolution
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    def forward(self, x):
        H, W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(-2, -1).contiguous()
        return x


class LocalAttentionBlock(nn.Module):
    def __init__(self, dim, resolution, heads, window_size, token_mlp='mix'):
        super(LocalAttentionBlock, self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.resolution = resolution
        self.heads = heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim // 2, 3 * (dim // 2), bias=True)
        self.norm1 = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_local = LocalAttention(dim // 2, resolution, heads,
                                         (window_size, window_size))
        self.conv_local = LocalConv(dim // 2, resolution)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x):
        # print("self.resolution",self.resolution)
        H, W = self.resolution
        B, L, C = x.shape
        # print("blc", B, L, C, H)
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img[:, :, : C // 2]).reshape(B, -1, 3, C // 2).permute(2, 0, 1, 3).contiguous()
        x1 = self.attn_local(qkv)
        x2 = self.conv_local(img[:, :, C // 2:])
        local_x = torch.cat((x1, x2), dim=2)
        local_x = self.proj(local_x)
        x = x + local_x

        x = x + (self.mlp(self.norm2(x), H, W))
        # print('a', x.shape)
        return x


class Cross_LocalAttentionBlock(nn.Module):
    def __init__(self, dim, resolution, heads, window_size, token_mlp='mix'):
        super(Cross_LocalAttentionBlock, self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.resolution = resolution
        self.heads = heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim // 2, 3 * (dim // 2), bias=True)
        self.norm1 = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_local = Cross_LocalAttention(dim // 2, resolution, heads,
                                         (window_size, window_size))
        self.conv_local = LocalConv(dim // 2, resolution)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x, cross_q=None):
        H, W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img[:, :, : C // 2]).reshape(B, -1, 3, C // 2).permute(2, 0, 1, 3).contiguous()
        x1, q = self.attn_local(qkv, cross_q=cross_q)
        x2 = self.conv_local(img[:, :, C // 2:])
        local_x = torch.cat((x1, x2), dim=2)
        local_x = self.proj(local_x)
        x = x + local_x

        x = x + (self.mlp(self.norm2(x), H, W))
        # print('a', x.shape)
        return x, q


class SelfCrossAttnBlock(nn.Module):
    def __init__(self, dim, resolution, heads, window_size, cross_size, token_mlp='mix'):
        super(SelfCrossAttnBlock, self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.resolution = resolution
        self.heads = heads
        self.window_size = window_size
        self.cross_size = cross_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)

        self.selfCrossAttn = SelfCrossAttention(dim, resolution, heads,
                                                window_size, (cross_size, cross_size))
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x):
        H = W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()

        x1 = self.selfCrossAttn(qkv)
        global_x = self.proj(x1)
        x = x + global_x

        x = x + (self.mlp(self.norm2(x), H, W))
        # print('b', x.shape)
        return x


class GlobalAttentionBlock(nn.Module):
    def __init__(self, dim, resolution, heads, split_size,
                 token_mlp='mix'):
        super(GlobalAttentionBlock, self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.heads = heads
        self.resolution = resolution
        self.split_size = split_size

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attns = nn.ModuleList()
        self.attns.append(
            LocalAttention(dim // 2, resolution, heads, (resolution[0], split_size))
        )
        self.attns.append(
            LocalAttention(dim // 2, resolution, heads, (split_size, resolution[1]))
        )
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x):
        H, W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()

        x1 = self.attns[0](qkv[:, :, :, :C // 2])
        x2 = self.attns[1](qkv[:, :, :, C // 2:])
        global_x = torch.cat((x1, x2), dim=2)
        global_x = self.proj(global_x)
        x = x + global_x

        x = x + (self.mlp(self.norm2(x), H, W))
        # print('b', x.shape)
        return x


class DualTransformerBlock(nn.Module):
    def __init__(self, dim, resolution, heads, split_size,
                 encoder_last_stage=False, token_mlp='mix'):
        super(DualTransformerBlock, self).__init__()

        self.encoder_last_stage = encoder_last_stage
        self.local_perception = LocalAttentionBlock(
            dim=dim,
            resolution=resolution,
            heads=heads,
            window_size=8,
            token_mlp=token_mlp
        )
        if not encoder_last_stage:
            self.global_perception = GlobalAttentionBlock(
                dim=dim,
                resolution=resolution,
                heads=heads,
                split_size=split_size,
                token_mlp=token_mlp
            )

    def forward(self, x):
        x = self.local_perception(x)
        if not self.encoder_last_stage:
            x = self.global_perception(x)
        return x

class Cross_DualTransformerBlock(nn.Module):
    def __init__(self, dim, resolution, heads, split_size,
                 encoder_last_stage=False, token_mlp='mix'):
        super(Cross_DualTransformerBlock, self).__init__()

        self.encoder_last_stage = encoder_last_stage
        self.local_perception = Cross_LocalAttentionBlock(
            dim=dim,
            resolution=resolution,
            heads=heads,
            window_size=8,
            token_mlp=token_mlp
        )
        if not encoder_last_stage:
            self.global_perception = GlobalAttentionBlock(
                dim=dim,
                resolution=resolution,
                heads=heads,
                split_size=split_size,
                token_mlp=token_mlp
            )

    def forward(self, x, cross_q=None):
        x, q = self.local_perception(x, cross_q=cross_q)
        if not self.encoder_last_stage:
            x = self.global_perception(x)
        return x, q


class DualTransformerBlock2(nn.Module):
    def __init__(self, dim, resolution, heads, cross_size,
                 encoder_last_stage=False, token_mlp='mix'):
        super(DualTransformerBlock2, self).__init__()

        self.encoder_last_stage = encoder_last_stage
        self.local_perception = LocalAttentionBlock(
            dim=dim,
            resolution=resolution,
            heads=heads,
            window_size=7,
            token_mlp=token_mlp
        )
        if not encoder_last_stage:
            self.global_perception = SelfCrossAttnBlock(
                dim=dim,
                resolution=resolution,
                heads=heads,
                window_size=7,
                cross_size=cross_size,
                token_mlp=token_mlp
            )

    def forward(self, x):
        x = self.local_perception(x)
        if not self.encoder_last_stage:
            x = self.global_perception(x)
        return x


# Encoder
class MiT(nn.Module):
    def __init__(self, image_size, in_dim, layers, token_mlp='mix_skip'):
        super().__init__()
        patch_sizes = [8, 4, 4, 4]
        strides = [4, 2, 2, 2]
        padding_sizes = [2, 1, 1, 1]
        split_sizes = [1, 2, 4, 8]
        # cross_sizes = [14, 14, 14, 14]
        heads = [1, 2, 5, 8]
        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 6, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )
        self.patch_embed4 = OverlapPatchEmbeddings(
            image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], in_dim[2], in_dim[3]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], image_size // 4, heads[0],
                                  split_sizes[0], False, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], image_size // 8, heads[1],
                                  split_sizes[1], False, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], image_size // 16, heads[2],
                                  split_sizes[2], False, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

        self.block4 = nn.ModuleList(
            [DualTransformerBlock(in_dim[3], image_size // 32, heads[3],
                                  split_sizes[3], True, token_mlp) for _ in range(layers[3])]
        )
        self.norm4 = nn.LayerNorm(in_dim[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        # stage 1
        x, H, W = self.patch_embed1(x)
        # print('aa', x.shape)
        for blk in self.block1:
            x = blk(x)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


# Decoder
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H = W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
            self, input_size, in_out_chan, heads, cross_size, layers, n_class=9,
            norm_layer=nn.LayerNorm, encoder_last_stage=False,
            token_mlp='mix', is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        self.dims = dims
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size,
                dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.blocks = nn.ModuleList(
            [DualTransformerBlock(out_dim, input_size, heads, cross_size,
                                  encoder_last_stage, token_mlp) for _ in range(layers)]
        )

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            # print(x1.shape, x2.shape, cat_x.shape, self.dims, cat_linear_x.shape)
            for blk in self.blocks:
                cat_linear_x = blk(cat_linear_x)
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(cat_linear_x).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2)).contiguous()
            else:
                out = self.layer_up(cat_linear_x)
        else:
            out = self.layer_up(x1)
        return out


class MLAFormer(nn.Module):
    def __init__(self, num_classes=6, token_mlp_mode="mix_skip", cca_stages=[]):
        super().__init__()

        self.cca_stages = cca_stages
        dims = [64, 128, 320, 512]
        self.backbone = MiT(256, dims, layers=[1, 2, 2, 2])

        self.tissue_classifier = TissueClassifier(dims[-1], tissue_classes=19)
        self.seg = Segmentation(num_classes, 1)

        if len(cca_stages) != 0:
            self.ccaForSC = CCABlockForSC(
                dims=dims,
                resolutions=[56, 28, 14, 7],
                heads=[1, 2, 5],
                stages=cca_stages,
                token_mlp=token_mlp_mode
            )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder = self.backbone(x)

        if len(self.cca_stages) != 0:
            encoder = self.ccaForSC(encoder)

        classifier_x = self.tissue_classifier(encoder[-1])
        sem_x, inst_x = self.seg(encoder)
        print(classifier_x.shape, sem_x.shape, inst_x.shape)
        return classifier_x, sem_x, inst_x


class CCA(nn.Module):
    def __init__(self, dim, fore_dim, post_dim, resolution, heads):
        super(CCA, self).__init__()
        self.heads = heads
        self.resolution = resolution
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.proj_kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj_q1 = nn.Conv2d(fore_dim, dim, 3, 2, 1)
        self.proj_q2 = nn.ConvTranspose2d(post_dim, dim, 2, 2)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, fore_x, post_x):
        H = W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        kv = self.proj_kv(x).view(B, L, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]
        if fore_x is not None:
            fore_x = fore_x.reshape(B, 2 * H, 2 * W, -1).permute(0, 3, 1, 2).contiguous()
            q1 = self.proj_q1(fore_x).reshape(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2).contiguous()
            q1 = q1 * self.scale
            attn1 = (q1 @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn1 = F.softmax(attn1, dim=-1, dtype=attn1.dtype)
            x_1 = (attn1 @ v).transpose(1, 2).reshape(B, L, C)
        if post_x is not None:
            post_x = post_x.reshape(B, H // 2, W // 2, -1).permute(0, 3, 1, 2).contiguous()
            q2 = self.proj_q2(post_x).reshape(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2).contiguous()
            q2 = q2 * self.scale
            attn2 = (q2 @ k.transpose(-2, -1))
            attn2 = F.softmax(attn2, dim=-1, dtype=attn2.dtype)
            x_2 = (attn2 @ v).transpose(1, 2).reshape(B, L, C)

        if fore_x is not None and post_x is not None:
            x = x_1 + x_2
        elif fore_x is not None and post_x is None:
            x = x_1
        elif fore_x is None and post_x is not None:
            x = x_2
        else:
            raise ValueError('Error!')
        x = self.proj(x)
        return x


class CCA2(nn.Module):
    def __init__(self, dim, fore_dim, post_dim, resolution, heads):
        super(CCA2, self).__init__()
        self.heads = heads
        self.resolution = resolution
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim)

        if fore_dim is not None:
            self.proj_kv1 = nn.Linear(fore_dim, dim * 2)
        self.proj_kv2 = nn.Linear(post_dim, dim * 2)

        if fore_dim is not None:
            self.proj = nn.Linear(dim * 3, dim)
        else:
            self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x, fore_x, post_x):
        H = W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        q = self.proj_q(x).reshape(B, L, self.heads, C // self.heads).permute(0, 2, 1, 3).contiguous()
        q = q * self.scale

        kv2 = self.proj_kv2(post_x).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = kv2[0], kv2[1]
        attn2 = (q @ k2.transpose(-2, -1))
        attn2 = F.softmax(attn2, dim=-1, dtype=attn2.dtype)
        x_2 = (attn2 @ v2).transpose(1, 2).reshape(B, L, C)

        if fore_x is not None:
            kv1 = self.proj_kv1(fore_x).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1,
                                                                                               4).contiguous()
            k1, v1 = kv1[0], kv1[1]
            attn1 = (q @ k1.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn1 = F.softmax(attn1, dim=-1, dtype=attn1.dtype)
            x_1 = (attn1 @ v1).transpose(1, 2).reshape(B, L, C)

            x = torch.cat([x, x_1, x_2], dim=2)
            x = self.proj(x)
        else:
            x = torch.cat([x, x_2], dim=2)
            x = self.proj(x)

        return x


class CCABlock(nn.Module):
    def __init__(self, dim, fore_dim, post_dim, resolution, heads, token_mlp='mix_skip'):
        super(CCABlock, self).__init__()
        self.dim = dim
        self.resolution = resolution
        self.norm_x = nn.LayerNorm(dim)
        if fore_dim is not None:
            self.norm_x1 = nn.LayerNorm(fore_dim)
        self.norm_x2 = nn.LayerNorm(post_dim)

        self.attn = CCA2(dim, fore_dim, post_dim, resolution, heads)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x, x1, x2):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        shortcut = x
        x = self.norm_x(x)
        if x1 is not None:
            x1 = x1.permute(0, 2, 3, 1).contiguous().view(B, H * W * 4, -1)
            x1 = self.norm_x1(x1)

        x2 = x2.permute(0, 2, 3, 1).contiguous().view(B, H * W // 4, -1)
        x2 = self.norm_x2(x2)

        x = self.attn(x, x1, x2)
        x = shortcut + x
        x = x + (self.mlp(self.norm2(x), H, W))  # B L C
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        # print(x.shape)
        return x


class CCABlockForSC(nn.Module):
    def __init__(self, dims: list, resolutions: list, heads: list, stages: list, token_mlp='mix_skip'):
        super(CCABlockForSC, self).__init__()
        self.dims = dims
        self.len_sc = len(heads)
        self.stages = stages
        self.cca = nn.ModuleList(
            [None, None, None, None]
        )
        for i in stages:
            if i == 0:
                self.cca[i] = CCABlock(
                    dim=dims[i],
                    fore_dim=None,
                    post_dim=dims[i + 1],
                    resolution=resolutions[i],
                    heads=heads[i],
                    token_mlp=token_mlp
                )
            else:
                self.cca[i] = CCABlock(
                    dim=dims[i],
                    fore_dim=None,
                    post_dim=dims[i + 1],
                    resolution=resolutions[i],
                    heads=heads[i],
                    token_mlp=token_mlp
                )

    def forward(self, skip_connections):
        new_skip_connections = skip_connections.copy()
        for i in self.stages:
            if i == 0:
                new_skip_connections[i] = self.cca[i](skip_connections[i],
                                                      None,
                                                      skip_connections[i + 1])
            else:
                new_skip_connections[i] = self.cca[i](skip_connections[i],
                                                      None,
                                                      skip_connections[i + 1])
        return new_skip_connections


class ForeCrossAttention(nn.Module):
    def __init__(self, dim, fore_dim, resolution, heads):
        super(ForeCrossAttention, self).__init__()
        self.heads = heads
        self.resolution = resolution
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_qv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj_k = nn.Conv2d(fore_dim, dim, 3, 2, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, fore_x):
        H = W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        qv = self.proj_qv(x).view(B, L, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        q, v = qv[0], qv[1]
        # print(q.shape, v.shape)
        q = q * self.scale

        fore_x = fore_x.reshape(B, 2 * H, 2 * W, -1).permute(0, 3, 1, 2).contiguous()
        k = self.proj_k(fore_x).reshape(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2).contiguous()
        # print(k.shape)
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x


class PostCrossAttention(nn.Module):
    def __init__(self, dim, post_dim, resolution, heads):
        super(PostCrossAttention, self).__init__()
        self.heads = heads
        self.resolution = resolution
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_qv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj_k = nn.ConvTranspose2d(post_dim, dim, kernel_size=2, stride=2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, post_x):
        H = W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        qv = self.proj_qv(x).view(B, L, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        q, v = qv[0], qv[1]
        q = q * self.scale

        post_x = post_x.reshape(B, H // 2, W // 2, -1).permute(0, 3, 1, 2).contiguous()
        k = self.proj_k(post_x).reshape(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2).contiguous()
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x


class ContextCrossAttnBlock(nn.Module):
    def __init__(self, dim, fore_dim, post_dim, resolution, heads, token_mlp='mix_skip'):
        super(ContextCrossAttnBlock, self).__init__()
        self.dim = dim
        self.fore_dim = fore_dim
        self.resolution = resolution

        if fore_dim is not None:
            self.norm_x1 = nn.LayerNorm(dim)
            self.norm_fore_x = nn.LayerNorm(fore_dim)
            self.foreCrossAttn = ForeCrossAttention(dim, fore_dim, resolution, heads)
            self.norm_x2 = nn.LayerNorm(dim)

        self.norm_x3 = nn.LayerNorm(dim)
        self.norm_post_x = nn.LayerNorm(post_dim)
        self.postCrossAttn = PostCrossAttention(dim, post_dim, resolution, heads)
        self.norm_x4 = nn.LayerNorm(dim)

        if token_mlp == "mix":
            if fore_dim is not None:
                self.mlp1 = MixFFN(dim, int(dim * 4))
            self.mlp2 = MixFFN(dim, int(dim * 4))
        elif token_mlp == "mix_skip":
            if fore_dim is not None:
                self.mlp1 = MixFFN(dim, int(dim * 4))
            self.mlp2 = MixFFN_skip(dim, int(dim * 4))
        else:
            if fore_dim is not None:
                self.mlp1 = MixFFN(dim, int(dim * 4))
            self.mlp2 = MLP_FFN(dim, int(dim * 4))

    def forward(self, x, fore_x, post_x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        if fore_x is not None and self.fore_dim is not None:
            norm_x1 = self.norm_x1(x)
            fore_x = fore_x.permute(0, 2, 3, 1).contiguous().view(B, H * W * 4, -1)
            norm_fore_x = self.norm_fore_x(fore_x)
            fore_attn = self.foreCrossAttn(norm_x1, norm_fore_x)
            add1 = x + fore_attn

            norm_x2 = self.norm_x2(add1)
            mlp1 = self.mlp1(norm_x2, H, W)
            x = add1 + mlp1

        norm_x3 = self.norm_x3(x)
        post_x = post_x.permute(0, 2, 3, 1).contiguous().view(B, H * W // 4, -1)
        # print('x2:', x2.shape)
        norm_post_x = self.norm_post_x(post_x)
        post_attn = self.postCrossAttn(norm_x3, norm_post_x)
        add3 = x + post_attn

        norm_x4 = self.norm_x4(add3)
        mlp2 = self.mlp2(norm_x4, H, W)
        x = add3 + mlp2
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return x


class ContextCrossAttnForSC(nn.Module):
    def __init__(self, dims: list, resolutions: list, heads: list, stages: list, token_mlp='mix_skip'):
        super(ContextCrossAttnForSC, self).__init__()
        self.dims = dims
        self.len_sc = len(heads)
        self.stages = stages
        self.cca = nn.ModuleList(
            [None, None, None, None]
        )
        for i in stages:
            if i == 0:
                self.cca[i] = ContextCrossAttnBlock(
                    dim=dims[i],
                    fore_dim=None,
                    post_dim=dims[i + 1],
                    resolution=resolutions[i],
                    heads=heads[i],
                    token_mlp=token_mlp
                )
            else:
                self.cca[i] = ContextCrossAttnBlock(
                    dim=dims[i],
                    fore_dim=dims[i - 1],
                    post_dim=dims[i + 1],
                    resolution=resolutions[i],
                    heads=heads[i],
                    token_mlp=token_mlp
                )

    def forward(self, skip_connections):
        new_skip_connections = skip_connections.copy()
        for i in self.stages:
            if i == 0:
                new_skip_connections[i] = self.cca[i](skip_connections[i],
                                                      None,
                                                      skip_connections[i + 1])
            else:
                new_skip_connections[i] = self.cca[i](skip_connections[i],
                                                      skip_connections[i - 1],
                                                      skip_connections[i + 1])
        # print(new_skip_connections[0].requires_grad)
        return new_skip_connections


class TissueClassifier(nn.Module):
    def __init__(self, dim, tissue_classes):
        super(TissueClassifier, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, tissue_classes)

    def forward(self, encode_feature):
        B, C, H, W = encode_feature.shape
        feature_gap = self.gap(encode_feature)
        # print(feature_gap.shape)
        feature_gap = feature_gap.reshape(B, C)
        x = self.fc(feature_gap)
        # print(x.shape)
        return x


class Segmentation(nn.Module):
    def __init__(self, sem_num_classes, inst_num_classes, token_mlp_mode="mix_skip"):
        super(Segmentation, self).__init__()
        split_size = [1, 2, 4, 8]
        heads = [1, 2, 4, 8]
        d_base_feat_size = 8  # 16 for 512 inputsize   7 for 224
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]
        sem_layers = [1, 2, 2, 1]
        inst_layers = [1, 1, 1, 1]

        self.sem_decoder_3 = MyDecoderLayer(d_base_feat_size, in_out_chan[3], heads[3],
                                            split_size[3], sem_layers[3], n_class=sem_num_classes,
                                            encoder_last_stage=True, token_mlp=token_mlp_mode,
                                            is_last=False)
        self.sem_decoder_2 = MyDecoderLayer(d_base_feat_size * 2, in_out_chan[2], heads[2],
                                            split_size[2], sem_layers[2], n_class=sem_num_classes,
                                            encoder_last_stage=False, token_mlp=token_mlp_mode,
                                            is_last=False
                                            )
        self.sem_decoder_1 = MyDecoderLayer(d_base_feat_size * 4, in_out_chan[1], heads[1],
                                            split_size[1], sem_layers[1], n_class=sem_num_classes,
                                            encoder_last_stage=False, token_mlp=token_mlp_mode,
                                            is_last=False
                                            )
        self.sem_decoder_0 = MyDecoderLayer(d_base_feat_size * 8, in_out_chan[0], heads[0],
                                            split_size[0], sem_layers[0], n_class=sem_num_classes,
                                            encoder_last_stage=False, token_mlp=token_mlp_mode,
                                            is_last=True)

        self.inst_decoder_3 = MyDecoderLayer(d_base_feat_size, in_out_chan[3], heads[3],
                                             split_size[3], inst_layers[3], n_class=inst_num_classes,
                                             encoder_last_stage=True, token_mlp=token_mlp_mode,
                                             is_last=False)
        self.inst_decoder_2 = MyDecoderLayer(d_base_feat_size * 2, in_out_chan[2], heads[2],
                                             split_size[2], inst_layers[2], n_class=inst_num_classes,
                                             encoder_last_stage=False, token_mlp=token_mlp_mode,
                                             is_last=False
                                             )
        self.inst_decoder_1 = MyDecoderLayer(d_base_feat_size * 4, in_out_chan[1], heads[1],
                                             split_size[1], inst_layers[1], n_class=inst_num_classes,
                                             encoder_last_stage=False, token_mlp=token_mlp_mode,
                                             is_last=False
                                             )
        self.inst_decoder_0 = MyDecoderLayer(d_base_feat_size * 8, in_out_chan[0], heads[0],
                                             split_size[0], inst_layers[0], n_class=inst_num_classes,
                                             encoder_last_stage=False, token_mlp=token_mlp_mode,
                                             is_last=True)

        # self.cross_3 = CrossAttentionBlock(in_out_chan[3][1] // 2, d_base_feat_size * 2, heads[2],
        #                                    window_size=8, token_mlp=token_mlp_mode)
        # self.cross_2 = CrossAttentionBlock(in_out_chan[2][1] // 2, d_base_feat_size * 4, heads[1],
        #                                    window_size=8, token_mlp=token_mlp_mode)
        # self.cross_1 = CrossAttentionBlock(in_out_chan[1][1] // 2, d_base_feat_size * 8, heads[0],
        #                                    window_size=8, token_mlp=token_mlp_mode)

    def forward(self, encoder):
        b, c, _, _ = encoder[3].shape
        # ---------------Decoder-------------------------
        # print("stage3-----")
        sem_tmp_3 = self.sem_decoder_3(encoder[3].permute(0, 2, 3, 1).contiguous().reshape(b, -1, c))
        inst_tmp_3 = self.inst_decoder_3(encoder[3].permute(0, 2, 3, 1).contiguous().reshape(b, -1, c))
        # sem_tmp_3, inst_tmp_3 = self.cross_3(sem_tmp_3, inst_tmp_3)

        # print("stage2-----")
        sem_tmp_2 = self.sem_decoder_2(sem_tmp_3, encoder[2].permute(0, 2, 3, 1).contiguous())
        inst_tmp_2 = self.inst_decoder_2(inst_tmp_3, encoder[2].permute(0, 2, 3, 1).contiguous())
        # sem_tmp_2, inst_tmp_2 = self.cross_2(sem_tmp_2, inst_tmp_2)

        # print("stage1-----")
        sem_tmp_1 = self.sem_decoder_1(sem_tmp_2, encoder[1].permute(0, 2, 3, 1).contiguous())
        inst_tmp_1 = self.inst_decoder_1(inst_tmp_2, encoder[1].permute(0, 2, 3, 1).contiguous())
        # sem_tmp_1, inst_tmp_1 = self.cross_1(sem_tmp_1, inst_tmp_1)

        # print("stage0-----")
        sem_tmp_0 = self.sem_decoder_0(sem_tmp_1, encoder[0].permute(0, 2, 3, 1).contiguous())
        inst_tmp_0 = self.inst_decoder_0(inst_tmp_1, encoder[0].permute(0, 2, 3, 1).contiguous())
        # print(sem_tmp_0.shape, inst_tmp_0.shape)
        return sem_tmp_0, inst_tmp_0


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, resolution, heads, window_size, token_mlp='mix'):
        super(CrossAttentionBlock, self).__init__()
        self.dim = dim
        self.resolution = resolution
        self.heads = heads
        self.window_size = window_size

        self.sem_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.inst_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.sem_norm1 = nn.LayerNorm(dim)
        self.inst_norm1 = nn.LayerNorm(dim)

        self.sem_proj = nn.Linear(dim, dim)
        self.inst_proj = nn.Linear(dim, dim)
        # print('aa', dim, heads)
        self.sem_attn_local = LocalAttention(dim, resolution, heads,
                                             (window_size, window_size))
        self.inst_attn_local = LocalAttention(dim, resolution, heads,
                                              (window_size, window_size))
        self.sem_norm2 = nn.LayerNorm(dim)
        self.inst_norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.sem_mlp = MixFFN(dim, int(dim * 4))
            self.inst_mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.sem_mlp = MixFFN_skip(dim, int(dim * 4))
            self.inst_mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.sem_mlp = MLP_FFN(dim, int(dim * 4))
            self.inst_mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, sem_x, inst_x):
        H = W = self.resolution
        B, L, C = sem_x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        sem_img = self.sem_norm1(sem_x)
        inst_img = self.inst_norm1(inst_x)

        sem_qkv = self.sem_qkv(sem_img).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()
        inst_qkv = self.inst_qkv(inst_img).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()

        sem_cross_attn = self.sem_attn_local([sem_qkv[0], inst_qkv[1], inst_qkv[2]])
        inst_cross_attn = self.inst_attn_local([inst_qkv[0], sem_qkv[1], sem_qkv[2]])

        sem_attn = self.sem_proj(sem_cross_attn)
        inst_attn = self.inst_proj(inst_cross_attn)

        sem_x = sem_x + sem_attn
        inst_x = inst_x + inst_attn

        sem_x = sem_x + (self.sem_mlp(self.sem_norm2(sem_x), H, W))
        inst_x = inst_x + (self.inst_mlp(self.inst_norm2(inst_x), H, W))
        # print('a', x.shape)
        return sem_x, inst_x


class xjBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, img_size, patch_size, stride, padding, in_ch, input_resolution, depth, num_heads, split_size,
                 last_stage=False, token_mlp='mix', downsample=None, norm= None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DualTransformerBlock(dim=dim, resolution=input_resolution,
                                 heads=num_heads, split_size=split_size,
                                 encoder_last_stage=last_stage, token_mlp= token_mlp
                                 )
            for _ in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(img_size=img_size, patch_size=patch_size, stride=stride, padding=padding,
                                         in_ch=in_ch, dim=dim)
        else:
            self.downsample = None

        if norm is not None:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = None

    def forward(self, x):
        # print("xx", x.shape)
        B = x.shape[0]
        if self.downsample is not None:
            x, H, W = self.downsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # print("xxx", x.shape)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print("xxxxx", x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class xjCross_BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, shared_ratio=0.5):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks

        block = []
        for i in range(depth):
            swinblock = Cross_SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                   num_heads=num_heads, qkv=nn.Linear(dim, dim * 3, bias=qkv_bias),
                                                   window_size=window_size,
                                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                   mlp_ratio=mlp_ratio,
                                                   qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop,
                                                   drop_path=drop_path[i] if isinstance(drop_path,
                                                                                        list) else drop_path,
                                                   norm_layer=norm_layer,
                                                   )
            block.append(swinblock)
        self.blocks = nn.ModuleList(block)

        # self.blocks = nn.ModuleList([
        #     SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
        #                          num_heads=num_heads, window_size=window_size,
        #                          shift_size=0 if (i % 2 == 0) else window_size // 2,
        #                          mlp_ratio=mlp_ratio,
        #                          qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                          drop=drop, attn_drop=attn_drop,
        #                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #                          norm_layer=norm_layer)
        #     for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, list_cross_q=None):
        list_out_q = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                if list_cross_q is not None:
                    x, out_q = blk(x, cross_q=list_cross_q[i])
                else:
                    x, out_q = blk(x)
                list_out_q.append(out_q)
        if self.upsample is not None:
            x = self.upsample(x)
        return x, list_out_q

