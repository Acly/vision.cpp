from pathlib import Path
from typing import NamedTuple
import torch
import torchvision
import math
import pytest
from torch import Tensor, nn
from torch.nn import functional as F
from timm.layers import to_2tuple, trunc_normal_
from einops import rearrange

from . import workbench
from .workbench import fuse_batch_norm, to_nhwc, to_nchw, convert_to_nhwc, fuse_conv_2d_batch_norm
from .workbench import input_tensor, generate_state

torch.set_printoptions(precision=3, linewidth=100, edgeitems=6, sci_mode=False)

nhwc_layout = dict(memory_layout="nhwc")


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def test_relative_position_index():
    window_attention = WindowAttention(dim=8, window_size=(3, 3), num_heads=2)
    expected = window_attention.relative_position_index

    result = torch.zeros_like(expected, dtype=torch.int32)
    result = workbench.invoke_test("biref_relative_position_index", result, {})
    result = result.to(expected.dtype)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("masking", ["mask", "no_mask"])
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("attn", ["default", "flash_attn"])
def test_window_attention(masking: bool, backend: str, attn: str):
    num_heads = 2
    window_attention = WindowAttention(dim=8, window_size=(3, 3), num_heads=num_heads)
    state = generate_state(window_attention.state_dict())
    window_attention.load_state_dict(state)
    window_attention.eval()

    x = input_tensor(2, 9, 8)
    mask = None
    if masking == "mask":
        mask = torch.zeros(2, 9, 9).masked_fill(torch.rand(2, 9, 9) > 0.5, -100.0)
        state["mask"] = mask
    expected = window_attention(x, mask)

    del state["relative_position_index"]  # computed in C++
    if mask is not None:
        state["mask"] = mask.half()
        state["relative_position_bias_table"] = state["relative_position_bias_table"].half()
    result = workbench.invoke_test("biref_window_attention", x, state, {"attn": attn}, backend)

    assert torch.allclose(result, expected, rtol=1e-3)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.H: int | None = None
        self.W: int | None = None

    def forward(self, x, mask_matrix):
        assert self.W is not None and self.H is not None, "W and H must be set before forward"
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x  # + self.drop_path(x)
        x = x + self.mlp(self.norm2(x))  # self.drop_path(self.mlp(self.norm2(x)))

        return x


def test_swin_block():
    num_heads = 2
    swin_block = SwinTransformerBlock(8, num_heads, window_size=3, shift_size=0)
    state = generate_state(swin_block.state_dict())
    swin_block.load_state_dict(state)
    swin_block.eval()

    x = input_tensor(1, 36, 8)
    mask = torch.zeros(2, 9, 9).masked_fill(torch.rand(2, 9, 9) > 0.5, -100.0)
    state["mask"] = mask.half()
    swin_block.W, swin_block.H = 6, 6
    expected = swin_block(x, None)

    result = workbench.invoke_test("biref_swin_block", x, state)

    assert torch.allclose(result, expected, atol=1e-2)  # fp16 GELU


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


def test_patch_merging():
    patch_merging = PatchMerging(8)
    state = generate_state(patch_merging.state_dict())
    patch_merging.load_state_dict(state)
    patch_merging.eval()

    x = input_tensor(1, 4 * 6, 8)
    expected = patch_merging(x, 4, 6)

    result = workbench.invoke_test("biref_patch_merging", x, state)

    assert torch.allclose(result, expected)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def attention_mask(self, H, W):
        Hp = math.ceil(H / self.window_size) * self.window_size
        Wp = math.ceil(W / self.window_size) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # calculate attention mask for SW-MSA
        # Turn int to torch.tensor for the compatiability with torch.compile in PyTorch 2.5.
        Hp = torch.ceil(torch.tensor(H) / self.window_size).to(torch.int64) * self.window_size
        Wp = torch.ceil(torch.tensor(W) / self.window_size).to(torch.int64) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = (
            attn_mask.masked_fill(attn_mask != 0, float("-inf"))
            .masked_fill(attn_mask == 0, float(0.0))
            .to(x.dtype)
        )

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


def test_attention_mask():
    window_size = 6
    w, h = 18, 18
    swin_layer = BasicLayer(8, 2, 2, window_size=window_size)
    expected = swin_layer.attention_mask(h, w)

    x = torch.zeros_like(expected)
    result = workbench.invoke_test("biref_attention_mask", x, {})

    assert torch.allclose(result, expected)


def test_swin_layer():
    num_heads = 2
    depth = 2
    w, h = 6, 6
    win = 3
    swin_layer = BasicLayer(8, depth, num_heads, window_size=win, downsample=PatchMerging)
    state = generate_state(swin_layer.state_dict())
    swin_layer.load_state_dict(state)
    swin_layer.eval()

    ww = math.ceil(w / win) * win
    wh = math.ceil(h / win) * win
    attn_mask = swin_layer.attention_mask(ww, wh)
    state[f"swin_layer_{w}x{h}.attn_mask"] = attn_mask

    x = input_tensor(1, w * h, 8)
    out, out_h, out_w, expected, down_h, down_w = swin_layer(x, 6, 6)
    assert down_h == 3 and down_w == 3

    result = workbench.invoke_test("biref_swin_layer", x, state)

    assert torch.allclose(result, expected)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


def test_patch_embed():
    patch_embed = PatchEmbed(patch_size=4, in_channels=3, embed_dim=8, norm_layer=nn.LayerNorm)
    state = generate_state(patch_embed.state_dict())
    patch_embed.load_state_dict(state)
    patch_embed.eval()

    x = input_tensor(1, 3, 8, 12)
    expected = patch_embed(x)

    state = convert_to_nhwc(state, key="proj")
    x = to_nhwc(x)
    result = workbench.invoke_test("biref_patch_embed", x, state)
    result = to_nchw(result)

    assert torch.allclose(result, expected)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = x + absolute_pos_embed  # B Wh*Ww C

        outs = []  # x.contiguous()]
        x = x.flatten(2).transpose(1, 2)
        # x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


def test_swin_transformer():
    w, h = 32, 32
    swin_transformer = SwinTransformer(
        embed_dim=8, depths=[2, 2, 2, 2], num_heads=[2, 2, 4, 2], window_size=3
    )
    state = workbench.randomize(swin_transformer.state_dict())
    swin_transformer.load_state_dict(state)
    swin_transformer.eval()

    x = torch.rand(1, 3, w, h)
    expected = swin_transformer(x)

    x = to_nhwc(x)
    state = convert_to_nhwc(state, key="patch_embed.proj")

    result = workbench.invoke_test("biref_swin_transformer", x[0], state)

    for i, e in enumerate(expected):
        result[i] = to_nchw(result[i])
        assert torch.allclose(result[i], e, atol=0.002)


def _interpolate(x, target_size):
    return F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)


def _downscale(x, target_size):
    return F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)


def forward_enc(x: Tensor, xs: list[Tensor], xs_low: list[Tensor]):
    # x1, x2, x3, x4 = self.bb(x)
    x1, x2, x3, x4 = xs
    # cat
    B, C, H, W = x.shape
    # x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
    x1_, x2_, x3_, x4_ = xs_low
    x1 = torch.cat([x1, _interpolate(x1_, x1.shape[2:])], dim=1)
    x2 = torch.cat([x2, _interpolate(x2_, x2.shape[2:])], dim=1)
    x3 = torch.cat([x3, _interpolate(x3_, x3.shape[2:])], dim=1)
    x4 = torch.cat([x4, _interpolate(x4_, x4.shape[2:])], dim=1)

    to_concatenate = (
        *[
            _downscale(x1, x4.shape[2:]),
            _downscale(x2, x4.shape[2:]),
            _downscale(x3, x4.shape[2:]),
        ][-3:],
        x4,
    )
    x4 = torch.cat(to_concatenate, dim=1)
    return (x1, x2, x3, x4)


def test_encode():
    x = input_tensor(1, 3, 32, 32)
    xs = [
        input_tensor(1, 4, 16, 16),
        input_tensor(1, 8, 8, 8),
        input_tensor(1, 16, 4, 4),
        input_tensor(1, 32, 2, 2),
    ]
    xs_low = [
        input_tensor(1, 4, 8, 8),
        input_tensor(1, 8, 4, 4),
        input_tensor(1, 16, 2, 2),
        input_tensor(1, 32, 1, 1),
    ]
    expected = forward_enc(x, xs, xs_low)

    state = {}
    state.update({f"xs{i}": to_nhwc(xs[i]) for i in range(4)})
    state.update({f"xs_low{i}": to_nhwc(xs_low[i]) for i in range(4)})

    results = workbench.invoke_test("biref_encode", x, state, nhwc_layout)

    for i, e in enumerate(expected):
        result = to_nchw(results[i])
        assert torch.allclose(result, e)


#
#
# Decoder
#


@pytest.mark.parametrize("scenario", ["small", "large"])
@pytest.mark.parametrize("memory_layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_conv_2d_deform(scenario: str, memory_layout: str, backend: str):
    torch.manual_seed(42)
    if memory_layout == "nhwc" and backend == "vulkan":
        pytest.skip("conv_2d_deform with nhwc layout is not supported on Vulkan")

    w, h, c_in, c_out, k = {
        "small": (4, 4, 5, 2, 3),
        "large": (49, 38, 81, 17, 7),
    }[scenario]
    x = torch.rand(1, c_in, h, w) - 0.5
    weight = torch.rand(c_out, c_in, k, k) - 0.5
    offset = 1.0 - input_tensor(1, 2 * k * k, h, w)
    mask = torch.rand(1, k * k, h, w)
    expected = torchvision.ops.deform_conv2d(x, offset, weight, mask=mask, padding=(k // 2, k // 2))

    state = {
        "weight": weight,
        "offset": offset,
        "mask": mask,
    }
    if memory_layout == "nhwc":
        x = to_nhwc(x)
        state = {k: to_nhwc(v) for k, v in state.items()}
    params = dict(memory_layout=memory_layout, padding=k // 2)
    result = workbench.invoke_test("conv_2d_deform", x, state, params, backend=backend)
    if memory_layout == "nhwc":
        result = to_nchw(result)

    assert torch.allclose(result, expected, atol=0.1 if backend == "vulkan" else 0.001)


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) is tuple or type(kernel_size) is int

        kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) is tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


def shorten_weight_name(name: str):
    return (
        name.replace("offset_conv", "offset")
        .replace("modulator_conv", "modulator")
        .replace("regular_conv", "conv")
        .replace("atrous_conv", "conv")
        .replace("decoder_block", "block")
    )


def test_deformable_conv_2d():
    conv = DeformableConv2d(3, 2, kernel_size=3, stride=1, padding=1)
    state = generate_state(conv.state_dict())
    conv.load_state_dict(state)
    conv.eval()
    x = input_tensor(1, 3, 4, 4)
    expected = conv(x)

    state = convert_to_nhwc(state, key="conv")
    state = {shorten_weight_name(k): v for k, v in state.items()}
    x = to_nhwc(x)
    result = workbench.invoke_test("biref_deformable_conv_2d", x, state, nhwc_layout)
    result = to_nchw(result)

    assert torch.allclose(result, expected)


class GlobalAvgPool(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.append(nn.Conv2d(in_channels, in_channels, 1, stride=1, bias=False))
        self.append(nn.BatchNorm2d(in_channels))
        self.append(nn.ReLU(inplace=True))


def fuse_all_conv_2d_batch_norm(
    model: dict[str, torch.Tensor], key_module: str, key_conv: str, key_bn: str
):
    fused_weights = {}
    for k in model:
        if not fuse_conv_2d_batch_norm(model, k, key_module, key_conv, key_bn, fused_weights):
            fused_weights[k] = model[k]
    return fused_weights


def fuse_all_batch_norm(state: dict[str, torch.Tensor], key_bn: str):
    out = {}
    for k, v in state.items():
        v = fuse_batch_norm(state, k, key_bn)
        if v is not None and k != "num_batches_tracked":
            out[k] = v
    return out


@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_global_avg_pool(backend: str):
    pool = GlobalAvgPool(3)
    state = generate_state(pool.state_dict())
    pool.load_state_dict(state)
    pool.eval()
    x = input_tensor(1, 3, 4, 4)
    expected = pool(x)

    state = fuse_all_conv_2d_batch_norm(state, "", "1", "2")
    state = convert_to_nhwc(state, key="1.weight")
    x = to_nhwc(x)
    result = workbench.invoke_test("biref_global_avg_pool", x, state, nhwc_layout, backend=backend)
    result = to_nchw(result)

    assert torch.allclose(result, expected)


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = DeformableConv2d(
            in_channels,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=[1, 3, 7]):
        super(ASPPDeformable, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(
                in_channels,
                self.in_channelster,
                conv_size,
                padding=int(conv_size // 2),
            )
            for conv_size in parallel_block_sizes
        ])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_channelster),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(
            self.in_channelster * (2 + len(self.aspp_deforms)),
            out_channels,
            1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


def test_aspp_deformable():
    aspp = ASPPDeformable(3, 2)
    state = workbench.randomize(aspp.state_dict())
    aspp.load_state_dict(state)
    aspp.eval()
    x = input_tensor(1, 3, 16, 16)
    expected = aspp(x)

    state = fuse_all_conv_2d_batch_norm(state, "", "conv1", "bn1")
    state = fuse_all_conv_2d_batch_norm(state, "global_avg_pool.", "1", "2")
    state = fuse_all_batch_norm(state, "bn")
    state = convert_to_nhwc(state, key="conv")
    state = convert_to_nhwc(state, key="pool.1")
    state = {shorten_weight_name(k): v for k, v in state.items()}
    x = to_nhwc(x)

    result = workbench.invoke_test("biref_aspp_deformable", x, state, nhwc_layout)
    result = to_nchw(result)

    assert torch.allclose(result, expected)


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicDecBlk, self).__init__()
        # inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        if hasattr(self, "dec_att"):
            x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


def test_basic_dec_blk():
    block = BasicDecBlk(8, 5)
    state = workbench.randomize(block.state_dict())
    block.load_state_dict(state)
    block.eval()
    x = input_tensor(1, 8, 12, 16) - 0.5
    expected = block(x)

    state = fuse_all_conv_2d_batch_norm(state, "", "conv_in", "bn_in")
    state = fuse_all_conv_2d_batch_norm(state, "", "conv_out", "bn_out")
    state = fuse_all_conv_2d_batch_norm(state, "dec_att.", "conv1", "bn1")
    state = fuse_all_conv_2d_batch_norm(state, "global_avg_pool.", "1", "2")
    state = fuse_all_batch_norm(state, "bn")
    state = convert_to_nhwc(state, key="conv")
    state = convert_to_nhwc(state, key="pool.1")
    state = {shorten_weight_name(k): v for k, v in state.items()}
    x = to_nhwc(x)

    result = workbench.invoke_test("biref_basic_dec_blk", x, state, nhwc_layout)
    result = to_nchw(result)

    assert torch.allclose(result, expected)


def image2patches(
    image,
    grid_h=2,
    grid_w=2,
    patch_ref=None,
    transformation="b c (hg h) (wg w) -> (b hg wg) c h w",
):
    if patch_ref is not None:
        grid_h, grid_w = (
            image.shape[-2] // patch_ref.shape[-2],
            image.shape[-1] // patch_ref.shape[-1],
        )
    patches = rearrange(image, transformation, hg=grid_h, wg=grid_w)
    return patches


def test_image_to_patches():
    transformation = "b c (hg h) (wg w) -> b (c hg wg) h w"
    x = torch.arange(3 * 8 * 8).reshape(1, 3, 8, 8).float()
    expected = image2patches(x, 2, 2, None, transformation)

    result = workbench.invoke_test("biref_image_to_patches_2", x, {})

    assert torch.allclose(result, expected)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels=64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicLatBlk, self).__init__()
        # inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


class Config(NamedTuple):
    dec_ipt: bool = True
    dec_ipt_split: bool = True
    ms_supervision: bool = True
    out_ref: bool = True
    batch_size: int = 4


# fmt: off
class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        DecoderBlock = BasicDecBlk
        LateralBlock = BasicLatBlk

        if self.config.dec_ipt:
            self.split = self.config.dec_ipt_split
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            self.ipt_blk5 = DBlock(2**10*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk4 = DBlock(2**8*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk3 = DBlock(2**6*3 if self.split else 3, [N_dec_ipt, channels[1]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk2 = DBlock(2**4*3 if self.split else 3, [N_dec_ipt, channels[2]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk1 = DBlock(2**0*3 if self.split else 3, [N_dec_ipt, channels[3]//8][ipt_cha_opt], inter_channels=ic)
        else:
            self.split = None

        self.decoder_block4 = DecoderBlock(channels[0]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[1])
        self.decoder_block3 = DecoderBlock(channels[1]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[2])
        self.decoder_block2 = DecoderBlock(channels[2]+([N_dec_ipt, channels[1]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3])
        self.decoder_block1 = DecoderBlock(channels[3]+([N_dec_ipt, channels[2]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3]//2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3]//2+([N_dec_ipt, channels[3]//8][ipt_cha_opt] if self.config.dec_ipt else 0), 1, 1, 1, 0))

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

            if self.config.out_ref:
                _N = 16
                self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channels[3], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))

                self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                
                self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def forward(self, features):
        if self.training and self.config.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features
        outs = []

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=x4, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches_batch, size=x4.shape[2:], mode='bilinear', align_corners=True))), 1)
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p4_gdt = self.gdt_convs_4(p4)
            if self.training:
                # >> GT:
                m4_dia = m4
                gdt_label_main_4 = gdt_gt * F.interpolate(m4_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_4)
                # >> Pred:
                gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
                outs_gdt_pred.append(gdt_pred_4)
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            # >> Finally:
            p4 = p4 * gdt_attn_4
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        # return [_p3]

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p3, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), 1)
        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                m3_dia = m3
                gdt_label_main_3 = gdt_gt * F.interpolate(m3_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p2, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), 1)
        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                # >> GT:
                m2_dia = m2
                gdt_label_main_2 = gdt_gt * F.interpolate(m2_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)

        if self.config.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=True))), 1)
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision and self.training:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        return outs if not (self.config.out_ref and self.training) else ([outs_gdt_pred, outs_gdt_label], outs)
# fmt: on


def test_decoder():
    from safetensors.torch import load_file

    # Running the full decoder doesn't really work with randomized weights
    reference = "BiRefNet.safetensors"
    if not Path(reference).exists():
        pytest.skip(f"Reference file {reference} not found. Skipping test.")
    ref_state = load_file(reference)

    channels = [3072, 1536, 768, 384]
    decoder = Decoder(channels)
    state = decoder.state_dict()
    for k, v in state.items():
        key = f"decoder.{k}"
        if key in ref_state:
            state[k] = ref_state[key].float()
        else:
            print(f"Warning: {k} not found in decoder state_dict")
    decoder.load_state_dict(state)
    decoder.eval()

    x = torch.rand(1, 3, 1024, 1024)
    x1 = torch.rand(1, 384, 256, 256) * 2 - 1
    x2 = torch.rand(1, 768, 128, 128) * 2 - 1
    x3 = torch.rand(1, 1536, 64, 64) * 2 - 1
    x4 = torch.rand(1, 1024 * 3, 32, 32) * 2 - 1

    expected = decoder((x, x1, x2, x3, x4))[0]
    expected = expected.sigmoid()

    state = fuse_all_batch_norm(state, "bn")
    state = {shorten_weight_name(k): v for k, v in state.items()}
    state = convert_to_nhwc(state, key="conv")
    state = convert_to_nhwc(state, key="pool.1")

    x = to_nhwc(x)
    state["x1"] = to_nhwc(x1)
    state["x2"] = to_nhwc(x2)
    state["x3"] = to_nhwc(x3)
    state["x4"] = to_nhwc(x4)

    result = workbench.invoke_test("biref_decode", x, state, nhwc_layout)
    result = to_nchw(result)

    assert torch.allclose(result, expected)
