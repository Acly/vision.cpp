import itertools
import torch
import numpy as np
import math
import pytest
from torch import Tensor

from . import workbench
from .workbench import to_nhwc, to_nchw, convert_to_nhwc, fuse_conv_2d_batch_norm

torch.set_printoptions(precision=2, linewidth=100, sci_mode=False)

nhwc_layout = dict(memory_layout="nhwc")

#
# Image Encoder
#


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, bias=False
    ):
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


def fuse_all_conv_2d_batch_norm(model: dict[str, Tensor]):
    fused_weights = {}
    for k in model:
        if not fuse_conv_2d_batch_norm(model, k, "", "c", "bn", fused_weights):
            fused_weights[k] = model[k]
    return fused_weights


@pytest.mark.parametrize("bias", [False, True])
def test_conv_2d_batch_norm(bias: bool):
    conv2dbn = Conv2d_BN(4, 6, ks=3, stride=2, pad=1, bias=bias)
    state = workbench.randomize(conv2dbn.state_dict())
    conv2dbn.load_state_dict(state)
    conv2dbn.eval()

    x = workbench.input_tensor(1, 4, 8, 8)
    expected = conv2dbn(x)

    state = fuse_all_conv_2d_batch_norm(state)
    state = convert_to_nhwc(state)
    x = to_nhwc(x)
    result = workbench.invoke_test("sam_conv_2d_batch_norm", x, state, nhwc_layout)
    result = to_nchw(result)

    assert torch.allclose(result, expected)


class PatchEmbed(torch.nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: tuple[int, int] = (
            resolution
            if isinstance(resolution, tuple)
            else (
                resolution,
                resolution,
            )
        )
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = torch.nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


def test_patch_embed():
    patch_embed = PatchEmbed(3, 4, (8, 8), torch.nn.GELU)
    state = workbench.randomize(patch_embed.state_dict())
    patch_embed.load_state_dict(state)
    patch_embed.eval()

    x = torch.rand(1, 3, 8, 8)
    expected = patch_embed(x)

    state = fuse_all_conv_2d_batch_norm(state)
    convert_to_nhwc(state)
    x = to_nhwc(x)
    result = to_nhwc(torch.zeros_like(expected))
    result = workbench.invoke_test("sam_patch_embed", x, state, nhwc_layout)
    result = to_nchw(result)

    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def test_layer_norm_2d():
    layer_norm = LayerNorm2d(4)
    state = workbench.randomize(layer_norm.state_dict())
    layer_norm.load_state_dict(state)
    layer_norm.eval()

    x = torch.rand(1, 4, 8, 8)
    expected = layer_norm(x)

    x = to_nhwc(x)
    result = to_nhwc(torch.zeros_like(expected))
    result = workbench.invoke_test("layer_norm", x, state, nhwc_layout)
    result = to_nchw(result)

    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class MBConv(torch.nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation(approximate="tanh")

        self.conv2 = Conv2d_BN(
            self.hidden_chans,
            self.hidden_chans,
            ks=3,
            stride=1,
            pad=1,
            groups=self.hidden_chans,
        )
        self.act2 = activation(approximate="tanh")

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation(approximate="tanh")

        self.drop_path = torch.nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


def test_mb_conv():
    mb_conv = MBConv(4, 4, 4.0, torch.nn.GELU, 0.2)
    state = workbench.randomize(mb_conv.state_dict())
    mb_conv.load_state_dict(state)
    mb_conv.eval()

    x = torch.rand(1, 4, 8, 8)
    expected = mb_conv(x)

    state = fuse_all_conv_2d_batch_norm(state)
    convert_to_nhwc(state)
    x = to_nhwc(x)
    result = workbench.invoke_test("sam_mb_conv", x, state, nhwc_layout)
    result = to_nchw(result)

    # precision: ggml_gelu uses fp16 look-up table & tanh approximation
    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class PatchMerging(torch.nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim == 320 or out_dim == 448 or out_dim == 576:
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def test_patch_merging():
    patch_merging = PatchMerging((32, 32), 8, 16, torch.nn.GELU)
    state = workbench.randomize(patch_merging.state_dict())
    patch_merging.load_state_dict(state)
    patch_merging.eval()

    x = torch.rand(1, 8, 32, 32)
    expected = patch_merging(x)

    state = fuse_all_conv_2d_batch_norm(state)
    convert_to_nhwc(state)
    x = to_nhwc(x)
    result = workbench.invoke_test("sam_patch_merging", x, state, nhwc_layout)
    result = result.transpose(1, 2).reshape_as(expected)

    # precision: ggml_gelu uses fp16 look-up table & tanh approximation
    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class Mlp(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = torch.nn.LayerNorm(in_features)
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def test_mlp():
    mlp = Mlp(4, 16)
    state = workbench.randomize(mlp.state_dict())
    mlp.load_state_dict(state)
    mlp.eval()

    x = torch.rand(1, 6, 4)
    expected = mlp(x)

    result = workbench.invoke_test("sam_mlp", x, state)

    # precision: ggml_gelu uses fp16 look-up table & tanh approximation
    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class AttentionRelBias(torch.nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
    ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, h)
        self.proj = torch.nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.register_buffer(
                "ab",
                self.attention_biases[:, self.attention_bias_idxs],
                persistent=False,
            )

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        qkv = qkv.view(B, N, self.num_heads, -1)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        attn = attn + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, self.dh)
        x = self.proj(x)
        return x


def test_attention_rel_bias():
    attention = AttentionRelBias(4, 2, num_heads=2, attn_ratio=1, resolution=(3, 3))
    state = workbench.randomize(attention.state_dict())
    attention.load_state_dict(state)
    attention.eval()

    x = torch.rand(4, 9, 4)
    expected = attention(x)

    state["attention_biases_indexed"] = state["attention_biases"][:, attention.attention_bias_idxs]
    result = workbench.invoke_test("sam_attention_rel_bias", x, state)

    assert torch.allclose(result, expected, atol=0.001)


class TinyViTBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=torch.nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = torch.nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = AttentionRelBias(
            dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_activation,
            drop=drop,
        )

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = (
                x.view(B, nH, self.window_size, nW, self.window_size, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_size * self.window_size, C)
            )

            x = self.attn(x)

            # window reverse
            x = (
                x.view(B, nH, nW, self.window_size, self.window_size, C)
                .transpose(2, 3)
                .reshape(B, pH, pW, C)
            )

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x


def test_tiny_vit_block():
    tiny_vit_block = TinyViTBlock(4, (8, 8), num_heads=2, window_size=5)
    state = workbench.randomize(tiny_vit_block.state_dict())
    tiny_vit_block.load_state_dict(state)
    tiny_vit_block.eval()

    x = torch.rand(1, 64, 4)
    expected = tiny_vit_block(x)

    state["attn.attention_biases_indexed"] = state["attn.attention_biases"][
        :, tiny_vit_block.attn.attention_bias_idxs
    ]
    state = fuse_all_conv_2d_batch_norm(state)
    state = convert_to_nhwc(state)
    result = workbench.invoke_test("sam_tiny_vit_block", x, state, nhwc_layout)

    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class ConvLayer(torch.nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = torch.nn.ModuleList([
            MBConv(
                dim,
                dim,
                conv_expand_ratio,
                activation,
                drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer(torch.nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=torch.nn.GELU,
        out_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = torch.nn.ModuleList([
            TinyViTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                local_conv_size=local_conv_size,
                activation=activation,
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(torch.nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = torch.nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=activation,
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                #                     patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # Classifier head
        self.norm_head = torch.nn.LayerNorm(embed_dims[-1])
        self.head = (
            torch.nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        )

        # init weights
        # self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            torch.nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        # print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"attention_biases"}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        B, _, C = x.size()
        x = x.view(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.norm_head(x)
        # x = self.head(x)
        return x


def test_tiny_vit():
    tiny_vit = TinyViT(
        img_size=1024,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
    )
    state = workbench.randomize(tiny_vit.state_dict())
    tiny_vit.load_state_dict(state)
    tiny_vit.eval()

    x = torch.rand(1, 3, 1024, 1024)
    expected = tiny_vit(x)

    # !! running out of memory when using default resolution on workbench

    # add_variance_epsilon(state)
    # result = torch.zeros_like(expected).contiguous()
    # result = workbench.invoke_test("sam_tiny_vit", x, state)

    # assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


#
# Prompt Encoder
#


class PositionEmbeddingRandom(torch.nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale=None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


def test_position_embedding_random():
    pos_enc = PositionEmbeddingRandom(4)
    state = workbench.randomize(pos_enc.state_dict())
    pos_enc.load_state_dict(state)
    pos_enc.eval()

    x = torch.tensor([[[63.5, 55.5], [32.5, 0.5], [0.0, 0.0]]])
    expected = pos_enc.forward_with_coords(x, (64, 64))

    result = workbench.invoke_test("sam_position_embedding_random", x, state)

    assert torch.allclose(result, expected)


class PromptEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation=torch.nn.GELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            torch.nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = torch.nn.ModuleList(point_embeddings)
        self.not_a_point_embed = torch.nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = torch.nn.Sequential(
            torch.nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            torch.nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            torch.nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = torch.nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool):
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        bs = 1  # batch size
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


def test_prompt_encoder_points():
    prompt_encoder = PromptEncoder(
        embed_dim=4,
        image_embedding_size=(8, 8),
        input_image_size=(64, 64),
        mask_in_chans=4,
    )
    state = workbench.randomize(prompt_encoder.state_dict())
    prompt_encoder.load_state_dict(state)
    prompt_encoder.eval()

    points = torch.tensor([[[63.0, 55.0], [32.0, 0.0]]])
    labels = torch.tensor([[1, 1]])
    expected, expected_dense = prompt_encoder(points=(points, labels), boxes=None, masks=None)

    points = torch.cat([points, -torch.ones(1, 1, 2)], dim=1)
    result = workbench.invoke_test("sam_embed_points", points, state)

    assert torch.allclose(result, expected)


def test_prompt_encoder_box():
    prompt_encoder = PromptEncoder(
        embed_dim=4,
        image_embedding_size=(8, 8),
        input_image_size=(64, 64),
        mask_in_chans=4,
    )
    state = workbench.randomize(prompt_encoder.state_dict())
    prompt_encoder.load_state_dict(state)
    prompt_encoder.eval()

    boxes = torch.tensor([[[63.0, 55.0], [32.0, 0.0]]])
    expected, _ = prompt_encoder(points=None, boxes=boxes, masks=None)

    result = workbench.invoke_test("sam_embed_box", boxes, state)

    assert torch.allclose(result, expected)


#
# Mask Decoder
#


class Attention(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = torch.nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


def test_attention():
    attention = Attention(4, 2, downsample_rate=2)
    state = workbench.randomize(attention.state_dict())
    attention.load_state_dict(state)
    attention.eval()

    q = torch.rand(1, 8, 4)
    k = torch.rand(1, 8, 4)
    v = torch.rand(1, 8, 4)
    expected = attention(q, k, v)

    state["input_k"] = k
    state["input_v"] = v
    result = workbench.invoke_test("sam_attention", q, state)

    assert torch.allclose(result, expected)


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act=torch.nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = torch.nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation=torch.nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = torch.nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = torch.nn.LayerNorm(embedding_dim)

        self.norm4 = torch.nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


@pytest.mark.parametrize("mode", ["skip_first_layer_pe", "default"])
def test_two_way_attention_block(mode):
    torch.manual_seed(330896961738400)

    two_way_attention = TwoWayAttentionBlock(
        4, num_heads=2, mlp_dim=16, skip_first_layer_pe=(mode == "skip_first_layer_pe")
    )
    state = workbench.randomize(two_way_attention.state_dict())
    two_way_attention.load_state_dict(state)
    two_way_attention.eval()

    queries = torch.rand(1, 8, 4)
    keys = torch.rand(1, 8, 4)
    query_pe = torch.rand(1, 8, 4)
    key_pe = torch.rand(1, 8, 4)
    expected_queries, expected_keys = two_way_attention(queries, keys, query_pe, key_pe)

    state = {
        k.replace("token_to_image", "t2i").replace("image_to_token", "i2t"): v
        for k, v in state.items()
    }
    state["input_keys"] = keys
    state["input_query_pe"] = query_pe
    state["input_key_pe"] = key_pe
    state["result_keys"] = torch.zeros_like(expected_keys).contiguous()
    result_queries, result_keys = workbench.invoke_test(
        "sam_two_way_attention_block", queries, state, {"mode": mode}
    )

    assert torch.allclose(result_queries, expected_queries)
    assert torch.allclose(result_keys, expected_keys)


class TwoWayTransformer(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation=torch.nn.ReLU,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = torch.nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = torch.nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ):
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


def test_two_way_transformer():
    two_way_transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=4,
        num_heads=2,
        mlp_dim=16,
    )
    state = workbench.randomize(two_way_transformer.state_dict())
    two_way_transformer.load_state_dict(state)
    two_way_transformer.eval()

    image_embedding = torch.rand(1, 4, 8, 8)
    image_pe = torch.rand(1, 4, 8, 8)
    point_embedding = torch.rand(1, 8, 4)
    expected_queries, expected_keys = two_way_transformer(
        image_embedding, image_pe, point_embedding
    )

    state = {
        k.replace("token_to_image", "t2i").replace("image_to_token", "i2t"): v
        for k, v in state.items()
    }
    # state["input_image_embedding"] = image_embedding
    image_embedding = to_nhwc(image_embedding)
    state["input_image_pe"] = to_nhwc(image_pe)
    state["input_point_embedding"] = point_embedding
    result_queries, result_keys = workbench.invoke_test(
        "sam_two_way_transformer", image_embedding, state, nhwc_layout
    )

    assert torch.allclose(result_queries, expected_queries, atol=1e-6, rtol=1e-4)
    assert torch.allclose(result_keys, expected_keys, atol=1e-6, rtol=1e-4)


class HypernetworkMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.nn.functional.sigmoid(x)
        return x


def test_hypernetwork_mlp():
    hypernetwork_mlp = HypernetworkMLP(4, 8, 8, num_layers=2)
    state = workbench.randomize(hypernetwork_mlp.state_dict())
    hypernetwork_mlp.load_state_dict(state)
    hypernetwork_mlp.eval()

    x = torch.rand(1, 4)
    expected = hypernetwork_mlp(x)

    result = workbench.invoke_test("sam_hypernetwork_mlp", x, state)

    assert torch.allclose(result, expected)


def output_upscaling(transformer_dim: int, activation=torch.nn.GELU):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        LayerNorm2d(transformer_dim // 4),
        activation(),
        torch.nn.ConvTranspose2d(
            transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
        ),
        activation(),
    )


def test_output_upscaling():
    upscaling = output_upscaling(transformer_dim=16)
    state = workbench.randomize(upscaling.state_dict())
    upscaling.load_state_dict(state)
    upscaling.eval()

    x = torch.rand(1, 16, 64, 64)
    expected = upscaling(x)

    x = to_nhwc(x)
    result = workbench.invoke_test("sam_output_upscaling", x, state, nhwc_layout, backend="vulkan")
    result = to_nchw(result)

    workbench.print_results(result, expected)
    assert torch.allclose(result, expected, rtol=0.1)  # fp16 weights


class MaskDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: torch.nn.Module,
        num_multimask_outputs: int = 3,
        activation=torch.nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = torch.nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = torch.nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = output_upscaling(transformer_dim, activation)
        self.output_hypernetworks_mlps = torch.nn.ModuleList([
            HypernetworkMLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = HypernetworkMLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ):
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ):
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = hyper_in @ upscaled_embedding.view(b, c, h * w)
        masks = masks.view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


def test_predict_masks():
    w = 8
    h = 8
    dim = 32
    decoder = MaskDecoder(
        transformer_dim=dim,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=dim,
            num_heads=2,
            mlp_dim=dim,
        ),
        num_multimask_outputs=3,
    )
    state = workbench.randomize(decoder.state_dict())
    decoder.load_state_dict(state)
    decoder.eval()

    image_embeddings = torch.rand(1, dim, h, w)
    image_pe = torch.rand(1, dim, h, w)
    sparse_prompt_embeddings = torch.rand(1, 8, dim)
    dense_prompt_embeddings = torch.rand(1, dim, h, w)
    expected_masks, iou_pred = decoder.predict_masks(
        image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings
    )

    state = {
        k.replace("token_to_image", "t2i").replace("image_to_token", "i2t"): v
        for k, v in state.items()
    }
    image_embeddings = to_nhwc(image_embeddings)
    state["dense_positional_embedding"] = to_nhwc(image_pe)
    state["input_sparse_prompt"] = sparse_prompt_embeddings
    state["input_dense_prompt"] = to_nhwc(dense_prompt_embeddings)
    result_masks = torch.zeros_like(expected_masks).contiguous()
    result_masks, result_iou_pred = workbench.invoke_test(
        "sam_predict_masks", image_embeddings, state, nhwc_layout, backend="vulkan"
    )

    assert torch.allclose(result_masks, expected_masks, rtol=1e-2, atol=1e-2)
    assert torch.allclose(result_iou_pred, iou_pred, rtol=1e-2)
