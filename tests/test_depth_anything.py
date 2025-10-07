import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tests import workbench
from tests.workbench import convert_to_nhwc, generate_state, input_tensor, to_nchw, to_nhwc


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor, flatten=False) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, (
            f"Input image height {H} is not a multiple of patch height {patch_H}"
        )
        assert W % patch_W == 0, (
            f"Input image width {W} is not a multiple of patch width: {patch_W}"
        )
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        # x = self.norm(x)
        if not flatten:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


def test_patch_embed():
    patch_embed = PatchEmbed(img_size=(16, 16), patch_size=(4, 4), in_chans=3, embed_dim=8)
    state = generate_state(patch_embed.state_dict())
    patch_embed.load_state_dict(state)
    patch_embed.eval()

    x = input_tensor(1, 3, 8, 12)
    expected = patch_embed(x)

    x = to_nhwc(x)
    state = convert_to_nhwc(state, key="proj")
    result = workbench.invoke_test("biref_patch_embed", x, state)

    assert torch.allclose(result, expected)


def interpolate_pos_encoding(pos_embed: Tensor, x: Tensor, w: int, h: int, patch_size: int):
    # This is 0.1 in official code, which would cause a small difference because ggml
    # does not support passing a scale_factor to interpolate
    interpolate_offset = 0.0
    interpolate_antialias = False
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return pos_embed
    pos_embed = pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
    w0, h0 = w0 + interpolate_offset, h0 + interpolate_offset
    # w0, h0 = w0 + 0.1, h0 + 0.1

    sqrt_N = math.sqrt(N)
    sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
        scale_factor=(sx, sy),
        # (int(w0), int(h0)), # to solve the upsampling shape issue
        mode="bicubic",
        antialias=interpolate_antialias,
    )

    assert int(w0) == patch_pos_embed.shape[-2]
    assert int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


def test_interpolate_pos_encoding():
    img_size = 12
    patch_size = 4
    num_patches = (img_size // patch_size) ** 2
    embed_dim = 8
    pos_embed = torch.randn(1, num_patches + 1, embed_dim)

    x = input_tensor(1, num_patches, embed_dim)
    expected = interpolate_pos_encoding(pos_embed, x, img_size, img_size, patch_size)

    state = {"pos_embed": pos_embed}
    params = {"img_size": img_size, "patch_size": patch_size}
    result = workbench.invoke_test("dino_interpolate_pos_encoding", x, state, params)

    assert torch.allclose(result, expected)


class PrepareTokensModule(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=(patch_size, patch_size), embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def prepare_tokens_with_masks(self, x: Tensor, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x, flatten=True)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + interpolate_pos_encoding(self.pos_embed, x, w, h, self.patch_size)
        return x


def test_prepare_tokens_with_masks():
    img_size = 12
    patch_size = 4
    embed_dim = 6
    module = PrepareTokensModule((img_size, img_size), patch_size, embed_dim)
    state = generate_state(module.state_dict())
    module.load_state_dict(state)
    module.eval()

    x = input_tensor(1, 3, img_size, img_size)
    expected = module.prepare_tokens_with_masks(x)

    x = to_nhwc(x)
    state = convert_to_nhwc(state, key="patch_embed.proj")
    result = workbench.invoke_test("dino_prepare_tokens", x, state)

    assert torch.allclose(result, expected)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = False, proj_bias: bool = True
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


def test_attention():
    dim = 6
    num_heads = 3
    module = Attention(dim=dim, num_heads=num_heads, qkv_bias=True, proj_bias=True)
    state = generate_state(module.state_dict())
    module.load_state_dict(state)
    module.eval()

    x = input_tensor(1, 12, dim)
    expected = module(x)
    result = workbench.invoke_test(
        "dino_attention", x, state, dict(n_heads=num_heads, flash_attn=0)
    )

    assert torch.allclose(result, expected)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values=1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x


def test_block():
    dim = 6
    num_heads = 3
    module = Block(dim=dim, num_heads=num_heads, init_values=1.0)
    state = generate_state(module.state_dict())
    module.load_state_dict(state)
    module.eval()

    x = input_tensor(1, 12, dim)
    expected = module(x)
    result = workbench.invoke_test("dino_block", x, state, dict(n_heads=num_heads))

    assert torch.allclose(result, expected, atol=1e-2)  # precision drop due to GELU in MLP


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn=False):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1
        )
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


@pytest.mark.parametrize("inputs", [1, 2])
def test_feature_fusion(inputs):
    features = 6
    x = [input_tensor(1, features, 4, 4)]
    size = (8, 8)
    if inputs == 2:
        x.append(input_tensor(1, features, 4, 4))
        size = None

    module = _make_fusion_block(features, use_bn=False)
    state = generate_state(module.state_dict())
    module.load_state_dict(state)
    module.eval()

    expected = module(*x, size=size)
    result = workbench.invoke_test("depthany_feature_fusion", x, state=state)

    assert torch.allclose(result, expected)


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv_block(x)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True
        )
        out = self.scratch.output_conv2(out)

        return out


def test_dpt_head():
    in_channels = 4
    features = 6
    h, w = 8, 8
    module = DPTHead(in_channels=in_channels, features=features, use_clstoken=False)
    state = generate_state(module.state_dict())
    module.load_state_dict(state)
    module.eval()

    x = [input_tensor(2, 1, h * w, in_channels) for _ in range(4)]
    expected = module(x, h, w)

    state = convert_to_nhwc(state, key="projects")
    result = workbench.invoke_test("depthany_head", x, state)

    assert torch.allclose(result, expected, atol=1e-3)
