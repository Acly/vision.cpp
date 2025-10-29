import pytest
import torch

from . import workbench
from .workbench import to_nchw, to_nhwc


def test_linear():
    x = torch.rand(2, 5)
    weight = torch.rand(3, 5)
    bias = torch.tensor([7, 21, -5]).float()

    result = workbench.invoke_test("linear", x, dict(weight=weight, bias=bias))

    expected = torch.nn.functional.linear(x, weight, bias)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("scenario", ["stride_1_pad_0", "stride_2_pad_1", "dilation_2_pad_2"])
@pytest.mark.parametrize("memory_layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("batch", ["single", "batch"])
@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_conv_2d_depthwise(scenario: str, memory_layout: str, batch: str, backend: str):
    stride, pad, dilate = {
        "stride_1_pad_0": (1, 0, 1),
        "stride_2_pad_1": (2, 1, 1),
        "dilation_2_pad_2": (1, 2, 2),
    }[scenario]
    if memory_layout == "nhwc" and dilate != 1:
        pytest.skip("Dilation is not supported in NHWC layout")

    x = torch.rand(1, 9, 5, 6).float()
    k = torch.rand(9, 1, 3, 3).float()

    if batch == "batch":
        x = torch.cat((x, x * -1), dim=0)

    expected = torch.nn.functional.conv2d(
        x, k, stride=stride, padding=pad, dilation=dilate, groups=k.shape[0]
    )

    if memory_layout == "nhwc":
        x = to_nhwc(x)
        k = k.permute(2, 3, 1, 0)
    test_case = f"conv_2d_depthwise_{memory_layout}"
    params = dict(stride=stride, pad=pad, dilation=dilate)
    result = workbench.invoke_test(test_case, x, dict(weight=k), params, backend)
    if memory_layout == "nhwc":
        result = to_nchw(result)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("scenario", ["3x3", "5x5", "stride2"])
def test_conv_transpose_2d(scenario: str):
    ksize, stride = {
        "3x3": (3, 1),
        "5x5": (5, 1),
        "stride2": (3, 2),
        "nchw": (3, 1),
    }[scenario]
    x = torch.arange(2 * 11 * 4 * 5).reshape(2, 11, 4, 5).float()
    weight = torch.arange(11 * 2 * ksize * ksize).reshape(11, 2, ksize, ksize).float()
    bias = None
    expected = torch.nn.functional.conv_transpose2d(x, weight, bias, stride=stride)

    x = to_nhwc(x)  # -> [N, H, W, C_in]
    result = workbench.invoke_test(
        "conv_transpose_2d",
        x,
        dict(weight=weight),
        dict(stride=stride),
        backend="vulkan",
    )
    result = to_nchw(result)

    assert torch.allclose(result, expected)


def test_batch_norm_2d():
    x = torch.rand(1, 3, 4, 5)
    weight = torch.rand(3)
    bias = torch.rand(3)
    mean = torch.rand(3)
    var = torch.arange(1, 4).float()
    expected = torch.nn.functional.batch_norm(x, mean, var, weight, bias, eps=1e-5)

    x = to_nhwc(x)

    var = (var + 1e-5).sqrt()
    state = dict(weight=weight, bias=bias, running_mean=mean, running_var=var)
    result = workbench.invoke_test("batch_norm_2d", x, state)
    result = to_nchw(result)

    assert torch.allclose(result, expected)


def test_layer_norm():
    dim = 20
    x = torch.rand(4, 5, dim)
    weight = torch.rand(dim)
    bias = torch.rand(dim)

    result = workbench.invoke_test("layer_norm", x, dict(weight=weight, bias=bias))

    expected = torch.nn.functional.layer_norm(x, [dim], weight, bias, eps=1e-5)
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_window_partition(backend: str):
    win = 3
    x = torch.arange(2 * 7 * 5).reshape(1, 5, 7, 2).float()
    B, H, W, C = x.shape

    pad_b = (win - H % win) % win
    pad_r = (win - W % win) % win
    padding = pad_b > 0 or pad_r > 0

    expected = x
    if padding:
        expected = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

    pH, pW = H + pad_b, W + pad_r
    nH = pH // win
    nW = pW // win
    # window partition
    expected = (
        expected.view(B, nH, win, nW, win, C).transpose(2, 3).reshape(B * nH * nW, win * win, C)
    )

    result = workbench.invoke_test("sam_window_partition", x, {}, backend=backend)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("shift", [(0, 2, -1, 0), (0, -2, 0, 3)])
@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_roll(shift: tuple[int, int, int, int], backend: str):
    x = torch.arange(4 * 5 * 6).reshape(1, 4, 5, 6).float()
    shifts = tuple(s for s in shift if s != 0)
    dims = tuple(i for i, s in enumerate(shift) if s != 0)
    expected = torch.roll(x, shifts=shifts, dims=dims)

    params = dict(s0=shift[3], s1=shift[2], s2=shift[1], s3=shift[0])
    result = workbench.invoke_test("roll", x, {}, params, backend)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("size", ["one", "small", "large"])
@pytest.mark.parametrize("scale", [0.6, 2.0])
@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_interpolate(mode: str, align_corners: bool, size: str, scale: float, backend: str):
    b, c, h, w = {
        "one": (1, 2, 1, 3),
        "small": (1, 3, 2, 3),
        "large": (4, 19, 20, 30),
    }[size]
    target = (round(h * scale), round(w * scale))
    x = torch.arange(b * c * h * w).reshape(b, c, h, w).float()
    expected = torch.nn.functional.interpolate(
        x, size=target, mode=mode, align_corners=align_corners
    )

    params = dict(mode=mode, h=target[0], w=target[1], align_corners=1 if align_corners else 0)
    result = workbench.invoke_test("interpolate", x, {}, params, backend)
    assert torch.allclose(result, expected)
