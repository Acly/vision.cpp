import ctypes
from functools import reduce
import torch
import os

from pathlib import Path

float_ptr = ctypes.POINTER(ctypes.c_float)


class RawTensor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("ne", ctypes.c_int * 4),
    ]

    @property
    def n_bytes(self):
        return 4 * self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3]


class RawParam(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
        ("type", ctypes.c_int),
    ]


def torch_to_raw_tensor(name: str, tensor: torch.Tensor):
    tensor_types = {
        torch.float32: 0,  # GGML_TYPE_F32
        torch.int32: 26,  # GGML_TYPE_I32
    }
    t = tensor.contiguous()
    if tensor.dtype not in tensor_types:
        print(f"Warning: tensor {name} has unsupported dtype {tensor.dtype}, converting to float")
        t = t.to(torch.float32)
    while t.dim() < 4:
        t = t.unsqueeze(0)
    assert t.dim() == 4
    raw_tensor = RawTensor()
    raw_tensor.name = name.encode()
    raw_tensor.data = ctypes.cast(t.data_ptr(), ctypes.c_void_p)
    raw_tensor.type = tensor_types[t.dtype]
    raw_tensor.ne[0] = t.size(3)
    raw_tensor.ne[1] = t.size(2)
    raw_tensor.ne[2] = t.size(1)
    raw_tensor.ne[3] = t.size(0)
    return (raw_tensor, t)


def raw_to_torch_tensor(raw_tensor: RawTensor):
    dtype = {
        0: torch.float32,  # GGML_TYPE_F32
        26: torch.int32,  # GGML_TYPE_I32
    }.get(raw_tensor.type, torch.float32)

    shape = tuple(raw_tensor.ne[i] for i in range(3, -1, -1))
    return torch.frombuffer(
        bytearray(ctypes.string_at(raw_tensor.data, raw_tensor.n_bytes)),
        dtype=dtype,
    ).reshape(shape)


def encode_params(params: dict[str, str | int | float]):
    raw_params = []
    for name, value in params.items():
        ptype = 0
        if isinstance(value, int):
            value = str(value).encode()
            ptype = 0
        elif isinstance(value, float):
            value = str(value).encode()
            ptype = 1
        elif isinstance(value, str):
            value = value.encode()
            ptype = 2
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {type(value)}")
        raw_params.append(RawParam(name=name.encode(), value=value, type=ptype))
    return raw_params


# Some python lib (numpy? torch?) loads OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_dir = Path(__file__).parent.parent
bin_dir = root_dir / "build" / "bin"

lib = ctypes.CDLL(str(bin_dir / "vision-workbench.dll"))
lib.visp_workbench.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(RawTensor),
    ctypes.c_int32,
    ctypes.POINTER(RawParam),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.POINTER(RawTensor)),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
]
lib.visp_workbench.restype = ctypes.c_int32


def invoke_test(
    test_case: str,
    input: torch.Tensor | list[torch.Tensor],
    state: dict[str, torch.Tensor],
    params: dict[str, str | int | float] = {},
    backend: str = "cpu",
):
    input = input if isinstance(input, list) else [input]
    raw_inputs = [torch_to_raw_tensor(f"input{i}", tensor) for i, tensor in enumerate(input)]
    raw_inputs += [torch_to_raw_tensor(name, tensor) for name, tensor in state.items()]
    input_tensors = [t for _, t in raw_inputs]
    input_tensors  # keep the tensors alive
    raw_inputs = [t for t, _ in raw_inputs]
    raw_params = encode_params(params)
    raw_output = ctypes.POINTER(RawTensor)()
    output_size = ctypes.c_int32(0)

    result = lib.visp_workbench(
        test_case.encode(),
        (RawTensor * len(raw_inputs))(*raw_inputs),
        len(raw_inputs),
        (RawParam * len(raw_params))(*raw_params) if len(raw_params) > 0 else None,
        len(raw_params),
        ctypes.byref(raw_output),
        ctypes.byref(output_size),
        1 if backend == "cpu" else 2,
    )

    assert result == 0, f"Test case {test_case} failed"
    if output_size.value == 0:
        return None

    output = [raw_to_torch_tensor(raw_output[i]) for i in range(output_size.value)]
    output = output[0] if len(output) == 1 else output
    return output


def input_tensor(*shape: tuple[int]):
    end = reduce(lambda x, y: x * y, shape, 1)
    return torch.arange(0, end).reshape(*shape) / end


def input_like(tensor: torch.Tensor):
    if tensor.dim() == 0:
        return torch.tensor(0.5, dtype=tensor.dtype)
    return input_tensor(*tensor.shape)


def generate_state(state_dict: dict[str, torch.Tensor]):
    return {k: input_like(v) if v.dtype.is_floating_point else v for k, v in state_dict.items()}


def randomize(state_dict: dict[str, torch.Tensor]):
    return {
        k: torch.rand_like(v) if v.dtype.is_floating_point else v for k, v in state_dict.items()
    }


def to_nhwc(tensor: torch.Tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()


def to_nchw(tensor: torch.Tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()


def convert_to_nhwc(state: dict[str, torch.Tensor], key=""):
    for k, v in state.items():
        is_conv = (
            v.ndim == 4
            and v.shape[2] == v.shape[3]
            and v.shape[2] in (1, 3, 4, 7)
            and k.endswith("weight")
        )
        if is_conv and (key == "" or key in k):
            if v.shape[1] == 1:  # depthwise
                state[k] = v.permute(2, 3, 1, 0).contiguous()
            else:
                state[k] = v.permute(0, 2, 3, 1).contiguous()
    return state


def fuse_batch_norm(model: dict[str, torch.Tensor], key: str, key_bn: str):
    suffix_weight = f"{key_bn}.weight"
    suffix_bias = f"{key_bn}.bias"

    if key.endswith(suffix_weight):
        base = key.removesuffix(suffix_weight)
        weight = model[key]
        var = model[f"{base}{key_bn}.running_var"]
        return weight / torch.sqrt(var + 1e-5)

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        bias = model[key]
        weight = model[f"{base}{key_bn}.weight"]
        mean = model[f"{base}{key_bn}.running_mean"]
        var = model[f"{base}{key_bn}.running_var"]
        return bias - mean * weight / torch.sqrt(var + 1e-5)

    elif key.endswith(f"{key_bn}.running_mean") or key.endswith(f"{key_bn}.running_var"):
        return None

    return model[key]


def fuse_conv_2d_batch_norm(
    model: dict[str, torch.Tensor],
    key: str,
    key_module: str,
    key_conv: str,
    key_norm: str,
    out: dict[str, torch.Tensor],
):
    suffix_conv = f"{key_module}{key_conv}.weight"
    suffix_norm = f"{key_module}{key_norm}."

    if key.endswith(suffix_conv):
        base = key.removesuffix(suffix_conv)
        conv_weight = model[key]
        bn_weight = model[f"{base}{suffix_norm}weight"]
        bn_bias = model[f"{base}{suffix_norm}bias"]
        bn_mean = model[f"{base}{suffix_norm}running_mean"]
        bn_var = model[f"{base}{suffix_norm}running_var"]
        conv_bias = model.get(f"{base}{key_module}{key_conv}.bias", torch.zeros_like(bn_bias))

        bn_weight_var = bn_weight / torch.sqrt(bn_var + 1e-5)
        fused_weight = conv_weight * bn_weight_var[:, None, None, None]
        fused_bias = conv_bias
        fused_bias -= bn_mean
        fused_bias *= bn_weight_var
        fused_bias += bn_bias
        # fused_bias = (conv_bias - bn_mean) * bn_weight_var + bn_bias

        out[key] = fused_weight
        out[key.replace("weight", "bias")] = fused_bias
        return True

    elif suffix_norm in key:
        return True  # batch norm was fused above

    return False  # no match


def print_results(result: torch.Tensor, expected: torch.Tensor):
    print("\ntorch seed:", torch.initial_seed())
    print("\nresult -----", result, sep="\n")
    print("\nexpected ---", expected, sep="\n")
