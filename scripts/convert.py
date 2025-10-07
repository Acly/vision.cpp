#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "torchvision",
#   "timm",
#   "pytest",
#   "opencv-python",
#   "ruff",
#   "einops>=0.8.1",
#   "spandrel>=0.4.1",
#   "gguf>=0.17.1",
# ]
# ///
import argparse
import itertools
import torch
import safetensors
import numpy as np

from enum import Enum
from pathlib import Path
from gguf import GGUFWriter, Metadata, GGML_QUANT_VERSION
from torch import Tensor

#
# Common


class TensorLayout(Enum):
    unknown = "unknown"
    nchw = "whcn"
    nhwc = "cwhn"

    @staticmethod
    def parse(s: str):
        if s == "whcn" or s == "nchw":
            return TensorLayout.nchw
        if s == "cwhn" or s == "nhwc":
            return TensorLayout.nhwc
        return TensorLayout.unknown


class Writer(GGUFWriter):
    def __init__(self, path: Path, arch_name: str, float_type: str, verbose: bool):
        super().__init__(path, arch_name)
        self.arch = arch_name
        self.float_type = float_type
        self.tensor_layout = TensorLayout.unknown
        self.verbose = verbose
        self.conv2d_weights: list[int] = []
        self._index = 0

    def add_tensor(self, name: str, tensor: Tensor, float_type: str | None = None):
        if len(name) >= 64:
            print("Warning: name too long", len(name), name)

        float_type = float_type or self.float_type
        if float_type == "f16" and tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)

        tensor_data = tensor.numpy()
        if self.verbose:
            print(name, tensor.shape, tensor_data.dtype)
        super().add_tensor(name, tensor_data)
        self._index += 1

    def convert_tensor_2d(self, tensor: Tensor):
        # assume tensor is NCHW layout (PyTorch default)
        if self.tensor_layout is TensorLayout.nhwc:
            return conv_2d_to_nhwc(tensor)
        else:
            # add tensor index to list to optionally convert layout on the fly later
            self.conv2d_weights.append(self._index)
            return tensor

    def add_int32(self, name: str, value: int):
        print("*", name, "=", value)
        super().add_int32(name, value)

    def set_tensor_layout(self, layout: TensorLayout):
        print("*", f"{self.arch}.tensor_data_layout", "=", layout.value)
        self.tensor_layout = layout
        self.add_tensor_data_layout(layout.value)

    def set_tensor_layout_default(self, layout: TensorLayout):
        if self.tensor_layout is TensorLayout.unknown:
            self.set_tensor_layout(layout)

    def add_conv2d_weight_indices(self):
        if self.conv2d_weights:
            self.add_array(f"{self.arch}.conv2d_weights", self.conv2d_weights)


batch_norm_eps = 1e-5


def is_conv_2d(name: str, tensor: Tensor):
    return (
        tensor.ndim == 4
        and tensor.shape[2] == tensor.shape[3]
        and tensor.shape[2] in (1, 3, 4, 7, 14)
        and name.endswith("weight")
    )


def conv_2d_to_nhwc(kernel: Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0)  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1)  # C_out H W C_in


def conv_transpose_2d_to_nhwc(kernel: Tensor):
    # C_in C_out H W -> C_out H W C_in
    return kernel.permute(1, 2, 3, 0)


def fuse_batch_norm(model: dict[str, Tensor], key: str, key_bn: str):
    suffix_weight = f"{key_bn}.weight"
    suffix_bias = f"{key_bn}.bias"

    if key.endswith(suffix_weight):
        base = key.removesuffix(suffix_weight)
        weight = model[key]
        var = model[f"{base}{key_bn}.running_var"]
        return weight / torch.sqrt(var + batch_norm_eps)

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        bias = model[key]
        weight = model[f"{base}{key_bn}.weight"]
        mean = model[f"{base}{key_bn}.running_mean"]
        var = model[f"{base}{key_bn}.running_var"]
        return bias - mean * weight / torch.sqrt(var + batch_norm_eps)

    elif key.endswith(f"{key_bn}.running_mean") or key.endswith(f"{key_bn}.running_var"):
        return None

    return model[key]


def fuse_conv_2d_batch_norm(
    model: dict[str, Tensor],
    key: str,
    name: str,
    key_module: str,
    key_conv: str,
    key_norm: str,
    writer: Writer,
):
    suffix_conv = f"{key_module}{key_conv}.weight"
    suffix_bias = f"{key_module}{key_conv}.bias"
    suffix_norm = f"{key_module}{key_norm}."

    if key.endswith(suffix_conv):
        conv_weight = model[key]
        base = key.removesuffix(suffix_conv)
        bn_weight = model.get(f"{base}{suffix_norm}weight")
        if bn_weight is None:
            return False
        bn_bias = model[f"{base}{suffix_norm}bias"]
        bn_mean = model[f"{base}{suffix_norm}running_mean"]
        bn_var = model[f"{base}{suffix_norm}running_var"]
        conv_bias = model.get(f"{base}{suffix_bias}", torch.zeros_like(bn_bias))

        bn_weight = bn_weight / torch.sqrt(bn_var + batch_norm_eps)
        fused_weight = conv_weight * bn_weight[:, None, None, None]
        fused_bias = (conv_bias - bn_mean) * bn_weight + bn_bias

        fused_weight = writer.convert_tensor_2d(fused_weight)
        writer.add_tensor(name, fused_weight)
        writer.add_tensor(name.replace("weight", "bias"), fused_bias)
        return True

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        return f"{base}{suffix_norm}weight" in model

    elif suffix_norm in key:
        return True  # batch norm was fused above

    return False  # tensor is not part of conv2d+batch-norm


#
# MobileSAM


def convert_sam(input_filepath: Path, writer: Writer):
    writer.add_license("apache-2.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model: dict[str, Tensor] = torch.load(input_filepath, map_location="cpu", weights_only=True)

    for key, tensor in model.items():
        name = key
        name = name.replace("image_encoder.", "enc.")
        name = name.replace("mask_decoder.", "dec.")
        name = name.replace("_image_to_token.", "_i2t.")
        name = name.replace("_token_to_image.", "_t2i.")

        if name.endswith("attention_biases"):
            num_heads = tensor.shape[0]
            resolution = {4: 7, 5: 14, 10: 7}[num_heads]
            attention_bias_idxs = build_attention_bias_indices(resolution)
            name = name + "_indexed"
            tensor = tensor[:, attention_bias_idxs]

        if "local_conv" in key:  # always convert to nhwc
            original_tensor_layout = writer.tensor_layout
            writer.tensor_layout = TensorLayout.nhwc
            fuse_conv_2d_batch_norm(model, key, name, "", "c", "bn", writer)
            writer.tensor_layout = original_tensor_layout
            continue

        if fuse_conv_2d_batch_norm(model, key, name, "", "c", "bn", writer):
            continue

        if name.endswith("neck.0.weight") or name.endswith("neck.2.weight"):
            assert tensor.shape[2] == tensor.shape[3] and tensor.shape[2] <= 3
            tensor = writer.convert_tensor_2d(tensor)

        # Precompute dense positional embeddings from random matrix stored in the model
        if name == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix":
            pe = build_dense_positional_embeddings(tensor)
            writer.add_tensor("dec.dense_positional_embedding", pe, "f32")

        if name in ["dec.iou_token.weight", "dec.mask_tokens.weight"]:
            writer.add_tensor(name, tensor, "f32")
            continue

        writer.add_tensor(name, tensor)


def build_attention_bias_indices(resolution: int):
    points = list(itertools.product(range(resolution), range(resolution)))
    N = len(points)
    attention_offsets = {}
    idxs = []
    for p1 in points:
        for p2 in points:
            offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            if offset not in attention_offsets:
                attention_offsets[offset] = len(attention_offsets)
            idxs.append(attention_offsets[offset])

    return torch.LongTensor(idxs).view(N, N)


def build_dense_positional_embeddings(
    positional_encoding_gaussian_matrix: torch.Tensor, image_embedding_size=64
):
    # from sam/modeling/prompt_encoder.py - PositionEmbeddingRandom
    h, w = image_embedding_size, image_embedding_size
    grid = torch.ones((h, w), dtype=torch.float32)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    coords = torch.stack((x_embed, y_embed), dim=-1)
    coords = 2 * coords - 1
    coords = coords @ positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    # outputs d_1 x ... x d_n x C shape
    pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    return pe


#
# BirefNet


def convert_birefnet(input_filepath: Path, writer: Writer):
    writer.add_license("mit")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    weights = safetensors.safe_open(input_filepath, "pt")
    model: dict[str, Tensor] = {k: weights.get_tensor(k) for k in weights.keys()}

    x = model["bb.layers.0.blocks.0.attn.proj.bias"]
    if x.shape[0] == 96:
        writer.add_string("swin.config", "tiny")
        writer.add_int32("swin.embed_dim", 96)
    elif x.shape[0] == 192:
        writer.add_string("swin.config", "large")
        writer.add_int32("swin.embed_dim", 192)
    else:
        raise ValueError(f"Unsupported Swin Transformer embed dim: {x.shape[0]}")

    image_size = 1024
    if "HR" in input_filepath.name or "2K" in input_filepath.name:
        image_size = 2048  # actually 2K should rather be 2560x1440
    elif "dynamic" in input_filepath.name:
        image_size = -1
    writer.add_int32("birefnet.image_size", image_size)
    writer.add_int32("birefnet.image_multiple", 128)

    for key, tensor in model.items():
        # Shorten some names to fit into 64 chars
        name = key
        name = name.replace("decoder_block", "block")
        name = name.replace("atrous_conv", "conv")
        name = name.replace("modulator_conv", "modulator")
        name = name.replace("offset_conv", "offset")
        name = name.replace("regular_conv", "conv")

        if name.endswith("relative_position_index"):
            continue  # precomputed in c++ code

        # Fuse all regular conv + batch norm pairs into a single conv with bias
        if fuse_conv_2d_batch_norm(model, key, name, "global_avg_pool.", "1", "2", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "dec_att.", "conv1", "bn1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "", "conv_in", "bn_in", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "", "conv_out", "bn_out", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_4.", "0", "1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_3.", "0", "1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_2.", "0", "1", writer):
            continue

        # Fuse batch norm into multiply+add for deformable conv
        tensor = fuse_batch_norm(model, key, "bn")
        if tensor is None:
            continue  # batch norm was fused

        if is_conv_2d(name, tensor):
            if "patch_embed" in name:  # part of SWIN, always store as NHWC
                tensor = conv_2d_to_nhwc(tensor)
            else:  # store rest in requested tensor layout
                tensor = writer.convert_tensor_2d(tensor)

        writer.add_tensor(name, tensor)


#
# Depth-Anything


def convert_depth_anything(input_filepath: Path, writer: Writer):
    writer.add_license("apache-2.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model: dict[str, Tensor] = torch.load(input_filepath, map_location="cpu", weights_only=True)

    for key, tensor in model.items():
        name = key

        if is_conv_2d(name, tensor):
            if "patch_embed" in name or "projects" in name:
                tensor = conv_2d_to_nhwc(tensor)
            else:
                tensor = writer.convert_tensor_2d(tensor)

        writer.add_tensor(name, tensor)


#
# MI-GAN


def convert_migan(input_filepath: Path, writer: Writer):
    writer.add_license("mit")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model: dict[str, Tensor] = torch.load(input_filepath, weights_only=True)

    if "encoder.b512.fromrgb.weight" in model:
        writer.add_int32("migan.image_size", 512)
    elif "encoder.b256.fromrgb.weight" in model:
        writer.add_int32("migan.image_size", 256)

    for name, tensor in model.items():
        if is_conv_2d(name, tensor):
            tensor = writer.convert_tensor_2d(tensor)

        writer.add_tensor(name, tensor)


#
# ESRGAN


def convert_esrgan(input_filepath: Path, writer: Writer):
    from spandrel import ModelLoader

    # Load the model using spandrel
    # - it converts the various versions of ESRGAN checkpoints to a common format
    model = ModelLoader().load_from_file(input_filepath)

    if model.model.shuffle_factor is not None:
        raise ValueError("RealESRGAN models with pixel shuffle are not supported yet.")
    if getattr(model.model, "plus", False):
        raise ValueError("RealESRGAN+ (plus) models are not supported yet.")

    writer.set_tensor_layout_default(TensorLayout.nchw)
    writer.add_int32("esrgan.scale", model.scale)
    for tag in model.tags:
        if tag.endswith("nb"):
            writer.add_int32("esrgan.block_count", int(tag[:-2]))
        if tag.endswith("nf"):
            writer.add_int32("esrgan.filter_count", int(tag[:-2]))

    for name, tensor in model.model.state_dict().items():
        if is_conv_2d(name, tensor):
            tensor = writer.convert_tensor_2d(tensor)
        writer.add_tensor(name, tensor)


#
# Main
#######

arch_names = {
    "sam": "mobile-sam",
    "birefnet": "birefnet",
    "depth-anything": "depth-anything",
    "migan": "migan",
    "esrgan": "esrgan",
}

file_types = {None: 0, "f32": 0, "f16": 1}

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert model weights (.pt/.pth/.safetensors) to GGUF format.")
    parser.add_argument("arch", choices=list(arch_names.keys()), help="Model architecture")
    parser.add_argument("input", type=str, help="Path to the input model file")
    parser.add_argument("--output", "-o", type=str, default="models", help="Path to the output directory or file")
    parser.add_argument("--quantize", "-q", choices=["f16"], default=None, help="Convert float weights to the specified data type")
    parser.add_argument("--layout", "-l", choices=["whcn", "cwhn"], default=None, help="Tensor data layout for 2D operations like convolution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model for metadata")
    parser.add_argument("--metadata", type=Path, help="Specify the path for an authorship metadata override file")
    # fmt: on
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    quant_suffix = f"-{args.quantize.upper()}" if args.quantize else ""
    layout_suffix = f"-{args.layout.upper()}" if args.layout else ""
    if output_path.is_dir() or output_path.suffix != ".gguf":
        output_path = output_path / f"{input_path.stem}{quant_suffix}{layout_suffix}.gguf"

    print(f"Converting {args.arch}")
    print("* input: ", input_path)
    print("* output:", output_path)

    try:
        writer = Writer(
            output_path,
            arch_names.get(args.arch, args.arch),
            args.quantize,
            args.verbose,
        )
        metadata = Metadata.load(args.metadata, input_path.with_suffix(""), args.model_name)

        if args.layout is not None:
            writer.set_tensor_layout(TensorLayout.parse(args.layout))

        match args.arch:
            case "sam":
                convert_sam(input_path, writer)
            case "birefnet":
                convert_birefnet(input_path, writer)
            case "depthany" | "depth-anything":
                convert_depth_anything(input_path, writer)
            case "migan":
                convert_migan(input_path, writer)
            case "esrgan":
                convert_esrgan(input_path, writer)
            case _:
                raise ValueError(f"Unknown architecture: {args.arch}")

        metadata.set_gguf_meta_model(writer)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_types[args.quantize])
        writer.add_conv2d_weight_indices()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
    except ValueError as e:
        print("\033[31mError:\033[0m", e)
        exit(1)
    except Exception as e:
        print("\033[31mError:\033[0m", e)
        exit(-1)

    print("")
