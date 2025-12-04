from ctypes import CDLL, byref, c_int32
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Sequence
import PIL.Image

from . import _lib as lib
from ._lib import get_lib, check


class ImageFormat(Enum):
    rgba_u8 = 0
    bgra_u8 = 1
    argb_u8 = 2
    rgb_u8 = 3
    alpha_u8 = 4

    rgba_f32 = 5
    rgb_f32 = 6
    alpha_f32 = 7


class ImageRef(NamedTuple):
    width: int
    height: int
    stride: int
    format: ImageFormat
    data: bytes


Image = ImageRef | PIL.Image.Image


class Backend(Enum):
    auto = 0
    cpu = 1
    gpu = 2

    vulkan = gpu | 1 << 8

    @property
    def is_cpu(self):
        return self.value & 0xFF00 == Backend.cpu.value

    @property
    def is_gpu(self):
        return self.value & 0xFF00 == Backend.gpu.value


class Device:
    @staticmethod
    def init(backend: Backend = Backend.auto):
        api = get_lib()
        handle = lib.Device()
        check(api.visp_device_init(backend.value, byref(handle)))
        return Device(api, handle)

    @property
    def type(self) -> Backend:
        return Backend(self._api.visp_device_type(self._handle))

    @property
    def name(self) -> str:
        return self._api.visp_device_name(self._handle).decode()

    @property
    def description(self) -> str:
        return self._api.visp_device_description(self._handle).decode()

    def __init__(self, api: CDLL, handle: lib.Handle):
        self._api = api
        self._handle = handle

    def __del__(self):
        self._api.visp_device_destroy(self._handle)


class Arch(Enum):
    sam = 0
    birefnet = 1
    depth_anything = 2
    migan = 3
    esrgan = 4
    unknown = 5


class Model:
    @classmethod
    def load(cls, path: str | Path, device: Device, arch=Arch.unknown):
        api = get_lib()
        handle = lib.Model()
        path_str = lib.path_to_char_p(path)
        if arch is Arch.unknown:
            arch_v = c_int32()
            check(api.visp_model_detect_family(path_str, byref(arch_v)))
            arch = Arch(arch_v.value)
        else:
            arch_v = arch.value

        check(api.visp_model_load(path_str, device._handle, arch_v, byref(handle)))
        return cls(api, handle, arch)

    def compute(self, *images: Image, args: Sequence[int] | None = None):
        if args is None:
            args = []

        in_views = [_img_view(i) for i in images]
        in_views_array = (lib.ImageView * len(in_views))(*in_views)
        args_array = (lib.c_int32 * len(args))(*args)
        out_view = lib.ImageView()
        out_data = lib.ImageData()
        check(
            self._api.visp_model_compute(
                self._handle,
                self.arch.value,
                in_views_array,
                len(in_views_array),
                args_array,
                len(args_array),
                byref(out_view),
                byref(out_data),
            )
        )
        try:
            result = lib.ImageView.to_pil_image(out_view)
        finally:
            self._api.visp_image_destroy(out_data)
        return result

    def __init__(self, api: CDLL, handle: lib.Handle, arch: Arch):
        self.arch = arch
        self._api = api
        self._handle = handle

    def __del__(self):
        self._api.visp_model_destroy(self._handle, self.arch.value)


def _img_view(i: Image) -> lib.ImageView:
    if isinstance(i, PIL.Image.Image):
        return lib.ImageView.from_pil_image(i)
    elif isinstance(i, ImageRef):
        return lib.ImageView.from_bytes(i.width, i.height, i.stride, i.format.value, i.data)
    else:
        raise TypeError("Expected a PIL Image or ImageRef")
