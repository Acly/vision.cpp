from ctypes import CDLL, byref
from enum import Enum
from pathlib import Path
from typing import NamedTuple
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


class ESRGAN:
    arch = Arch.esrgan

    @classmethod
    def load(cls, path: str | Path, device: Device):
        api = get_lib()
        handle = lib.Model()
        check(
            api.visp_model_load(
                lib.path_to_char_p(path), device._handle, cls.arch.value, byref(handle)
            )
        )
        return cls(api, handle)

    def compute(self, image: Image):
        api = self._api
        in_view = _img_view(image)
        out_view = lib.ImageView()
        out_data = lib.ImageData()
        check(api.visp_esrgan_compute(self._handle, in_view, byref(out_view), byref(out_data)))
        try:
            result = lib.ImageView.to_pil_image(out_view)
        finally:
            api.visp_image_destroy(out_data)
        return result

    def __init__(self, api: CDLL, handle: lib.Handle):
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
