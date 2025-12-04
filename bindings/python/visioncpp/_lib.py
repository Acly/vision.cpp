import ctypes
import platform
from pathlib import Path
from ctypes import c_byte, c_char_p, c_void_p, c_int32, POINTER
from PIL import Image


class Error(Exception):
    pass


def _image_format_to_string(format: int):
    match format:
        case 0:
            return "RGBA"
        case 3:
            return "RGB"
        case 4:
            return "L"
        case _:
            raise ValueError(f"Unsupported image format: {format}")


def _image_mode_from_string(mode: str):
    match mode:
        case "RGBA":
            return 0, 4  # visp::image_format, bytes per pixel
        case "RGB":
            return 3, 3
        case "L":
            return 4, 1
        case _:
            raise ValueError(f"Unsupported image mode: {mode}")


class ImageView(ctypes.Structure):
    _fields_ = [
        ("width", c_int32),
        ("height", c_int32),
        ("stride", c_int32),
        ("format", c_int32),
        ("data", c_void_p),
    ]

    @staticmethod
    def from_bytes(width: int, height: int, stride: int, format: int, data: bytes):
        ptr = (c_byte * len(data)).from_buffer_copy(data)
        return ImageView(width, height, stride, format, ctypes.cast(ptr, ctypes.c_void_p))

    @staticmethod
    def from_pil_image(image):
        assert isinstance(image, Image.Image), "Expected a PIL Image"
        data = image.tobytes()
        w, h = image.size
        format, bpp = _image_mode_from_string(image.mode)
        return ImageView.from_bytes(w, h, w * bpp, format, data)

    def to_pil_image(self):
        mode = _image_format_to_string(self.format)
        size = self.height * self.stride
        data = memoryview((c_byte * size).from_address(self.data))
        return Image.frombytes(mode, (self.width, self.height), data, "raw", mode, self.stride)


class _ImageData(ctypes.Structure):
    pass


class _Device(ctypes.Structure):
    pass


class _Model(ctypes.Structure):
    pass


ImageData = POINTER(_ImageData)
Device = POINTER(_Device)
Model = POINTER(_Model)

Handle = ctypes._Pointer


def _load():
    cur_dir = Path(__file__).parent
    system = platform.system().lower()
    if system == "windows":
        prefix = ""
        suffix = ".dll"
    elif system == "darwin":
        prefix = "lib"
        suffix = ".dylib"
    else:  # assume Linux / Unix
        prefix = "lib"
        suffix = ".so"
    libname = f"{prefix}visioncpp{suffix}"
    paths = [
        cur_dir / libname,
        cur_dir.parent.parent.parent / "build" / "bin" / libname,
    ]
    error = None
    for path in paths:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                return lib
            except OSError as e:
                error = e
                continue
    raise OSError(f"Could not load vision.cpp library from paths: {error}")


def init():
    lib = _load()

    lib.visp_get_last_error.restype = c_char_p

    lib.visp_image_destroy.argtypes = [ImageData]
    lib.visp_image_destroy.restype = None

    lib.visp_device_init.argtypes = [c_int32, POINTER(Device)]
    lib.visp_device_init.restype = c_int32

    lib.visp_device_destroy.argtypes = [Device]
    lib.visp_device_destroy.restype = None

    lib.visp_device_type.argtypes = [Device]
    lib.visp_device_type.restype = c_int32

    lib.visp_device_name.argtypes = [Device]
    lib.visp_device_name.restype = c_char_p

    lib.visp_device_description.argtypes = [Device]
    lib.visp_device_description.restype = c_char_p

    lib.visp_model_detect_family.argtypes = [c_char_p, POINTER(c_int32)]
    lib.visp_model_detect_family.restype = c_int32

    lib.visp_model_load.argtypes = [c_char_p, Device, c_int32, POINTER(Model)]
    lib.visp_model_load.restype = c_int32

    lib.visp_model_destroy.argtypes = [Model, c_int32]
    lib.visp_model_destroy.restype = None

    lib.visp_model_compute.argtypes = [
        Model,
        c_int32,
        POINTER(ImageView),
        c_int32,
        POINTER(c_int32),
        c_int32,
        POINTER(ImageView),
        POINTER(ImageData),
    ]
    lib.visp_model_compute.restype = c_int32

    return lib


_lib: ctypes.CDLL | None = None


def get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = init()
    return _lib


def check(return_value: int):
    if return_value == 0:
        assert _lib is not None, "Library not initialized"
        raise Error(_lib.visp_get_last_error().decode())


def path_to_char_p(p: str | Path):
    return str(p).encode()
