import sys
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "bindings" / "python"))
from visioncpp import Arch, Backend, Device, Model  # noqa

model_dir = root_dir / "models"
image_dir = root_dir / "tests" / "input"
result_dir = root_dir / "tests" / "results" / "python"
result_dir.mkdir(parents=True, exist_ok=True)
ref_dir = root_dir / "tests" / "reference"

@pytest.fixture
def device(pytestconfig):
    if pytestconfig.getoption("ci"):
        return Device.init(Backend.cpu)
    return Device.init()


def compare_images(name: str, result: Image.Image, tolerance: float = 0.015):
    name = f"{name}-gpu.png"
    result.save(str(result_dir / name))
    result = result.convert("RGB")
    result_array = np.array(result).astype(np.float32) / 255.0

    ref_image = Image.open(str(ref_dir / name)).convert("RGB")
    ref_array = np.array(ref_image).astype(np.float32) / 255.0

    if ref_array.shape != result_array.shape:
        raise AssertionError(
            f"Image shapes do not match: {ref_array.shape} vs {result_array.shape}"
        )
    rmse = np.sqrt(np.mean((ref_array - result_array) ** 2))
    if rmse > tolerance:
        raise AssertionError(f"Images differ: RMSE={rmse} exceeds tolerance={tolerance}")


def test_sam(device: Device):
    model = Model.load(model_dir / "MobileSAM-F16.gguf", device)
    assert model.arch is Arch.sam

    img = Image.open(str(image_dir / "cat-and-hat.jpg"))
    result_box = model.compute(img, args=[180, 110, 505, 330])
    result_point = model.compute(img, args=[200, 300])
    compare_images("mobile_sam-box", result_box)
    compare_images("mobile_sam-point", result_point)

def test_birefnet(device: Device):
    model = Model.load(model_dir / "BiRefNet-lite-F16.gguf", device)
    assert model.arch is Arch.birefnet

    img = Image.open(str(image_dir / "wardrobe.jpg"))
    result = model.compute(img)
    compare_images("birefnet", result)

def test_depth_anything(device: Device):
    model = Model.load(model_dir / "Depth-Anything-V2-Small-F16.gguf", device)
    assert model.arch is Arch.depth_anything

    img = Image.open(str(image_dir / "wardrobe.jpg"))
    result = model.compute(img)
    compare_images("depth-anything", result)

def test_migan(device: Device):
    model = Model.load(model_dir / "MIGAN-512-places2-F16.gguf", device)
    assert model.arch is Arch.migan

    img = Image.open(str(image_dir / "bench-image.jpg")).convert("RGBA")
    mask = Image.open(str(image_dir / "bench-mask.png"))
    result = model.compute(img, mask)
    result = Image.alpha_composite(img, result)
    compare_images("migan", result)

def test_esrgan(device: Device):
    model = Model.load(str(model_dir / "RealESRGAN-x4plus_anime-6B-F16.gguf"), device)
    assert model.arch is Arch.esrgan

    img = Image.open(str(image_dir / "vase-and-bowl.jpg"))
    result = model.compute(img)
    compare_images("esrgan", result)
