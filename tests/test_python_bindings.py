import sys
from pathlib import Path
from PIL import Image

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "bindings" / "python"))
from visioncpp import Backend, Device, ESRGAN  # noqa

model_dir = root_dir / "models"
image_dir = root_dir / "tests" / "input"
result_dir = root_dir / "tests" / "results" / "python"
result_dir.mkdir(parents=True, exist_ok=True)


def test_esrgan():
    dev = Device.init(Backend.gpu)
    model = ESRGAN.load(str(model_dir / "RealESRGAN-x4plus_anime-6B-F16.gguf"), dev)
    input_image = Image.open(str(image_dir / "vase-and-bowl.jpg"))
    output_image = model.compute(input_image)
    output_image.save(str(result_dir / "esrgan.png"))
