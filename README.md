# _vision_.cpp

Computer Vision ML inference in C++

* Self-contained C++ library
* Efficient inference on consumer CPU and GPUs
* Lightweight deployment on many platforms (Windows, Linux,)
* Modular design for experimentation and extension

Inspired by the [llama.cpp]() project and based on the same [infrastructure]().

> [!NOTE]
> The API is not considered stable at the moment and may change without notice.

### Features

| Model                                        | Task             | Backends    |
| :------------------------------------------- | :--------------- | :---------- |
| [**MobileSAM**](#segment-anything-model-sam) | Segmentation     | CPU, Vulkan |
| [**BiRefNet**](#birefnet)                    | Segmentation     | CPU, Vulkan |
| [**MI-GAN**](#mi-gan)                        | Inpainting       | CPU, Vulkan |
| [**ESRGAN**](#esrgan)                        | Super-resolution | CPU, Vulkan |

## Get Started

See [Building](#building) to build from source. Binaries can be found in `build/bin` afterwards.

### Example: Select an object in an image

Let's use MobileSAM to generate a segmentation mask for the <object>
at pixel position (320, 240).

You can download the required model from huggingface: [MobileSAM-F16.gguf](https://huggingface.co/Acly/MobileSAM-GGUF/resolve/main/MobileSAM-F16.gguf).

#### CLI

```sh
vision-cli -m MobileSAM-F16.gguf -i input.png -p 320 240 -o mask.png
```

#### API

```c++
#include <visp/vision.hpp>
using namespace visp;

void main() {
  backend   cpu = backend_init(backend_type::cpu);
  sam_model sam = sam_load_model("MobileSAM-F16.gguf", cpu);
  
  image_data input_image = image_load("input.png");
  sam_encode(sam, input_image, cpu);

  image_data object_mask = sam_compute(sam, {320, 240}, cpu);
  image_save(object_mask, "mask.png");
}
```
This shows the high-level API. Internally it is composed of multiple smaller
functions that handle model loading, pre-processing inputs, transferring
data to backend devices, post-processing output, etc. 
These can be used as building blocks for flexible pipelines which integrate
with your existing data sources and infrastructure.

#### UI


## Models

### Segment Anything Model (SAM)

```sh
vision-cli sam -m models/MobileSAM.gguf -i input.png -p 300 200 -o mask.png --composite comp.png
```

### BiRefNet

```sh
vision-cli birefnet -m models/BiRefNet_lite-F16.gguf -i input.png -o mask.png --composite comp.png
```

### MI-GAN

```sh
vision-cli migan -m models/migan_places2_512-F16.gguf -i image.png mask.png -o output.png
```

### ESRGAN

```sh
vision-cli esrgan -m models/4x_foolhardy_Remacrih-F16.gguf -i input.png -o output.png
```


### Converting models

Models need to be converted to GGUF before they can be used. This will also
rearrange or precompute tensors for more optimal inference.

To convert a model, install [uv](https://docs.astral.sh/uv/) and run:
```sh
uv run scripts/convert.py <arch> MyModel.pth -q f16
```
where `<arch>` is one of `sam, birefnet, esrgan, ...`. This will create `models/MyModel-F16.gguf`.

See `convert.py --help` for more options.

## Building

Building requires CMake and a compiler with C++20 support.

**Get the sources**
```sh
git clone https://github.com/Acly/vision.cpp.git --recursive
cd vision.cpp
```

**Configure and build**
```sh
cmake . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build
```

### _(Optional)_ Vulkan

Vulkan GPU support requires the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) to be installed.

```sh
cmake . -B build -D CMAKE_BUILD_TYPE=Release -D VISP_VULKAN=ON
cmake --build build
```

### _(Optional)_ Tests

Run all C++ tests with the following command:
```sh
cd build
ctest
```

Some tests require a Python environment. It can be set up with [uv](https://docs.astral.sh/uv/):
```sh
# Setup venv and install dependencies (once only)
uv sync

# Run only python tests
uv run pytest
```