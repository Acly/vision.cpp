# _vision_.cpp

Computer Vision ML inference in C++

* Self-contained C++ library
* Efficient inference on consumer CPU and GPUs (NVIDIA, AMD, Intel)
* Lightweight deployment on many platforms (Windows, Linux, MacOS)
* Growing number of supported models behind a simple API
* Modular design for full control and implementing your own models

Based on [ggml](https://github.com/ggml-org/ggml) similar to the [llama.cpp](https://github.com/ggml-org/llama.cpp) project.

### Features

| Model                                    | Task                     | Backends    |
| :--------------------------------------- | :----------------------- | :---------- |
| [**MobileSAM**](#mobilesam)              | Promptable segmentation  | CPU, Vulkan |
| [**BiRefNet**](#birefnet)                | Dichotomous segmentation | CPU, Vulkan |
| [**Depth-Anything**](#depth-anything-v2) | Depth estimation         | CPU, Vulkan |
| [**MI-GAN**](#mi-gan)                    | Inpainting               | CPU, Vulkan |
| [**ESRGAN**](#real-esrgan)               | Super-resolution         | CPU, Vulkan |
| [_Implement a model [**Guide**]_](docs/model-implementation-guide.md) | | |

**Backbones:** SWIN (v1), DINO (v2), TinyViT

## Get Started

Get the library and executables:
* Download a [release package](https://github.com/Acly/vision.cpp/releases) and extract it,
* or [build from source](#building).

### Example: Select an object in an image

Let's use MobileSAM to generate a segmentation mask of the plushy on the right
by passing in a box describing its approximate location.

<img width="400" height="256" alt="Example image showing box prompt at pixel location (420, 120) - (650, 430), and the output mask" src="https://github.com/user-attachments/assets/0b90ad96-c7d2-4c4c-b028-699433cef704" />

You can download the model and input image here: [MobileSAM-F16.gguf](https://huggingface.co/Acly/MobileSAM-GGUF/resolve/main/MobileSAM-F16.gguf) | [input.jpg](docs/media/input.jpg)


#### CLI

Find the `vision-cli` executable in the `bin` folder and run it to generate the mask:

```sh
vision-cli -m MobileSAM-F16.gguf -i input.jpg -p 420 120 650 430 -o mask.png
```
Pass `--composite output.png` to composite input and mask. Use `--help` for more options.

#### API

```c++
#include <visp/vision.h>
using namespace visp;

void main() {
  backend_device cpu = backend_init(backend_type::cpu);
  sam_model sam = sam_load_model("MobileSAM-F16.gguf", cpu);
  
  image_data input_image = image_load("input.jpg");
  sam_encode(sam, input_image);

  image_data object_mask = sam_compute(sam, box_2d{{420, 120}, {650, 320}});
  image_save(object_mask, "mask.png");
}
```
This shows the high-level API. Internally it is composed of multiple smaller
functions that handle model loading, pre-processing inputs, transferring data to
backend devices, post-processing output, etc. These can be used as building
blocks for flexible functions which integrate with your existing data sources
and infrastructure.



## Models

#### MobileSAM

<img width="400" height="256" alt="example-sam" src="https://github.com/user-attachments/assets/9c0fe151-9990-4bb1-b954-7caff560b110" />

[Model download](https://huggingface.co/Acly/MobileSAM-GGUF/tree/main) | [Paper (arXiv)](https://arxiv.org/pdf/2306.14289.pdf) | [Repository (GitHub)](https://github.com/ChaoningZhang/MobileSAM) | [Segment-Anything-Model](https://segment-anything.com/) | License: Apache-2

```sh
vision-cli sam -m MobileSAM-F16.gguf -i input.png -p 300 200 -o mask.png --composite comp.png
```

#### BiRefNet

<img width="400" height="256" alt="example-birefnet" src="https://github.com/user-attachments/assets/6fce086d-cb89-4717-92a6-9f4a20532b3c" />

[Model download](https://huggingface.co/Acly/BiRefNet-GGUF/tree/main) | [Paper (arXiv)](https://arxiv.org/pdf/2401.03407) | [Repository (GitHub)](https://github.com/ZhengPeng7/BiRefNet) | License: MIT

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i input.png -o mask.png --composite comp.png
```

#### Depth-Anything V2

<img width="400" height="256" alt="example-depth-anything" src="https://github.com/user-attachments/assets/62bde481-b898-4c46-a298-644198716953" />

[Model download](https://huggingface.co/Acly/Depth-Anything-V2-GGUF/tree/main) | [Paper (arXiv)](https://arxiv.org/abs/2406.09414) | [Repository (GitHub)](https://github.com/DepthAnything/Depth-Anything-V2) | License: Apache-2 / CC-BY-NC-4

```sh
vision-cli depth-anything -m Depth-Anything-V2-Small-F16.gguf -i input.png -o depth.png
```

#### MI-GAN

<img width="400" height="256" alt="example-migan" src="https://github.com/user-attachments/assets/cadf1994-7677-4822-94e5-a2ee6c07621f" />

[Model download](https://huggingface.co/Acly/MIGAN-GGUF/tree/main) | [Paper (thecvf.com)](https://openaccess.thecvf.com/content/ICCV2023/papers/Sargsyan_MI-GAN_A_Simple_Baseline_for_Image_Inpainting_on_Mobile_Devices_ICCV_2023_paper.pdf) | [Repository (GitHub)](https://github.com/Picsart-AI-Research/MI-GAN) | License: MIT

```sh
vision-cli migan -m MIGAN-512-places2-F16.gguf -i image.png mask.png -o output.png
```

#### Real-ESRGAN

<img width="400" height="256" alt="example-esrgan" src="https://github.com/user-attachments/assets/a41312d6-836c-4b11-ab5d-2e299ffee10c" />

[Model download](https://huggingface.co/Acly/Real-ESRGAN-GGUF) | [Paper (arXiv)](https://arxiv.org/abs/2107.10833) | [Repository (GitHub)](https://github.com/xinntao/Real-ESRGAN) | License: BSD-3-Clause

```sh
vision-cli esrgan -m ESRGAN-4x-foolhardy_Remacri-F16.gguf -i input.png -o output.png
```


### Converting models

Models need to be converted to GGUF before they can be used. This will also
rearrange or precompute tensors for more optimal inference.

To convert a model, install [uv](https://docs.astral.sh/uv/) and run:
```sh
uv run scripts/convert.py <arch> MyModel.pth
```
where `<arch>` is one of `sam, birefnet, esrgan, ...`.

This will create `models/MyModel.gguf`. See `convert.py --help` for more options.

## Building

Building requires CMake and a compiler with C++20 support.

**Get the sources**
```sh
git clone https://github.com/Acly/vision.cpp.git --recursive
cd vision.cpp
```

**Configure and build**
```sh
cmake . -B build
cmake --build build --config Release
```

### Vulkan _(Optional)_

Building with Vulkan GPU support requires the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) to be installed.

```sh
cmake . -B build -D VISP_VULKAN=ON
```

### Tests _(Optional)_

Build with `-DVISP_TESTS=ON`. Run all C++ tests with the following command:
```sh
cd build
ctest -C Release
```

Some tests require a Python environment. It can be set up with [uv](https://docs.astral.sh/uv/):
```sh
# Setup venv and install dependencies (once only)
uv sync --dev

# Run python tests
uv run pytest
```

## Performance

Performance optimization is an ongoing process. The aim is to be in the same ballpark
as other frameworks for inference speed, but with:
* much faster initialization and model loading time (<100 ms)
* lower memory overhead
* tiny deployment size (<5 MB for CPU, +30 MB for GPU)

### Inference speed

* CPU: AMD Ryzen 5 5600X (6 cores)
* GPU: NVIDIA GeForce RTX 4070

#### MobileSAM, 1024x1024

|      |      | _vision.cpp_ | PyTorch | ONNX Runtime |
| :--- | :--- | -----------: | ------: | -----------: |
| cpu  | f32  |       669 ms |  601 ms |       805 ms |
| gpu  | f16  |        19 ms |   16 ms |              |

#### BiRefNet, 1024x1024

| Model |      |      | _vision.cpp_ |  PyTorch | ONNX Runtime |
| :---- | :--- | :--- | -----------: | -------: | -----------: |
| Full  | cpu  | f32  |     16333 ms | 18290 ms |              |
| Full  | gpu  | f16  |       208 ms |   190 ms |              |
| Lite  | cpu  | f32  |      4505 ms | 10900 ms |      6978 ms |
| Lite  | gpu  | f16  |        85 ms |    84 ms |              |

#### Depth-Anything, 518x714

| Model |      |      | _vision.cpp_ | PyTorch |
| :---- | :--- | :--- | -----------: | ------: |
| Small | gpu  | f16  |        11 ms |   10 ms |
| Base  | gpu  | f16  |        24 ms |   22 ms |

#### MI-GAN, 512x512

| Model       |      |      | _vision.cpp_ | PyTorch |
| :---------- | :--- | :--- | -----------: | ------: |
| 512-places2 | cpu  | f32  |       523 ms |  637 ms |
| 512-places2 | gpu  | f16  |        21 ms |   17 ms |

#### Setup

* vision.cpp: using vision-bench, GPU via Vulkan, eg. `vision-bench -m sam`
* PyTorch: v2.7.1+cu128, eager eval, GPU via CUDA, average n iterations after warm-up

## Dependencies (integrated)

* [ggml](https://github.com/ggml-org/ggml) - ML tensor library | MIT
* [stb-image](https://github.com/nothings/stb) - Image load/save/resize | Public Domain
* [fmt](https://github.com/fmtlib/fmt) - String formatting _(only if compiler doesn't support &lt;format&gt;)_ | MIT
