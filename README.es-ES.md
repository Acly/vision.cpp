

# _vision_.cpp

Inferencia de IA/ML para Visión por Computador en C++

* Biblioteca C++ autocontenida
* Inferencia eficiente en CPUs y GPUs de consumo (NVIDIA, AMD, Intel)
* Despliegue ligero en múltiples plataformas (Windows, Linux, MacOS)
* Número creciente de modelos compatibles detrás de una API simple
* Diseño modular para control total e implementar tus propios modelos

Basado en [ggml](https://github.com/ggml-org/ggml) similar al proyecto [llama.cpp](https://github.com/ggml-org/llama.cpp).

### Características

| Modelo                                    | Tarea                     | Backends    |
| :--------------------------------------- | :----------------------- | :---------- |
| [**MobileSAM**](#mobilesam)              | Segmentación con prompts  | CPU, Vulkan |
| [**BiRefNet**](#birefnet)                | Segmentación dicotómica | CPU, Vulkan |
| [**Depth-Anything**](#depth-anything-v2) | Estimación de profundidad         | CPU, Vulkan |
| [**MI-GAN**](#mi-gan)                    | Inpainting               | CPU, Vulkan |
| [**ESRGAN**](#real-esrgan)               | Super-resolución         | CPU, Vulkan |
| [_Implementar un modelo [**Guía**]_](docs/model-implementation-guide.md) | | |

**Backbones:** SWIN (v1), DINO (v2), TinyViT

## Primeros pasos

Obtén la biblioteca y los ejecutables:
* Descarga un [paquete de lanzamiento](https://github.com/Acly/vision.cpp/releases) y extráelo,
* o [compílalo desde el código fuente](#building).

### Ejemplo: Seleccionar un objeto en una imagen

Utilicemos MobileSAM para generar una máscara de segmentación del peluche a la derecha, pasando un recuadro que describa su ubicación aproximada.

<img width="400" height="256" alt="Example image showing box prompt at pixel location (420, 120) - (650, 430), and the output mask" src="https://github.com/user-attachments/assets/0b90ad96-c7d2-4c4c-b028-699433cef704" />

Puedes descargar el modelo y la imagen de entrada aquí: [MobileSAM-F16.gguf](https://huggingface.co/Acly/MobileSAM-GGUF/resolve/main/MobileSAM-F16.gguf) | [input.jpg](docs/media/input.jpg)


#### CLI

Busca el ejecutable `vision-cli` en la carpeta `bin` y ejecútalo para generar la máscara:

```sh
vision-cli -m MobileSAM-F16.gguf -i input.jpg -p 420 120 650 430 -o mask.png
```
Para combinar la entrada y la máscara, pasa `--composite output.png`. Usa `--help` para ver más opciones.

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
Esto muestra la API de alto nivel. Internamente, está compuesta por múltiples funciones más pequeñas que manejan la carga del modelo, el preprocesamiento de entradas, la transferencia de datos a los dispositivos backend, el postprocesamiento de la salida, etc. Estas pueden usarse como bloques de construcción para funciones flexibles que se integren con tus fuentes de datos e infraestructura existentes.



## Modelos

#### MobileSAM

<img width="400" height="256" alt="example-sam" src="https://github.com/user-attachments/assets/9c0fe151-9990-4bb1-b954-7caff560b110" />

[Descarga del modelo](https://huggingface.co/Acly/MobileSAM-GGUF/tree/main) | [Artículo (arXiv)](https://arxiv.org/pdf/2306.14289.pdf) | [Repositorio (GitHub)](https://github.com/ChaoningZhang/MobileSAM) | [Segment-Anything-Model](https://segment-anything.com/) | Licencia: Apache-2

```sh
vision-cli sam -m MobileSAM-F16.gguf -i input.png -p 300 200 -o mask.png --composite comp.png
```

#### BiRefNet

<img width="400" height="256" alt="example-birefnet" src="https://github.com/user-attachments/assets/6fce086d-cb89-4717-92a6-9f4a20532b3c" />

[Descarga del modelo](https://huggingface.co/Acly/BiRefNet-GGUF/tree/main) | [Artículo (arXiv)](https://arxiv.org/pdf/2401.03407) | [Repositorio (GitHub)](https://github.com/ZhengPeng7/BiRefNet) | Licencia: MIT

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i input.png -o mask.png --composite comp.png
```

#### Depth-Anything V2

<img width="400" height="256" alt="example-depth-anything" src="https://github.com/user-attachments/assets/62bde481-b898-4c46-a298-644198716953" />

[Descarga del modelo](https://huggingface.co/Acly/Depth-Anything-V2-GGUF/tree/main) | [Artículo (arXiv)](https://arxiv.org/abs/2406.09414) | [Repositorio (GitHub)](https://github.com/DepthAnything/Depth-Anything-V2) | Licencia: Apache-2 / CC-BY-NC-4

```sh
vision-cli depth-anything -m Depth-Anything-V2-Small-F16.gguf -i input.png -o depth.png
```

#### MI-GAN

<img width="400" height="256" alt="example-migan" src="https://github.com/user-attachments/assets/cadf1994-7677-4822-94e5-a2ee6c07621f" />

[Descarga del modelo](https://huggingface.co/Acly/MIGAN-GGUF/tree/main) | [Artículo (thecvf.com)](https://openaccess.thecvf.com/content/ICCV2023/papers/Sargsyan_MI-GAN_A_Simple_Baseline_for_Image_Inpainting_on_Mobile_Devices_ICCV_2023_paper.pdf) | [Repositorio (GitHub)](https://github.com/Picsart-AI-Research/MI-GAN) | Licencia: MIT

```sh
vision-cli migan -m MIGAN-512-places2-F16.gguf -i image.png mask.png -o output.png
```

#### Real-ESRGAN

<img width="400" height="256" alt="example-esrgan" src="https://github.com/user-attachments/assets/a41312d6-836c-4b11-ab5d-2e299ffee10c" />

[Descarga del modelo](https://huggingface.co/Acly/Real-ESRGAN-GGUF) | [Artículo (arXiv)](https://arxiv.org/abs/2107.10833) | [Repositorio (GitHub)](https://github.com/xinntao/Real-ESRGAN) | Licencia: BSD-3-Clause

```sh
vision-cli esrgan -m ESRGAN-4x-foolhardy_Remacri-F16.gguf -i input.png -o output.png
```


### Conversión de modelos

Los modelos deben convertirse a GGUF antes de poder usarse. Esto también reorganizará o precomputará tensores para una inferencia más óptima.

Para convertir un modelo, instala [uv](https://docs.astral.sh/uv/) y ejecuta:
```sh
uv run scripts/convert.py <arch> MyModel.pth
```
donde `<arch>` es uno de `sam, birefnet, esrgan, ...`.

Esto creará `models/MyModel.gguf`. Consulta `convert.py --help` para ver más opciones.

## Compilación

La compilación requiere CMake y un compilador con soporte para C++20.

**Obtener el código fuente**
```sh
git clone https://github.com/Acly/vision.cpp.git --recursive
cd vision.cpp
```

**Configurar y compilar**
```sh
cmake . -B build
cmake --build build --config Release
```

### Vulkan _(Opcional)_

Compilar con soporte para GPU Vulkan requiere que esté instalado el [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/).

```sh
cmake . -B build -D VISP_VULKAN=ON
```

### Pruebas _(Opcional)_

Compila con `-DVISP_TESTS=ON`. Ejecuta todas las pruebas de C++ con el siguiente comando:
```sh
cd build
ctest -C Release
```

Algunas pruebas requieren un entorno de Python. Puede configurarse con [uv](https://docs.astral.sh/uv/):
```sh
# Setup venv and install dependencies (once only)
uv sync --dev

# Run python tests
uv run pytest
```

## Rendimiento

La optimización del rendimiento es un proceso continuo. El objetivo es estar en el mismo rango que otros frameworks en velocidad de inferencia, pero con:
* tiempos de inicialización y carga de modelos mucho más rápidos (<100 ms)
* menor sobrecarga de memoria
* tamaño de despliegue mínimo (<5 MB para CPU, +30 MB para GPU)

### Velocidad de inferencia

* CPU: AMD Ryzen 5 5600X (6 núcleos)
* GPU: NVIDIA GeForce RTX 4070

#### MobileSAM, 1024x1024

|      |      | _vision.cpp_ | PyTorch | ONNX Runtime |
| :--- | :--- | -----------: | ------: | -----------: |
| cpu  | f32  |       669 ms |  601 ms |       805 ms |
| gpu  | f16  |        19 ms |   16 ms |              |

#### BiRefNet, 1024x1024

| Modelo |      |      | _vision.cpp_ |  PyTorch | ONNX Runtime |
| :---- | :--- | :--- | -----------: | -------: | -----------: |
| Full  | cpu  | f32  |     16333 ms | 18290 ms |              |
| Full  | gpu  | f16  |       208 ms |   190 ms |              |
| Lite  | cpu  | f32  |      4505 ms | 10900 ms |      6978 ms |
| Lite  | gpu  | f16  |        85 ms |    84 ms |              |

#### Depth-Anything, 518x714

| Modelo |      |      | _vision.cpp_ | PyTorch |
| :---- | :--- | :--- | -----------: | ------: |
| Small | gpu  | f16  |        11 ms |   10 ms |
| Base  | gpu  | f16  |        24 ms |   22 ms |

#### MI-GAN, 512x512

| Modelo       |      |      | _vision.cpp_ | PyTorch |
| :---------- | :--- | :--- | -----------: | ------: |
| 512-places2 | cpu  | f32  |       523 ms |  637 ms |
| 512-places2 | gpu  | f16  |        21 ms |   17 ms |

#### Configuración

* vision.cpp: usando vision-bench, GPU vía Vulkan, p. ej. `vision-bench -m sam`
* PyTorch: v2.7.1+cu128, evaluación eager, GPU vía CUDA, promedio de n iteraciones después del calentamiento (warm-up)

## Dependencias (integradas)

* [ggml](https://github.com/ggml-org/ggml) - Biblioteca de tensores para ML | MIT
* [stb-image](https://github.com/nothings/stb) - Carga/guardado/redimensionado de imágenes | Dominio Público
* [fmt](https://github.com/fmtlib/fmt) - Formateo de cadenas _(solo si el compilador no soporta &lt;format&gt;)_ | MIT
