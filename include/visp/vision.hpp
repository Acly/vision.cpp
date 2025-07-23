//
// Vision.cpp
//
// Library for inference of computer vision neural networks.
//
// Overview
// --------
//
// Vision.cpp comes in 3 main headers:
//
// visp/image.hpp
//
//   Defines structures to store and reference pixel data. Supports loading, saving and
//   common processing of images. Most tasks take an `image_view` as input, which
//   is a non-owning reference to external pixel data. Output is returned as
//   `image_data` (allocated by the library) or written to an `image_span`.
//
// visp/ml.hpp
//
//   Contains ML infrastructure shared between all models: loading weights,
//   transferring data between backend devices (eg. GPU), and executing
//   compute graphs. Most of these are thin convenience wrappers around GGML.
//   Alternatively you can use GGML directly for greater flexibility.
//
// visp/vision.hpp (this file)
//
//   Provides a high-level API to run inference on various vision models for
//   common tasks. These operations are built for simplicity and don't provide
//   a lot of options. Rather, you will find below each operation it is split into
//   several steps, which can be used to build more flexible pipelines.
//
// Basic Use
// ---------
//
// Evaluating a model on an image with the high-level API usually looks like this:
//
// ```
// backend_device backend = backend_init();
// image_view input  = image_view({width, height}, image_format::rgb_u8, pixel_data_ptr);
// ARCH_model model  = ARCH_load_model("path/to/model.gguf", backend);
// image_data output = ARCH_compute(model, input);
// ```
//
// ARCH referes to the model architecture, such as `sam_model`, `birefnet_model`, etc.
// The `compute` function can be called repeatedly with different inputs.
//
// Advanced Use
// ------------
//
// Internally running the model is split into several steps:
// 1. Load the model weights from a GGUF file.
// 2. Allocate storage on the backend device and transfer the weights.
// 3. Detect model hyperparameters and precompute required buffers.
// 4. Build a compute graph for the model architecture.
// 5. Allocate storage for input, output and intermediate tensors on the backend device.
// 6. Pre-process the image and transfer it to the backend device.
// 7. Run the compute graph.
// 8. Transfer the output to the host and post-process it.
//
// You can run all steps individually in order to customize the pipeline. Check the
// implementation of the high-level API functions to get started.
//
// This allows to:
// * load model weights from a different source
// * control exactly when allocation happens
// * offload weights to host memory when not needed
// * implement pre-processing and post-processing for custom image formats
// * integrate model compute graphs into your own GGML graphs
// * ... etc.
//

#pragma once

#include "visp/image.hpp"
#include "visp/ml.hpp"
#include "visp/util.hpp"

#include <array>
#include <span>

namespace visp {

//
// Mobile SAM - image segmentation with prompt (point or box)

struct sam_model;

struct box_2d {
    i32x2 top_left;
    i32x2 bottom_right;
};

// Loads a SAM model from GGUF file onto the backend device.
// * only supports MobileSAM (TinyViT) for now
VISP_API sam_model sam_load_model(char const* filepath, backend_device const&);

// Creates an image embedding from RGB input, required for subsequent `sam_compute` calls.
VISP_API void sam_encode(sam_model&, image_view image);

// Computes a segmentation mask (alpha image) for an object in the image.
// * takes either a point, ie. a pixel location with origin (0, 0) in the top left
// * or a bounding box which contains the object
VISP_API image_data sam_compute(sam_model&, i32x2 point);
VISP_API image_data sam_compute(sam_model&, box_2d box);

// --- SAM pipeline

struct sam_params {
    int image_size = 1024;
    int mask_size = 256;
};

struct sam_prediction {
    tensor masks;
    tensor iou;
};

VISP_API image_data sam_process_input(image_view image, sam_params const&);
VISP_API f32x4 sam_process_point(i32x2 point, i32x2 image_extent, sam_params const&);
VISP_API f32x4 sam_process_box(box_2d box, i32x2 image_extent, sam_params const&);

VISP_API tensor sam_encode_image(model_ref, tensor image, sam_params const&);
VISP_API tensor sam_encode_points(model_ref, tensor coords);
VISP_API tensor sam_encode_box(model_ref, tensor coords);

VISP_API sam_prediction sam_predict_mask(model_ref m, tensor image_embed, tensor prompt_embed);

VISP_API image_data sam_process_mask(
    std::span<float const> mask_data, int mask_index, i32x2 target_extent, sam_params const&);

//
// BiRefNet - dichotomous image segmentation (background removal)

struct birefnet_model;

// Loads a BiRefNet model from GGUF file onto the backend device.
// * supports BiRefNet, BiRefNet_lite, BiRefNet_Matting variants at 1024px resolution
VISP_API birefnet_model birefnet_load_model(char const* filepath, backend_device const&);

// Takes RGB input and computes an alpha mask with foreground as 1.0 and background as 0.0.
VISP_API image_data birefnet_compute(birefnet_model&, image_view image);

// --- BiRefNet pipeline

struct birefnet_params {
    int image_size = 1024;
    swin_params encoder;
};

using birefnet_buffers = std::array<tensor_data, swin_params::n_layers + 2>;

VISP_API birefnet_params birefnet_detect_params(model_ref);
VISP_API birefnet_buffers birefnet_precompute(model_ref, birefnet_params const&);

VISP_API image_data birefnet_process_input(image_view, birefnet_params const&);
VISP_API image_data birefnet_process_output(
    std::span<float const> output_data, i32x2 target_extent, birefnet_params const&);

VISP_API tensor birefnet_predict(model_ref, tensor image, birefnet_params const&);

//
// MI-GAN - image inpainting

struct migan_model;

// Loads a MI-GAN model from GGUF file onto the backend device.
// * supports variants at 256px or 512px resolution
VISP_API migan_model migan_load_model(char const* filepath, backend_device const&);

// Fills pixels in the input image where the mask is 1.0 with new content.
VISP_API image_data migan_compute(migan_model&, image_view image, image_view mask);

// --- MI-GAN pipeline

struct migan_params {
    int resolution = 256;
    bool invert_mask = false;
};

VISP_API migan_params migan_detect_params(model_ref m);

VISP_API image_data migan_process_input(image_view image, image_view mask, migan_params const&);
VISP_API image_data migan_process_output(
    std::span<float const> data, i32x2 extent, migan_params const&);

VISP_API tensor migan_generate(model_ref, tensor image, migan_params const&);

//
// ESRGAN - image super-resolution

struct esrgan_model;

// Loads an ESRGAN model from GGUF file onto the backend device.
// * supports ESRGAN, RealESRGAN variants with flexible scale and number of blocks
// * currently does not spport RealESRGAN+ (plus) models or those which use pixel shuffle
VISP_API esrgan_model esrgan_load_model(char const* filepath, backend_device const&);

// Upscales the input image by the model's scale factor. Uses tiling for large inputs.
VISP_API image_data esrgan_compute(esrgan_model&, image_view image);

// --- ESRGAN pipeline

struct esrgan_params {
    int scale = 4;
    int n_blocks = 23;
};

VISP_API esrgan_params esrgan_detect_params(model_ref);
VISP_API int esrgan_estimate_graph_size(esrgan_params const&);

VISP_API tensor esrgan_generate(model_ref, tensor image, esrgan_params const&);

//
// Implementation

// internal
struct sam_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    sam_params params;

    compute_graph encoder;
    i32x2 image_extent{};
    tensor input_image = nullptr;
    tensor output_embed = nullptr;

    compute_graph decoder;
    tensor input_embed = nullptr;
    tensor input_prompt = nullptr;
    sam_prediction output = {};
    bool is_point_prompt = true;
};

// internal
struct birefnet_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    birefnet_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct migan_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    migan_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct esrgan_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    esrgan_params params;

    compute_graph graph;
    i32x2 tile_size{};
    tensor input = nullptr;
    tensor output = nullptr;
};

} // namespace visp