#pragma once

#include "visp/util.h"

#include <memory>
#include <span>

namespace visp {
using std::span;
struct image_data;
struct image_span;

//
// Image channel formats

enum class image_format {
    // 8-bit normalized integer formats
    rgba_u8,
    bgra_u8,
    argb_u8,
    rgb_u8,
    alpha_u8,

    // 32-bit float formats
    rgba_f32,
    rgb_f32,
    alpha_f32
};

VISP_API int n_channels(image_format);
VISP_API int n_bytes(image_format);
VISP_API bool is_float(image_format);

//
// Image view - read-only, non-owning reference to image data

struct VISP_API image_view {
    i32x2 extent{}; // width, height
    int stride = 0; // row stride in bytes
    image_format format = image_format::rgba_u8;
    void const* data = nullptr;

    image_view() = default;
    image_view(image_span const&);
    image_view(image_data const&);

    image_view(i32x2 extent, image_format, uint8_t const* data);
    image_view(i32x2 extent, image_format, span<uint8_t const> data);

    image_view(i32x2 extent, image_format, float const* data);
    image_view(i32x2 extent, span<float const> data); // alpha_f32
    image_view(i32x2 extent, span<f32x3 const> data); // rgb_f32
    image_view(i32x2 extent, span<f32x4 const> data); // rgba_f32

    span<uint8_t const> as_bytes() const;
    span<float const> as_floats() const;
};

VISP_API int n_channels(image_view const&);
VISP_API int n_pixels(image_view const&);
VISP_API size_t n_bytes(image_view const&);

//
// Image span - mutable, non-owning reference to image data
//
// * can also be passed to functions that expect image_view

struct VISP_API image_span {
    i32x2 extent = {}; // width, height
    int stride = 0;    // row stride in bytes
    image_format format = image_format::rgba_u8;
    void* data = nullptr;

    image_span() = default;
    image_span(image_data&);

    image_span(i32x2 extent, image_format, uint8_t* data);
    image_span(i32x2 extent, image_format, span<uint8_t> data);

    image_span(i32x2 extent, image_format, float* data);
    image_span(i32x2 extent, span<float> data); // alpha_f32
    image_span(i32x2 extent, span<f32x3> data); // rgb_f32
    image_span(i32x2 extent, span<f32x4> data); // rgba_f32

    span<uint8_t> as_bytes() const;
    span<float> as_floats() const;
};

//
// Image data - storage for pixel data
//
// * can also be passed to functions that expect image_view, or image_span (if non-const)

struct image_data {
    i32x2 extent{};
    image_format format = image_format::rgba_u8;
    std::unique_ptr<uint8_t[]> data;
};

// Allocate image data. Memory is not initialized!
VISP_API image_data image_alloc(i32x2 extent, image_format format);

// Set all pixels to zero.
VISP_API void image_clear(image_span const&);

// Load image from file (PNG, JPEG, etc.)
VISP_API image_data image_load(char const* filepath);

// Save image to file (PNG, JPEG, etc.)
VISP_API void image_save(image_view const& img, char const* filepath);

//
// Image algorithms

// Convert uint8 image to float32 image
// * `src` can have any image format, output is converted to format of `dst`
// * applies computation `dst = (src + offset) * scale` to each pixel
// * starts reading `src` at `tile_offset`
// * always writes all of `dst` -- if `src` is smaller the output is padded
VISP_API void image_u8_to_f32(
    image_view const& src,
    image_span const& dst,
    f32x4 offset = f32x4{0.f},
    f32x4 scale = f32x4{1.f},
    i32x2 tile_offset = {0, 0});

VISP_API image_data image_u8_to_f32(
    image_view const& src,
    image_format format,
    f32x4 offset = f32x4{0.f},
    f32x4 scale = f32x4{1.f});

// Convert float32 image to uint8 image
// * applies computation `dst = src * scale + offset` to every channel/pixel
VISP_API void image_f32_to_u8(
    image_view const& src, image_span const& dst, float scale = 1, float offset = 0);

VISP_API image_data image_f32_to_u8(
    image_view const& src, image_format, float scale = 1, float offset = 0);

// Converts an RGB/RGBA image to an alpha mask by keeping only the first channel (red)
VISP_API void image_to_mask(image_view const& src, image_span const& dst);
VISP_API image_data image_to_mask(image_view const& src);

// Write values in `alpha` to the alpha channel of `img`.
VISP_API void image_set_alpha(image_span const& img, image_view const& alpha);

// Resize image to target size with bilinear interpolation
VISP_API void image_scale(image_view const&, i32x2 target, image_span const& dst);
VISP_API image_data image_scale(image_view const&, i32x2 target);

// Box filter with kernel size `2*radius + 1`
VISP_API void image_blur(image_view const& src, image_span const& dst, int radius);

// Erosion filter for masks (minimum value in square neighborhood)
VISP_API void image_erosion(image_view const& src, image_span const& dst, int radius);

// Try to separate foreground and background contribution from pixels at the mask border
// * `img` must be a 4-channel image (RGBA), the alpha channel is not used
// * `mask` must be a single-channel image (alpha mask)
// * result is a 4-channel image with extracted foreground as RGB and alpha set to `mask`
VISP_API image_data image_estimate_foreground(
    image_view const& img, image_view const& mask, int radius = 30);

// Composite foreground and background images using alpha mask: `dst = fg * alpha + bg * (1-alpha)`
VISP_API void image_alpha_composite(
    image_view const& fg, image_view const& bg, image_view const& mask, image_span const& dst);

VISP_API image_data image_alpha_composite(
    image_view const& fg, image_view const& bg, image_view const& mask);

// Compute root-mean-square difference between two images
VISP_API float image_difference_rms(image_view const& a, image_view const& b);

//
// Image tiling - helpers for processing large images in tiles

struct VISP_API tile_layout {
    i32x2 image_extent;
    i32x2 overlap;
    i32x2 n_tiles;
    i32x2 tile_size;

    tile_layout() = default;

    tile_layout(i32x2 extent, int max_tile_size, int overlap, int align = 16);

    i32x2 start(i32x2 coord, i32x2 pad = {0, 0}) const;
    i32x2 end(i32x2 coord, i32x2 pad = {0, 0}) const;
    i32x2 size(i32x2 coord) const;

    int total() const;            // flat number of tiles
    i32x2 coord(int index) const; // flat index -> tile index
};

// Returns layout with same number of tiles mapped to a larger image
VISP_API tile_layout tile_scale(tile_layout const&, int scale);

// Merge a tile into the destination image. Both images must be rgb_f32 format.
// Blends pixels from `tile` and `dst` in overlap regions. `dst` must be all zeros initially.
VISP_API void tile_merge(
    image_view const& tile, image_span const& dst, i32x2 tile_coord, tile_layout const& layout);

} // namespace visp