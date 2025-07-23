#include "visp/image.hpp"
#include "image-impl.hpp"
#include "util/math.hpp"
#include "util/string.hpp"

#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>

#include <algorithm>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace visp {
using std::clamp;

//
// image format

int n_channels(image_format format) {
    switch (format) {
        case image_format::rgba_u8:
        case image_format::bgra_u8:
        case image_format::argb_u8:
        case image_format::rgba_f32: return 4;
        case image_format::rgb_u8:
        case image_format::rgb_f32: return 3;
        case image_format::alpha_u8:
        case image_format::alpha_f32: return 1;
        default: ASSERT(false, "unknown image format"); return 0;
    }
}

int n_bytes(image_format format) {
    return n_channels(format) * (is_float(format) ? 4 : 1);
}

bool is_float(image_format format) {
    return int(format) >= int(image_format::rgba_f32);
}

i32x4 get_channel_map(image_format format) {
    switch (format) {
        case image_format::bgra_u8: return {2, 1, 0, 3};
        case image_format::argb_u8: return {1, 2, 3, 0};
        case image_format::alpha_u8:
        case image_format::alpha_f32: return {0, 0, 0, 0};
        case image_format::rgb_u8:
        case image_format::rgb_f32: return {0, 1, 2, 0};
        default: return {0, 1, 2, 3}; // rgba
    }
}

int alpha_channel(image_format format) {
    switch (format) {
        case image_format::bgra_u8: return 3;
        case image_format::argb_u8: return 0;
        case image_format::alpha_u8:
        case image_format::alpha_f32: return 0;
        case image_format::rgb_u8:
        case image_format::rgb_f32: return -1; // no alpha channel
        default: return 3;                     // rgba
    }
}

//
// image view

image_view::image_view(i32x2 extent, image_format format, uint8_t const* data)
    : extent(extent), stride(extent[0] * n_bytes(format)), format(format), data(data) {}

image_view::image_view(i32x2 extent, image_format format, span<uint8_t const> data)
    : image_view(extent, format, data.data()) {
    ASSERT(data.size() >= n_bytes(*this));
}

image_view::image_view(i32x2 extent, image_format format, float const* data)
    : extent(extent), stride(extent[0] * n_bytes(format)), format(format), data(data) {}

image_view::image_view(i32x2 extent, span<float const> data)
    : image_view(extent, image_format::alpha_f32, data.data()) {
    ASSERT(data.size() >= size_t(n_pixels(*this)));
}

image_view::image_view(i32x2 extent, span<f32x3 const> data)
    : image_view(extent, image_format::rgb_f32, &data[0][0]) {
    ASSERT(data.size() >= size_t(n_pixels(*this)));
}

image_view::image_view(i32x2 extent, span<f32x4 const> data)
    : image_view(extent, image_format::rgba_f32, &data[0][0]) {
    ASSERT(data.size() >= size_t(n_pixels(*this)));
}

image_view::image_view(image_data const& img)
    : image_view(img.extent, img.format, img.data.get()) {}

image_view::image_view(image_span const& o)
    : extent(o.extent), stride(o.stride), format(o.format), data(o.data) {}

span<uint8_t const> image_view::as_bytes() const {
    return {reinterpret_cast<uint8_t const*>(data), n_bytes(*this)};
}

span<float const> image_view::as_floats() const {
    ASSERT(is_float(format));
    return {reinterpret_cast<float const*>(data), n_bytes(*this) / sizeof(float)};
}

int n_channels(image_view const& img) {
    return n_channels(img.format);
}

int n_pixels(image_view const& img) {
    return img.extent[0] * img.extent[1];
}

size_t n_bytes(image_view const& img) {
    return size_t(img.extent[1]) * img.stride;
}

//
// image span

image_span::image_span(image_data& img) : image_span(img.extent, img.format, img.data.get()) {}

image_span::image_span(i32x2 extent, image_format format, uint8_t* data)
    : extent(extent), stride(extent[0] * n_bytes(format)), format(format), data(data) {}

image_span::image_span(i32x2 extent, image_format format, span<uint8_t> data)
    : image_span(extent, format, data.data()) {
    ASSERT(data.size() >= n_bytes(*this));
}
image_span::image_span(i32x2 extent, image_format format, float* data)
    : extent(extent), stride(extent[0] * n_bytes(format)), format(format), data(data) {}

image_span::image_span(i32x2 extent, span<float> data)
    : image_span(extent, image_format::alpha_f32, data.data()) {
    ASSERT(data.size() >= size_t(n_pixels(*this)));
}

image_span::image_span(i32x2 extent, span<f32x3> data)
    : image_span(extent, image_format::rgb_f32, &data[0][0]) {
    ASSERT(data.size() >= size_t(n_pixels(*this)));
}

image_span::image_span(i32x2 extent, span<f32x4> data)
    : image_span(extent, image_format::rgba_f32, &data[0][0]) {
    ASSERT(data.size() >= size_t(n_pixels(*this)));
}

span<uint8_t> image_span::as_bytes() const {
    return {reinterpret_cast<uint8_t*>(data), n_bytes(*this)};
}

span<float> image_span::as_floats() const {
    ASSERT(is_float(format));
    return {reinterpret_cast<float*>(data), n_bytes(*this) / sizeof(float)};
}

//
// image data

image_data image_alloc(i32x2 extent, image_format format) {
    size_t size = extent[0] * extent[1] * n_bytes(format);
    // TODO rgba_f32 allocation should be 16-byte aligned
    return image_data{extent, format, std::unique_ptr<uint8_t[]>(new uint8_t[size])};
}

image_format image_format_from_channels(int n_channels) {
    switch (n_channels) {
        case 1: return image_format::alpha_u8;
        case 3: return image_format::rgb_u8;
        case 4: return image_format::rgba_u8;
        default: ASSERT(false, "Invalid number of channels");
    }
    return image_format::rgba_u8;
}

image_data image_load(char const* filepath) {
    i32x2 extent = {0, 0};
    int channels = 0;
    uint8_t* pixels = stbi_load(filepath, &extent[0], &extent[1], &channels, 0);
    if (!pixels) {
        throw except("Failed to load image {}: {}", filepath, stbi_failure_reason());
    }
    image_format format = image_format_from_channels(channels);
    return image_data{extent, format, std::unique_ptr<uint8_t[]>(pixels)};
}

void image_save(image_view const& img, char const* filepath) {
    ASSERT(img.extent[0] > 0 && img.extent[1] > 0);
    
    if (!(img.format == image_format::alpha_u8 || img.format == image_format::rgb_u8 ||
          img.format == image_format::rgba_u8)) {
        throw except("Unsupported image format [{}]", int(img.format));
    }
    int comp = n_channels(img.format);
    if (!stbi_write_png(
            filepath, img.extent[0], img.extent[1], comp, img.data, img.extent[0] * comp)) {
        throw except("Failed to save image {}", filepath);
    }
}

//
// image conversion (uint8 <-> float32)

template <typename Src, typename Dst>
void convert(
    image_source<Src> src, image_target<Dst> dst, f32x4 offset, f32x4 scale, i32x2 tile_offset) {

    for (int y = 0; y < dst.extent[1]; ++y) {
        for (int x = 0; x < dst.extent[0]; ++x) {
            i32x2 i = {x, y};
            i32x2 i_src = min(i + tile_offset, src.extent - i32x2{1, 1});
            dst.store(i, (src.load(i_src) + offset) * scale);
        }
    }
}

void image_u8_to_f32(
    image_view const& src, image_span const& dst, f32x4 offset, f32x4 scale, i32x2 tile_offset) {

    ASSERT(!is_float(src.format) && is_float(dst.format));

    switch (n_channels(dst)) {
        case 1: convert<uint8_t, float>(src, dst, offset, scale, tile_offset); break;
        case 3:
            switch (n_channels(src)) {
                case 3: convert<u8x3, f32x3>(src, dst, offset, scale, tile_offset); break;
                case 4: convert<u8x4, f32x3>(src, dst, offset, scale, tile_offset); break;
            }
            break;
        case 4:
            switch (n_channels(src)) {
                case 3: convert<u8x3, f32x4>(src, dst, offset, scale, tile_offset); break;
                case 4: convert<u8x4, f32x4>(src, dst, offset, scale, tile_offset); break;
            }
            break;
        default: ASSERT(false, "Number of channels in source and destination or not compatible");
    }
}

image_data image_u8_to_f32(image_view const& src, image_format format, f32x4 offset, f32x4 scale) {
    image_data dst = image_alloc(src.extent, format);
    image_u8_to_f32(src, dst, offset, scale, i32x2{0, 0});
    return dst;
}

template <typename Src, typename Dst>
void convert2(image_source<Src> src, image_target<Dst> dst, f32x4 offset, f32x4 scale) {
    int n = n_pixels(dst);
    for (int i = 0; i < n; ++i) {
        dst.store(i, src.load(i) * scale + offset);
    }
}

void image_f32_to_u8(image_view const& src, image_span const& dst, float scale, float offset) {
    ASSERT(src.extent == dst.extent);
    ASSERT(is_float(src.format) && !is_float(dst.format));

    f32x4 s = {scale, scale, scale, scale};
    f32x4 o = {offset, offset, offset, offset};

    switch (n_channels(dst)) {
        case 1: convert2<float, uint8_t>(src, dst, o, s); break;
        case 4:
            switch (n_channels(src)) {
                case 3: convert2<f32x3, u8x4>(src, dst, o, s); break;
                case 4: convert2<f32x4, u8x4>(src, dst, o, s); break;
            }
            break;
        default: ASSERT(false, "Number of channels in source and destination or not compatible");
    }
}

image_data image_f32_to_u8(image_view const& src, image_format format, float scale, float offset) {
    image_data dst = image_alloc(src.extent, format);
    image_f32_to_u8(src, dst, scale, offset);
    return dst;
}

void image_to_mask(image_view const& src, image_span const& dst) {
    ASSERT(src.extent == dst.extent);
    ASSERT(dst.format == image_format::alpha_u8);

    int n = n_pixels(dst);
    int chan = n_channels(src);
    span<uint8_t const> src_data = src.as_bytes();
    span<uint8_t> dst_data = dst.as_bytes();

    for (int i = 0; i < n; ++i) {
        dst_data[i] = src_data[i * chan];
    }
}

image_data image_to_mask(image_view const& src) {
    image_data dst = image_alloc(src.extent, image_format::alpha_u8);
    image_to_mask(src, dst);
    return dst;
}

void image_set_alpha(image_span const& img, image_view const& alpha) {
    ASSERT(img.extent == alpha.extent);
    ASSERT(!is_float(img.format) && n_channels(img) == 4);
    ASSERT(alpha.format == image_format::alpha_u8);

    int n = n_pixels(img);
    int chan = get_channel_map(img.format)[3]; // alpha channel index
    image_source<uint8_t> src(alpha);
    image_target<u8x4> dst(img);

    for (int i = 0; i < n; ++i) {
        dst[i][chan] = src[i];
    }
}

//
// image algorithms

void image_scale(image_view const& img, i32x2 target, image_span const& dst) {
    ASSERT(img.stride >= img.extent[0] * n_channels(img));

    int result;
    if (is_float(img.format)) {
        result = stbir_resize_float_generic(
            (float const*)img.data, img.extent[0], img.extent[1], img.stride, //
            (float*)dst.data, target[0], target[1], 0,                        //
            n_channels(img), alpha_channel(img.format), 0, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT,
            STBIR_COLORSPACE_LINEAR, nullptr);
    } else {
        result = stbir_resize_uint8_generic(
            (uint8_t const*)img.data, img.extent[0], img.extent[1], img.stride, //
            (uint8_t*)dst.data, target[0], target[1], 0,                        //
            n_channels(img), alpha_channel(img.format), 0, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT,
            STBIR_COLORSPACE_SRGB, nullptr);
    }
    if (result == 0) {
        throw except(
            "Failed to resize image {}x{} to {}x{}", img.extent[0], img.extent[1], target[0],
            target[1]);
    }
}

image_data image_scale(image_view const& img, i32x2 target) {
    image_data dst = image_alloc(target, img.format);
    image_scale(img, target, dst);
    return dst;
}

template <typename T>
void blur(image_source<T> src, image_target<T> dst, int radius) {
    i32x2 extent = src.extent;
    ASSERT(src.extent == dst.extent);
    ASSERT(radius > 0);
    ASSERT(radius <= extent[0] / 2 && radius <= extent[1] / 2);

    std::vector<T> temp(n_pixels(src));
    float weight = 1.0f / (2 * radius + 1);

    // Horizontal pass (src -> temp)
    for (int y = 0; y < extent[1]; ++y) {
        int row_offset = y * extent[0];
        T sum = float(radius) * src[row_offset];
        for (int x = 0; x <= radius; ++x) {
            sum = sum + src[row_offset + x];
        }

        temp[row_offset] = sum * weight;

        for (int x = 1; x < extent[0]; ++x) {
            int left_x = clamp(x - radius - 1, 0, extent[0] - 1);
            sum = sum - src[row_offset + left_x];

            int right_x = clamp(x + radius, 0, extent[0] - 1);
            sum = sum + src[row_offset + right_x];

            temp[row_offset + x] = sum * weight;
        }
    }

    // Vertical pass (temp -> dst)
    for (int x = 0; x < extent[0]; ++x) {
        T sum = float(radius) * temp[x];
        for (int y = 0; y <= radius; ++y) {
            sum = sum + temp[y * extent[0] + x];
        }

        dst[x] = sum * weight;

        for (int y = 1; y < extent[1]; ++y) {
            int top_y = clamp(y - radius - 1, 0, extent[1] - 1);
            sum = sum - temp[top_y * extent[0] + x];

            int bottom_y = clamp(y + radius, 0, extent[1] - 1);
            sum = sum + temp[bottom_y * extent[0] + x];

            dst[y * extent[0] + x] = sum * weight;
        }
    }
}

void image_blur(image_view const& src, image_span const& dst, int radius) {
    ASSERT(src.extent == dst.extent);
    ASSERT(radius > 0);

    switch (src.format) {
        case image_format::alpha_f32: blur<float>(src, dst, radius); break;
        case image_format::rgba_f32: blur<f32x4>(src, dst, radius); break;
        default: ASSERT(false, "Unsupported image format for blur operation");
    }
}

// Approximate Fast Foreground Colour Estimation
// https://ieeexplore.ieee.org/document/9506164
auto blur_fusion_foreground_estimator(
    image_view img_in, image_view fg_in, image_view bg_in, image_view mask_in, int radius) {

    i32x2 extent = img_in.extent;
    size_t n = n_pixels(img_in);
    ASSERT(fg_in.extent == extent && bg_in.extent == extent && mask_in.extent == extent);
    image_source<f32x4> img(img_in);
    image_source<f32x4> fg(fg_in);
    image_source<f32x4> bg(bg_in);
    image_source<float> mask(mask_in);

    image_data blurred_mask_data = image_alloc(extent, image_format::alpha_f32);
    image_target<float> blurred_mask(blurred_mask_data);
    blur(mask, blurred_mask, radius);

    image_data fg_masked_data = image_alloc(extent, image_format::rgba_f32);
    image_target<f32x4> fg_masked(fg_masked_data);
    for (size_t i = 0; i < n; ++i) {
        fg_masked[i] = fg[i] * mask[i];
    }

    image_data blurred_fg_data = image_alloc(extent, image_format::rgba_f32);
    image_target<f32x4> blurred_fg(blurred_fg_data);
    blur(fg_masked, blurred_fg, radius);

    for (size_t i = 0; i < n; ++i) {
        blurred_fg[i] = blurred_fg[i] / (blurred_mask[i] + 1e-5f);
    }

    auto& bg_masked = fg_masked; // Reuse fg_masked memory for bg
    for (size_t i = 0; i < n; ++i) {
        bg_masked[i] = bg[i] * (1.0f - mask[i]);
    }

    image_data blurred_bg_data = image_alloc(extent, image_format::rgba_f32);
    image_target<f32x4> blurred_bg(blurred_bg_data);
    blur(bg_masked, blurred_bg, radius);

    for (size_t i = 0; i < n; ++i) {
        blurred_bg[i] = blurred_bg[i] / ((1.0f - blurred_mask[i]) + 1e-5f);
        f32x4 f = blurred_fg[i] +
            mask[i] * (img[i] - mask[i] * blurred_fg[i] - (1.0f - mask[i]) * blurred_bg[i]);
        f[3] = mask[i];
        blurred_fg[i] = clamp(f, 0.0f, 1.0f);
    }
    return std::pair{std::move(blurred_fg_data), std::move(blurred_bg_data)};
}

image_data image_estimate_foreground(image_view const& img, image_view const& mask, int radius) {
    ASSERT(img.extent == mask.extent);

    auto&& [fg, blur_bg] = blur_fusion_foreground_estimator(img, img, img, mask, radius);
    return blur_fusion_foreground_estimator(img, fg, blur_bg, mask, 3).first;
}

template <typename FG, typename BG>
void alpha_composite(
    image_source<FG> fg, image_source<BG> bg, image_source<uint8_t> alpha, image_target<u8x4> out) {

    int n = n_pixels(fg);
    for (int i = 0; i < n; ++i) {
        float w = alpha.load(i)[3];
        f32x4 v = w * fg.load(i) + (1.0f - w) * bg.load(i);
        v[3] = 1.0f;
        out.store(i, v);
    }
}

void image_alpha_composite(
    image_view const& fg, image_view const& bg, image_view const& mask, image_span const& dst) {

    ASSERT(fg.extent == bg.extent && fg.extent == mask.extent);

    switch (n_channels(bg.format)) {
        case 3: alpha_composite<u8x4, u8x3>(fg, bg, mask, dst); break;
        case 4: alpha_composite<u8x4, u8x4>(fg, bg, mask, dst); break;
        default: ASSERT(false, "Unsupported number of channels in background image"); return;
    }
}

image_data image_alpha_composite(image_view const& fg, image_view const& bg, image_view const& m) {
    image_data dst = image_alloc(fg.extent, image_format::rgba_u8);
    image_alpha_composite(fg, bg, m, dst);
    return dst;
}

template <typename T>
void erosion(image_source<T> src, image_target<T> dst, int radius) {
    for (int y = 0; y < src.extent[1]; ++y) {
        for (int x = 0; x < src.extent[0]; ++x) {
            T val = one(T{});
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = clamp(x + dx, 0, src.extent[0] - 1);
                    int ny = clamp(y + dy, 0, src.extent[1] - 1);
                    int i = ny * src.extent[0] + nx;
                    val = std::min(val, src[i]);
                }
            }
            dst[y * src.extent[0] + x] = val;
        }
    }
}

void image_erosion(image_view const& src, image_span const& dst, int radius) {
    ASSERT(src.extent == dst.extent);
    ASSERT(radius > 0);
    switch (src.format) {
        case image_format::alpha_u8: erosion<uint8_t>(src, dst, radius); break;
        case image_format::alpha_f32: erosion<float>(src, dst, radius); break;
        default: ASSERT(false, "erosion operation only supports single channel alpha formats");
    }
}

template <typename T>
float difference_rms(image_source<T> a, image_source<T> b) {
    float sum_sq_diff = 0.0f;
    size_t n = n_pixels(a);
    for (size_t i = 0; i < n; ++i) {
        f32x4 diff = a.load(i) - b.load(i);
        sum_sq_diff += dot(diff, diff);
    }
    return std::sqrt(sum_sq_diff / n);
}

float image_difference_rms(image_view const& a, image_view const& b) {
    ASSERT(a.extent == b.extent);

    switch (a.format) {
        case image_format::alpha_u8: return difference_rms<uint8_t>(a, b);
        case image_format::rgb_u8: return difference_rms<u8x3>(a, b);
        case image_format::rgba_u8: return difference_rms<u8x4>(a, b);
        case image_format::alpha_f32: return difference_rms<float>(a, b);
        case image_format::rgb_f32: return difference_rms<f32x3>(a, b);
        case image_format::rgba_f32: return difference_rms<f32x4>(a, b);
        default: ASSERT(false, "Invalid image format"); return 0.0f;
    }
}

//
// image tiling

tile_layout::tile_layout(i32x2 extent, int max_tile_size, int overlap, int align)
    : image_extent{extent[0], extent[1]}, overlap{overlap, overlap} {

    n_tiles = div_ceil(image_extent, max_tile_size);
    i32x2 img_extent_overlap = image_extent + (n_tiles - i32x2{1, 1}) * overlap;
    tile_size = div_ceil(img_extent_overlap, n_tiles);
    tile_size = div_ceil(tile_size, align) * align;
}

tile_layout tile_scale(tile_layout const& o, int scale) {
    tile_layout scaled;
    scaled.image_extent = o.image_extent * scale;
    scaled.overlap = o.overlap * scale;
    scaled.n_tiles = o.n_tiles;
    scaled.tile_size = o.tile_size * scale;
    return scaled;
}

i32x2 tile_layout::start(i32x2 coord, i32x2 pad) const {
    i32x2 offset = coord * (tile_size - overlap);
    return offset + i32x2{coord[0] == 0 ? 0 : pad[0], coord[1] == 0 ? 0 : pad[1]};
}

i32x2 tile_layout::end(i32x2 coord, i32x2 pad) const {
    i32x2 offset = start(coord) + tile_size;
    offset = offset -
        i32x2{coord[0] == n_tiles[0] - 1 ? 0 : pad[0], coord[1] == n_tiles[1] - 1 ? 0 : pad[1]};
    return min(offset, image_extent);
}

i32x2 tile_layout::size(i32x2 coord) const {
    return end(coord) - start(coord);
}

int tile_layout::total() const {
    return n_tiles[0] * n_tiles[1];
}

i32x2 tile_layout::coord(int index) const {
    ASSERT(index >= 0 && index < total());
    return {index % n_tiles[0], index / n_tiles[0]};
}

void tile_merge(
    image_view const& tile_img,
    image_span const& dst_img,
    i32x2 tile_coord,
    tile_layout const& layout) {

    image_source<f32x3> tile(tile_img);
    image_target<f32x3> dst(dst_img);

    i32x2 beg = layout.start(tile_coord);
    i32x2 end = layout.end(tile_coord);
    i32x2 pad_beg = layout.start(tile_coord, layout.overlap);
    i32x2 pad_end = layout.end(tile_coord, layout.overlap);

    for (int y = beg[1]; y < end[1]; ++y) {
        for (int x = beg[0]; x < end[0]; ++x) {
            i32x2 idx = {x, y};

            float weight = 1.0f;
            i32x2 coverage = {0, 0};
            for (int i = 0; i < 2; ++i) {
                if (idx[i] < pad_beg[i]) {
                    weight *= float(layout.overlap[i] - (pad_beg[i] - idx[i]) + 1);
                    coverage[i] = layout.overlap[i];
                } else if (idx[i] >= pad_end[i]) {
                    weight *= float(layout.overlap[i] - (idx[i] - pad_end[i]));
                    coverage[i] = layout.overlap[i];
                }
            }
            float norm = float((coverage[0] + 1) * (coverage[1] + 1));
            float blend = weight > 0 ? weight / norm : 1.0f;

            dst.store(idx, dst.load(idx) + blend * tile.load(idx - beg));
        }
    }
}

} // namespace visp
