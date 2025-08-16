#include "testing.h"
#include "visp/image-impl.h"
#include "visp/image.h"
#include "visp/util.h"

#include <array>
#include <filesystem>

namespace visp {

VISP_TEST(image_formats) {
    auto formats = std::array{image_format::rgba_u8, image_format::bgra_u8,  image_format::argb_u8,
                              image_format::rgb_u8,  image_format::alpha_u8, image_format::rgba_f32,
                              image_format::rgb_f32, image_format::alpha_f32};

    for (image_format format : formats) {
        int tsize = is_float(format) ? 4 : 1;
        CHECK_EQUAL(n_channels(format) * tsize, n_bytes(format));

        image_data img = image_alloc(i32x2{8, 6}, format);
        CHECK(n_bytes(img) == size_t(8 * 6 * n_bytes(format)));

        image_view view(img);
        if (is_float(format)) {
            CHECK(view.as_floats().size() == n_bytes(img) / sizeof(float));
        } else {
            CHECK(view.as_bytes().size() == n_bytes(img));
        }

        image_span img_span(img);
        if (is_float(format)) {
            CHECK(img_span.as_floats().size() == n_bytes(img) / sizeof(float));
        } else {
            CHECK(img_span.as_bytes().size() == n_bytes(img));
        }
    }
}

VISP_TEST(image_load) {
    image_data img = image_load((test_dir().input / "cat-and-hat.jpg").string().c_str());
    CHECK(img.extent == i32x2{512, 512});
    CHECK(img.format == image_format::rgb_u8);
    CHECK(n_bytes(img) == 512 * 512 * 3);
}

VISP_TEST(image_save) {
    image_data img = image_alloc(i32x2{16, 16}, image_format::rgba_u8);
    for (int i = 0; i < 16 * 16; ++i) {
        img.data.get()[i * 4 + 0] = 255;
        img.data.get()[i * 4 + 1] = uint8_t(i);
        img.data.get()[i * 4 + 2] = 0;
        img.data.get()[i * 4 + 3] = 255;
    }
    path filepath = (test_dir().results / "image-save.png");
    image_save(img, filepath.string().c_str());
    CHECK(exists(filepath));

    image_data result = image_load(filepath.string().c_str());
    CHECK_IMAGES_EQUAL(result, img);
}

void test_image_u8_to_f32(
    image_format in_format,
    image_format out_format,
    span<uint8_t const> input_data,
    span<float const> expected_data) {

    image_view input(i32x2{2, 2}, in_format, input_data);
    image_view expected(i32x2{2, 2}, out_format, expected_data.data());
    f32x4 offset = f32x4{0.1f, 0.2f, 0.3f, 0.4f};
    f32x4 scale = f32x4{0.5f, 1.0f, -1.f, 1.0f};
    image_data output = image_u8_to_f32(input, out_format, offset, scale);
    test_with_tolerance tol{0.01f};
    CHECK_IMAGES_EQUAL(output, expected);
}

VISP_TEST(image_alpha_u8_to_alpha_f32) {
    test_image_u8_to_f32(
        image_format::alpha_u8, image_format::alpha_f32, //
        std::array<uint8_t, 4>{0, 128, 190, 255},        //
        std::array<float, 4>{0.05f, 0.3f, 0.4225f, 0.55f});
}
VISP_TEST(image_rgb_u8_to_rgb_f32) {
    test_image_u8_to_f32(
        image_format::rgb_u8, image_format::rgb_f32,                                  //
        std::array<uint8_t, 12>{0, 128, 192, 255, 0, 128, 128, 255, 0, 128, 64, 255}, //
        std::array<float, 12>{
            0.05f, 0.7f, -1.05f, 0.55f, 0.2f, -0.8f, 0.3f, 1.2f, -0.3f, 0.3f, 0.45f, -1.3f});
}
VISP_TEST(image_rgba_u8_to_rgb_f32) {
    test_image_u8_to_f32(
        image_format::rgba_u8, image_format::rgb_f32, //
        std::array<uint8_t, 16>{
            0, 128, 192, 42, //
            255, 0, 128, 42, //
            128, 255, 0, 42, //
            128, 64, 255, 42},
        std::array<float, 12>{
            0.05f, 0.7f, -1.05f, //
            0.55f, 0.2f, -0.8f,  //
            0.3f, 1.2f, -0.3f,   //
            0.3f, 0.45f, -1.3f});
}
VISP_TEST(image_rgba_u8_to_rgba_f32) {
    test_image_u8_to_f32(
        image_format::rgba_u8, image_format::rgba_f32, //
        std::array<uint8_t, 16>{
            0, 128, 192, 0,   //
            255, 0, 128, 64,  //
            128, 255, 0, 128, //
            128, 64, 255, 255},
        std::array<float, 16>{
            0.05f, 0.7f, -1.05f, 0.4f,     //
            0.55f, 0.2f, -0.8f, 0.65f,     //
            0.3f, 1.2f, -0.3f, 0.9f, 0.3f, //
            0.45f, -1.3f, 1.4f});
}
VISP_TEST(image_bgra_u8_to_rgb_f32) {
    test_image_u8_to_f32(
        image_format::bgra_u8, image_format::rgb_f32, //
        std::array<uint8_t, 16>{
            192, 128, 0, 42, //
            128, 0, 255, 42, //
            0, 255, 128, 42, //
            255, 64, 128, 42},
        std::array<float, 12>{
            0.05f, 0.7f, -1.05f, //
            0.55f, 0.2f, -0.8f,  //
            0.3f, 1.2f, -0.3f,   //
            0.3f, 0.45f, -1.3f});
}
VISP_TEST(image_argb_u8_to_rgb_f32) {
    test_image_u8_to_f32(
        image_format::argb_u8, image_format::rgb_f32, //
        std::array<uint8_t, 16>{
            42, 0, 128, 192, //
            42, 255, 0, 128, //
            42, 128, 255, 0, //
            42, 128, 64, 255},
        std::array<float, 12>{
            0.05f, 0.7f, -1.05f, //
            0.55f, 0.2f, -0.8f,  //
            0.3f, 1.2f, -0.3f,   //
            0.3f, 0.45f, -1.3f});
}

VISP_TEST(image_u8_to_f32_tiled_pad) {
    std::array<uint8_t, 9> input_data = {0, 0, 102, 0, 0, 255, 0, 0, 102};
    std::array<float, 4> expected_data = {1.0f, 1.0f, 0.4f, 0.4f};
    image_view input(i32x2{3, 3}, image_format::alpha_u8, input_data);
    image_view expected(i32x2{2, 2}, image_format::alpha_f32, expected_data.data());
    f32x4 offset = f32x4{0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 scale = f32x4{1.0f, 1.0f, 1.0f, 1.0f};
    i32x2 tile_offset = {2, 1};

    std::array<float, 4> output_data;
    image_span output(i32x2{2, 2}, image_format::alpha_f32, output_data.data());
    image_u8_to_f32(input, output, offset, scale, tile_offset);
    CHECK_IMAGES_EQUAL(output, expected);
}

VISP_TEST(image_alpha_f32_to_alpha_u8) {
    std::array<float, 4> input_data{0.0f, 0.3f, 0.4225f, 1.1f};
    std::array<uint8_t, 4> expected_data = {0, 76, 107, 255};
    image_view input(i32x2{2, 2}, image_format::alpha_f32, input_data.data());
    image_view expected(i32x2{2, 2}, image_format::alpha_u8, expected_data);

    image_data output = image_f32_to_u8(input, image_format::alpha_u8);
    CHECK(output.extent == i32x2{2, 2});
    CHECK(output.format == image_format::alpha_u8);
    CHECK_IMAGES_EQUAL(output, expected);
}

VISP_TEST(image_rgb_f32_to_rgba_u8) {
    std::array<float, 6> input_data{0.0f, 0.31f, -0.51f, 1.0f, 0.2f, 1.8f};
    std::array<uint8_t, 8> expected_data = {0, 79, 0, 255, 255, 51, 255, 255};
    image_view input(i32x2{2, 1}, image_format::rgb_f32, input_data.data());
    image_view expected(i32x2{2, 1}, image_format::rgba_u8, expected_data);

    image_data output = image_f32_to_u8(input, image_format::rgba_u8);
    CHECK(output.extent == i32x2{2, 1});
    CHECK(output.format == image_format::rgba_u8);
    CHECK_IMAGES_EQUAL(output, expected);
}

VISP_TEST(image_scale) {
    image_data img = image_alloc(i32x2{8, 8}, image_format::rgba_u8);
    for (int i = 0; i < 8 * 8; ++i) {
        img.data[i * 4 + 0] = uint8_t(255);
        img.data[i * 4 + 1] = uint8_t(4 * (i / 8));
        img.data[i * 4 + 2] = uint8_t(4 * (i % 8));
        img.data[i * 4 + 3] = uint8_t(255);
    }
    image_data result = image_scale(img, i32x2{4, 4});
    CHECK(result.extent == i32x2{4, 4});
    CHECK(result.format == image_format::rgba_u8);
    for (int i = 0; i < 16; ++i) {
        CHECK(result.data[i * 4 + 0] == 255);
        CHECK(int(result.data[i * 4 + 1]) == 2 + 8 * (i / 4));
        CHECK(int(result.data[i * 4 + 2]) == 2 + 8 * (i % 4));
        CHECK(result.data[i * 4 + 3] == 255);
    }
}

VISP_TEST(image_alpha_composite) {
    std::array<uint8_t, 2 * 2 * 4> fg_data = {255, 0, 0,   255, 0,   255, 0, 255, //
                                              0,   0, 255, 255, 255, 255, 0, 255};
    image_view fg{i32x2{2, 2}, image_format::rgba_u8, fg_data};

    std::array<uint8_t, 2 * 2 * 3> bg_data = {0,   0,   0,   128, 128, 128, //
                                              255, 255, 255, 64,  64,  64};
    image_view bg{i32x2{2, 2}, image_format::rgb_u8, bg_data};

    std::array<uint8_t, 2 * 2> mask_data = {255, 128, 64, 0};
    image_view mask{i32x2{2, 2}, image_format::alpha_u8, mask_data};

    std::array<uint8_t, 2 * 2 * 4> expected_output = {255, 0,   0,   255, 63, 191, 63, 255, //
                                                      191, 191, 255, 255, 64, 64,  64, 255};
    image_view expected{i32x2{2, 2}, image_format::rgba_u8, expected_output};

    image_data output = image_alpha_composite(fg, bg, mask);
    CHECK_IMAGES_EQUAL(output, expected);
}

VISP_TEST(image_blur) {
    constexpr i32x2 extent{6, 6};
    // clang-format off
    std::array<float, extent[0] * extent[1]> input_data = {
         1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
         7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
        19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
        31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f
    };    
    std::array<float, extent[0] * extent[1]> expected_data = {
         3.33334f,  4.0f,  5.0f,  6.0f,  7.0f,  7.66667f,
         7.33334f,  8.0f,  9.0f, 10.0f, 11.0f, 11.66667f,
        13.33334f, 14.0f, 15.0f, 16.0f, 17.0f, 17.66667f,
        19.33334f, 20.0f, 21.0f, 22.0f, 23.0f, 23.66667f,
        25.33334f, 26.0f, 27.0f, 28.0f, 29.0f, 29.66667f,
        29.33334f, 30.0f, 31.0f, 32.0f, 33.0f, 33.66667f
    };
    // clang-format on
    std::array<float, extent[0] * extent[1]> output_data{};

    auto input = image_view(extent, input_data);
    auto output = image_span(extent, output_data);
    image_blur(input, output, 1);

    auto expected = image_view(extent, expected_data);
    CHECK_IMAGES_EQUAL(output, expected);
}

VISP_TEST(image_erosion) {
    constexpr i32x2 extent{6, 6};
    std::array<float, extent[0] * extent[1]> input_data = {
        0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, //
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, //
        0.8f, 0.8f, 1.0f, 0.5f, 0.0f, 1.0f, //
        0.8f, 0.8f, 1.0f, 0.5f, 0.5f, 0.0f, //
        0.8f, 1.0f, 1.0f, 0.5f, 0.5f, 0.0f, //
        0.0f, 1.0f, 1.0f, 0.2f, 0.5f, 0.0f  //
    };
    std::array<float, extent[0] * extent[1]> expected_data = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, //
        0.8f, 0.8f, 0.5f, 0.0f, 0.0f, 0.0f, //
        0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.0f, //
        0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.0f  //
    };
    std::array<float, extent[0] * extent[1]> output_data{};

    auto input = image_view(extent, input_data);
    auto output = image_span(extent, output_data);
    image_erosion(input, output, 1);

    auto expected = image_view(extent, expected_data);
    CHECK_IMAGES_EQUAL(output, expected);
}

VISP_TEST(tile_merge) {
    std::array<std::array<f32x3, 5 * 5>, 4> tiles;
    for (int t = 0; t < 4; ++t) {
        float v = float(t);
        std::fill(tiles[t].begin(), tiles[t].end(), f32x3{v, v, v});
    }
    std::array<f32x3, 8 * 8> dst{};
    auto dst_span = image_span({8, 8}, dst);
    auto const layout = tile_layout(i32x2{8, 8}, 6, 2, 1);
    tile_merge(image_view({5, 5}, tiles[0]), dst_span, {0, 0}, layout);
    tile_merge(image_view({5, 5}, tiles[1]), dst_span, {1, 0}, layout);
    tile_merge(image_view({5, 5}, tiles[2]), dst_span, {0, 1}, layout);
    tile_merge(image_view({5, 5}, tiles[3]), dst_span, {1, 1}, layout);

    float e00 = float(4 * 0 + 2 * 1 + 2 * 2 + 1 * 3) / 9.f;
    float e10 = float(2 * 0 + 4 * 1 + 1 * 2 + 2 * 3) / 9.f;
    float e01 = float(2 * 0 + 1 * 1 + 4 * 2 + 2 * 3) / 9.f;
    float e11 = float(1 * 0 + 2 * 1 + 2 * 2 + 4 * 3) / 9.f;
    // clang-format off
    auto expected_float = std::array<float, 8 * 8>{
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        2.f/3.f, 2.f/3.f, 2.f/3.f, e00    , e10    , 5.f/3.f, 5.f/3.f, 5.f/3.f,
        4.f/3.f, 4.f/3.f, 4.f/3.f, e01    , e11    , 7.f/3.f, 7.f/3.f, 7.f/3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f
    };
    // clang-format on
    std::array<f32x3, 8 * 8> expected_rgb;
    for (int i = 0; i < 8 * 8; ++i) {
        expected_rgb[i] = f32x3{expected_float[i], expected_float[i], expected_float[i]};
    }
    auto expected = image_view({8, 8}, expected_rgb);
    CHECK_IMAGES_EQUAL(dst_span, expected);
}

VISP_TEST(tile_merge_blending) {
    std::array<f32x3, 22 * 19> dst{};
    auto dst_span = image_span({22, 19}, dst);

    auto layout = tile_layout(i32x2{22, 19}, 10, 3, 2);
    auto te = layout.tile_size;
    auto tile = std::vector<f32x3>(te[0] * te[1], f32x3{1.f, 1.f, 1.f});
    auto tile_span = image_view(te, tile);

    for (int y = 0; y < layout.n_tiles[1]; ++y) {
        for (int x = 0; x < layout.n_tiles[0]; ++x) {
            tile_merge(tile_span, dst_span, {x, y}, layout);
        }
    }
    for (float value : dst_span.as_floats()) {
        CHECK_EQUAL(value, 1.0f);
    }
}

} // namespace visp
