#include "testing.h"
#include "visp/ml.h"

#include <numeric>

namespace visp {

VISP_TEST(model_transfer_type_conversion) {
    model_weights src = model_init(2);

    tensor i = ggml_new_tensor_1d(src, GGML_TYPE_I32, 2);
    ggml_set_name(i, "i32_tensor");
    auto i32_data = std::array{4, -1};
    i->data = i32_data.data();

    tensor f = ggml_new_tensor_1d(src, GGML_TYPE_F16, 2);
    ggml_set_name(f, "f16_tensor");
    auto f16_data = std::array{ggml_fp32_to_fp16(2.5f), ggml_fp32_to_fp16(-0.5f)};
    f->data = f16_data.data();

    backend_device dev = backend_init(backend_type::cpu);
    model_weights dst = model_init(2);
    model_transfer(src, dst, dev, GGML_TYPE_F32); // f16 -> f32 conversion

    int32_t const* i32_result = (int32_t const*)ggml_get_tensor(dst, "i32_tensor")->data;
    CHECK_EQUAL(i32_result[0], 4);
    CHECK_EQUAL(i32_result[1], -1);

    tensor f_result = ggml_get_tensor(dst, "f16_tensor");
    CHECK(f_result->type == GGML_TYPE_F32);
    float const* f32_result = (float const*)f_result->data;
    CHECK_EQUAL(f32_result[0], 2.5f);
    CHECK_EQUAL(f32_result[1], -0.5f);
}

VISP_TEST(model_transfer_layout_conversion) {
    model_weights src = model_init(3);

    tensor conv_dw = ggml_new_tensor_4d(src, GGML_TYPE_F32, 2, 2, 1, 3); // wh1c
    ggml_set_name(conv_dw, "conv_dw");
    auto conv_dw_data = std::array<float, 2 * 2 * 1 * 3>{};
    std::iota(conv_dw_data.begin(), conv_dw_data.end(), 1.0f);
    conv_dw->data = conv_dw_data.data();

    tensor conv = ggml_new_tensor_4d(src, GGML_TYPE_F32, 2, 2, 4, 3); // whco
    ggml_set_name(conv, "conv");
    auto conv_data = std::array<float, 2 * 2 * 3 * 4>{};
    std::iota(conv_data.begin(), conv_data.end(), 1.0f);
    conv->data = conv_data.data();

    tensor no_conv = ggml_new_tensor_1d(src, GGML_TYPE_F32, 2);
    ggml_set_name(no_conv, "no_conv");
    auto no_conv_data = std::array<float, 2>{1.0f, 2.0f};
    no_conv->data = no_conv_data.data();

    auto conv_weights = std::array{0, 1};
    auto src_layout = tensor_data_layout::whcn;
    auto dst_layout = tensor_data_layout::cwhn;

    backend_device dev = backend_init(backend_type::cpu);
    model_weights dst = model_init(3);
    model_transfer(src, dst, dev, GGML_TYPE_COUNT, src_layout, dst_layout, conv_weights);

    auto conv_dw_expected = std::array{
        1.0f, 5.0f, 9.0f,  //
        2.0f, 6.0f, 10.0f, //
        3.0f, 7.0f, 11.0f, //
        4.0f, 8.0f, 12.0f  //
    };
    float const* conv_dw_result = (float const*)ggml_get_tensor(dst, "conv_dw")->data;
    for (int i = 0; i < int(conv_dw_expected.size()); ++i) {
        CHECK_EQUAL(conv_dw_result[i], conv_dw_expected[i]);
    }

    auto conv_expected = std::array{
        1.0f,  5.0f,  9.0f,  13.0f, 2.0f, 6.0f, 10.0f, 14.0f, //
        3.0f,  7.0f,  11.0f, 15.0f, 4.0f, 8.0f, 12.0f, 16.0f, //

        17.0f,  21.0f,  25.0f, 29.0f, 18.0f, 22.0f, 26.0f, 30.0f, //
        19.0f, 23.0f, 27.0f, 31.0f, 20.0f, 24.0f, 28.0f, 32.0f, //

        33.0f, 37.0f, 41.0f, 45.0f, 34.0f, 38.0f, 42.0f, 46.0f, //
        35.0f, 39.0f, 43.0f, 47.0f, 36.0f, 40.0f, 44.0f, 48.0f  //
    };
    float const* conv_result = (float const*)ggml_get_tensor(dst, "conv")->data;
    for (int i = 0; i < int(conv_expected.size()); ++i) {
        CHECK_EQUAL(conv_result[i], conv_expected[i]);
    }

    float const* no_conv_result = (float const*)ggml_get_tensor(dst, "no_conv")->data;
    CHECK_EQUAL(no_conv_result[0], 1.0f);
    CHECK_EQUAL(no_conv_result[1], 2.0f);
}

} // namespace visp