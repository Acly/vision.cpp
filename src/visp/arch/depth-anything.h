#pragma once

#include "visp/arch/dino.h"
#include "visp/vision.h"
#include "visp/ml.h"
#include "visp/nn.h"

namespace visp {
namespace dpt {

tensor residual_conv(model_ref m, tensor x) {
    tensor out = x;
    out = ggml_relu(m, out);
    out = conv_2d(m["conv1"], out, 1, 1);
    out = ggml_relu(m, out);
    out = conv_2d(m["conv2"], out, 1, 1);
    x = ggml_add_inplace(m, x, out);
    return named(m, x);
}

tensor feature_fusion(model_ref m, tensor x0, tensor x1, int64_t const* size = nullptr) {
    tensor x = x0;
    if (x1) {
        tensor res = residual_conv(m["resConfUnit1"], x1);
        x = ggml_add_inplace(m, x, res);
    }
    x = residual_conv(m["resConfUnit2"], x);

    int64_t w = size ? size[0] : x->ne[0] * 2;
    int64_t h = size ? size[1] : x->ne[1] * 2;
    int32_t mode = int32_t(GGML_SCALE_MODE_BILINEAR) | GGML_SCALE_FLAG_ALIGN_CORNERS;
    x = interpolate(m, x, {w, h}, mode);

    x = conv_2d(m["out_conv"], x);
    return named(m, x);
}

tensor head(model_ref m, span<tensor> features, int64_t patch_w, int64_t patch_h) {
    ASSERT(features.size() == 4);
    std::array<tensor, 4> layer;
    for (int i = 0; i < 4; ++i) {
        tensor x = features[i];
        x = slice(m, x, {}, {1, x->ne[1]}, {}, {});
        x = ggml_reshape_4d(m, x, x->ne[0], patch_w, patch_h, x->ne[3]);

        model_ref proj = m["projects"][i];
        proj.flags |= model_build_flag::cwhn;
        x = conv_2d(proj, x); // 1x1 conv, keep CWHN layout and directly use mul_mat

        x = cwhn_to_contiguous_2d(m, x);
        switch (i) {
            case 0: x = conv_transpose_2d(m["resize_layers"][i], x, 4); break;
            case 1: x = conv_transpose_2d(m["resize_layers"][i], x, 2); break;
            case 3: x = conv_2d(m["resize_layers"][i], x, 2, 1); break;
        }
        layer[i] = x;
    }

    model_ref scratch = m["scratch"];
    tensor layer1_rn = conv_2d(scratch["layer1_rn"], layer[0], 1, 1);
    tensor layer2_rn = conv_2d(scratch["layer2_rn"], layer[1], 1, 1);
    tensor layer3_rn = conv_2d(scratch["layer3_rn"], layer[2], 1, 1);
    tensor layer4_rn = conv_2d(scratch["layer4_rn"], layer[3], 1, 1);

    tensor path4 = feature_fusion(scratch["refinenet4"], layer4_rn, nullptr, layer3_rn->ne);
    tensor path3 = feature_fusion(scratch["refinenet3"], path4, layer3_rn, layer2_rn->ne);
    tensor path2 = feature_fusion(scratch["refinenet2"], path3, layer2_rn, layer1_rn->ne);
    tensor path1 = feature_fusion(scratch["refinenet1"], path2, layer1_rn);

    tensor out = conv_2d(scratch["output_conv1"], path1, 1, 1);
    out = interpolate(
        m, out, {patch_w * 14, patch_h * 14},
        int32_t(GGML_SCALE_MODE_BILINEAR) | GGML_SCALE_FLAG_ALIGN_CORNERS);

    model_ref output_conv2 = scratch["output_conv2"];
    out = conv_2d(output_conv2[0], out, 1, 1);
    out = ggml_relu_inplace(m, out);
    out = conv_2d(output_conv2[2], out);
    out = ggml_relu_inplace(m, out);
    return out;
}

} // namespace dpt

inline tensor depthany_predict(model_ref m, tensor image, depthany_params const& p) {
    auto [c, w, h, n] = nelements(image);
    int64_t w_patch = w / p.dino.patch_size;
    int64_t h_patch = h / p.dino.patch_size;

    auto features = dino_intermediate_layers(m["pretrained"], image, p.feature_layers, p.dino);
    tensor depth = dpt::head(m["depth_head"], features, w_patch, h_patch);
    depth = ggml_relu_inplace(m, depth);
    return compute_graph_output(m, depth);
}

i32x2 depthany_image_extent(i32x2 extent, depthany_params const& p) {
    int min_side = std::min(extent[0], extent[1]);
    int tgt_side = std::max(p.image_size, next_multiple(min_side, p.image_multiple));
    i32x2 target = extent * tgt_side / min_side;
    return next_multiple(target, p.image_multiple);
}

inline depthany_params depthany_detect_params(model_file const&, i32x2 input_extent) {
    depthany_params p;
    p.dino.patch_size = 14;
    if (input_extent[0] > 0 && input_extent[1] > 0) {
        p.image_extent = depthany_image_extent(input_extent, p);
    }
    return p;
}

inline image_data depthany_process_input(image_view image, depthany_params const& p) {
    constexpr f32x4 mean = f32x4{0.485f, 0.456f, 0.406f, 0.f};
    constexpr f32x4 std = f32x4{0.229f, 0.224f, 0.225f, 1.f};

    image_data resized;
    if (image.extent != p.image_extent) {
        resized = image_scale(image, p.image_extent);
        image = image_view(resized);
    }
    return image_u8_to_f32(image, image_format::rgb_f32, -mean, 1.f / std);
}

inline image_data depthany_process_output(
    span<float const> data, i32x2 extent, depthany_params const& p) {

    image_view depth_output(p.image_extent, data);
    image_data depth_resized;
    if (depth_output.extent != extent) {
        depth_resized = image_scale(depth_output, extent);
        depth_output = depth_resized;
    }
    return image_f32_to_u8(depth_output, image_format::alpha_u8);
}

} // namespace visp