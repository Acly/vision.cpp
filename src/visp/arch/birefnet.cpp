#include "visp/arch/birefnet.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <ggml.h>

namespace visp {
namespace birefnet {

tensor mlp(model_ref m, tensor x) {
    x = linear(m["fc1"], x);
    x = ggml_gelu_inplace(m, x);
    x = linear(m["fc2"], x);
    return named(m, x);
}

// Ensures that the tensor's data is not overwritten during computation.
tensor make_constant(tensor x, tensor_name name) {
    ggml_set_name(x, name.c_str());
    ggml_set_input(x);  // allocate at the beginning of the graph buffer
    ggml_set_output(x); // don't reuse memory for computations
    return x;
}

void compute_relative_position_index(span<int32_t> dst, int window_size) {
    int n = window_size;
    int n2 = n * n;
    int n4 = n2 * n2;
    for (int i = 0; i < n4; ++i) {
        int x0 = i % n;
        int y0 = (i / n) % n;
        int x1 = (i / n2) % n;
        int y1 = (i / n2 / n) % n;
        dst[i] = (y1 - y0 + n - 1) * (2 * n - 1) + (x1 - x0 + n - 1);
    }
}

tensor_data create_relative_position_index(ggml_context* ctx, int window_size) {
    int n = window_size;
    auto result = tensor_alloc(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n * n * n * n));
    auto name = format<tensor_name>("window_attention_{}.rel_pos_index", n);
    compute_relative_position_index(result.as_i32(), n);
    make_constant(result.x, name);
    return result;
}

tensor window_partition(model_ref m, tensor x, int window) {
    auto [c, w, h, b] = nelements(x);
    ASSERT(w % window == 0 && h % window == 0, "Expecting padded input");

    x = ggml_reshape_4d(m, x, c * window, w / window, window, (h / window) * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, window * window, (w / window) * (h / window) * b);
    return x;
}

tensor window_reverse(model_ref m, tensor x, int64_t w, int64_t h, int window) {
    int64_t c = x->ne[0];
    int64_t b = x->ne[2] / (w / window) / (h / window);
    ASSERT(x->ne[2] % (w / window) == 0, "Expecting ne[2] to be multiple of window count");

    x = ggml_reshape_4d(m, x, c * window, window, w / window, (h / window) * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, c, w, h, b);
    return x;
}

tensor window_attention(model_ref m, tensor x, tensor mask, int num_heads, int window) {
    auto [c, n, b, _] = nelements(x);

    tensor qkv = linear(m["qkv"], x);
    qkv = ggml_reshape_4d(m, qkv, c / num_heads, num_heads, 3, n * b);
    qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 1, 3, 2));

    auto split = [=](tensor tensor, size_t index, bool transpose = false) mutable {
        tensor = slice(m, tensor, {}, {}, {}, index);
        tensor = ggml_reshape_4d(m, tensor, c / num_heads, num_heads, n, b);
        if (transpose) {
            tensor = ggml_cont(m, ggml_permute(m, tensor, 1, 2, 0, 3));
        } else {
            tensor = ggml_cont(m, ggml_permute(m, tensor, 0, 2, 1, 3));
        }
        return tensor;
    };
    tensor q = split(qkv, 0);
    tensor k = split(qkv, 1);
    tensor v = split(qkv, 2, true);

    q = ggml_scale_inplace(m, q, 1.0f / std::sqrt(float(c / num_heads)));

    tensor attn = ggml_mul_mat(m, k, q);

    tensor_name rel_pos_name = format<tensor_name>("window_attention_{}.rel_pos_index", window);
    tensor rel_pos_index = ggml_get_tensor(m, rel_pos_name.c_str());
    tensor rel_pos_table = m.weights("relative_position_bias_table");
    tensor rel_pos_bias = ggml_get_rows(m, rel_pos_table, rel_pos_index);
    rel_pos_bias = ggml_reshape_4d(m, rel_pos_bias, num_heads, window * window, window * window, 1);
    rel_pos_bias = ggml_cont(m, ggml_permute(m, rel_pos_bias, 2, 0, 1, 3));
    attn = ggml_add_inplace(m, attn, rel_pos_bias);

    if (mask) {
        int64_t nw = mask->ne[2];
        attn = ggml_reshape_4d(m, attn, n * n, num_heads, nw, b / nw);
        mask = ggml_reshape_4d(m, mask, n * n, 1, nw, 1);
        attn = ggml_add_inplace(m, attn, mask);
        attn = ggml_reshape_4d(m, attn, n, n, num_heads, b);
    }
    attn = ggml_soft_max(m, attn);

    x = ggml_mul_mat(m, v, attn);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, n, b);

    x = linear(m["proj"], x);
    return named(m, x);
}

tensor swin_block(model_ref m, tensor x, tensor mask, swin_block_params const& p) {
    auto [c, n, b, _] = nelements(x);
    auto [num_heads, window, w, h, shift] = p;
    ASSERT(n == w * h && "Spatial dimensions do not match");

    tensor shortcut = x;
    x = layer_norm(m["norm1"], x);
    x = ggml_reshape_4d(m, x, c, w, h, b);

    int pad_r = (window - w % window) % window;
    int pad_b = (window - h % window) % window;
    if (pad_r > 0 || pad_b > 0) {
        x = ggml_pad(m, x, 0, pad_r, pad_b, 0);
    }

    ASSERT(shift == 0 || mask != nullptr);
    if (shift > 0) {
        x = ggml_roll(m, x, 0, -shift, -shift, 0);
    }

    x = window_partition(m, x, window);
    x = window_attention(m["attn"], x, mask, num_heads, window);
    x = window_reverse(m, x, w + pad_r, h + pad_b, window);

    if (shift > 0) { // undo shift
        x = ggml_roll(m, x, 0, shift, shift, 0);
    }

    if (pad_r > 0 || pad_b > 0) { // undo padding
        x = ggml_reshape_4d(m, x, c, w + pad_r, h + pad_b, b);
        x = slice(m, x, {}, {0, w}, {0, h}, {});
        x = ggml_cont(m, x);
    }

    x = ggml_reshape_3d(m, x, c, n, b);
    x = ggml_add_inplace(m, x, shortcut);

    tensor x_mlp = layer_norm(m["norm2"], x);
    x_mlp = mlp(m["mlp"], x_mlp);
    x = ggml_add_inplace(m, x, x_mlp);

    return named(m, x);
}

tensor patch_merging(model_ref m, tensor x, int64_t w, int64_t h) {
    auto [c, n, b, _] = nelements(x);
    ASSERT(n == w * h, "Spatial dimensions do not match");
    ASSERT(w % 2 == 0 && h % 2 == 0, "Expecting even spatial dimensions");

    x = ggml_reshape_4d(m, x, c, w, h, b);
    // clang-format off
    x = concat(m, {
        slice(m, x, {}, {0, w, 2}, {0, h, 2}, {}),
        slice(m, x, {}, {0, w, 2}, {1, h, 2}, {}),
        slice(m, x, {}, {1, w, 2}, {0, h, 2}, {}),
        slice(m, x, {}, {1, w, 2}, {1, h, 2}, {})}, 0);
    // clang-format on
    x = ggml_reshape_3d(m, x, c * 4, n / 4, b);

    x = layer_norm(m["norm"], x);
    x = linear(m["reduction"], x);
    return named(m, x);
}

void compute_attention_mask(span<float> out, int64_t w, int64_t h, int window_size) {
    int n = window_size;
    int n2 = n * n;
    int n4 = n2 * n2;
    int shift = window_size / 2;
    int64_t nw_x = (w + n - 1) / n;
    int64_t nw_y = (h + n - 1) / n;
    int64_t w_pad = nw_x * n;
    int64_t h_pad = nw_y * n;

    std::fill(out.begin(), out.end(), 0.0f);

    for (int iw_y = 0; iw_y < nw_y; ++iw_y) {
        for (int iw_x = 0; iw_x < nw_x; ++iw_x) {
            // Skip all windows that aren't at the right or bottom edges of the image
            if (iw_y < nw_y - 1 && iw_x < nw_x - 1) {
                continue;
            }
            int64_t base = iw_y * nw_x * n4 + iw_x * n4;

            for (int y0 = 0; y0 < n; ++y0) {
                for (int x0 = 0; x0 < n; ++x0) {
                    for (int y1 = 0; y1 < n; ++y1) {
                        for (int x1 = 0; x1 < n; ++x1) {
                            // Window-local coordinates to global image coordinates
                            int yy0 = iw_y * n + y0;
                            int xx0 = iw_x * n + x0;
                            int yy1 = iw_y * n + y1;
                            int xx1 = iw_x * n + x1;
                            // Check if two patches being matched belong to the same window
                            // that is: they are both in the shift zone, or both outside
                            bool match_y = (yy0 < h_pad - shift) == (yy1 < h_pad - shift);
                            bool match_x = (xx0 < w_pad - shift) == (xx1 < w_pad - shift);
                            // If not, set mask to -100 (added to attention before softmax)
                            if (!match_y || !match_x) {
                                int64_t idx = base + (y0 * n + x0) * n2 + (y1 * n + x1);
                                out[idx] = -100.f;
                            }
                        }
                    }
                }
            }
        }
    }
}

tensor_data create_attention_mask(ggml_context* ctx, int64_t w, int64_t h, int window_size) {
    int n = window_size;
    int64_t nw_x = (w + n - 1) / n;
    int64_t nw_y = (h + n - 1) / n;
    auto result = tensor_alloc(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n * n, n * n, nw_x * nw_y));
    auto name = format<tensor_name>("swin_layer_{}x{}.attn_mask", w, h);
    compute_attention_mask(result.as_f32(), w, h, window_size);
    make_constant(result.x, name);
    return result;
}

swin_layer_result swin_layer(
    model_ref m, tensor x, int64_t w, int64_t h, swin_layer_t const& p, int window_size) {
    // Attention masks need to be precomputed
    tensor_name attn_mask_name = format<tensor_name>("swin_layer_{}x{}.attn_mask", w, h);
    tensor attn_mask = ggml_get_tensor(m, attn_mask_name.c_str());

    model_ref blocks = m["blocks"];
    for (int i = 0; i < p.depth; ++i) {
        x = swin_block(
            blocks[i], x, attn_mask,
            {.n_heads = p.n_heads,
             .window_size = window_size,
             .w = w,
             .h = h,
             .shift = i % 2 == 0 ? 0 : window_size / 2});
    }
    if (p.downsample) {
        tensor x_down = patch_merging(m["downsample"], x, w, h);
        return {x, w, h, x_down, (w + 1) / 2, (h + 1) / 2};
    }
    return {x, w, h, x, w, h};
}

tensor patch_embed(model_ref m, tensor x, int patch_size) {
    ASSERT(x->ne[1] % patch_size == 0 && x->ne[2] % patch_size == 0);

    m.flags |= model_build_flag::cwhn;
    x = conv_2d(m["proj"], x, patch_size);
    auto [c, ww, wh, b] = nelements(x);
    x = ggml_reshape_3d(m, x, c, ww * wh, b);
    x = layer_norm(m["norm"], x);
    x = ggml_reshape_4d(m, x, c, ww, wh, b);
    return named(m, x);
}

swin_result swin_transformer(model_ref m, tensor x, swin_params const& p) {
    x = patch_embed(m["patch_embed"], x, 4);

    auto [c, w, h, b] = nelements(x);
    x = ggml_reshape_3d(m, x, c, w * h, b);

    swin_layer_result r{x, w, h, x, w, h};
    swin_result outs = {};

    for (int i = 0; i < swin_params::n_layers; ++i) {
        model_ref layer = m["layers"][i];
        r = swin_layer(layer, r.x_down, r.w_down, r.h_down, p.layers[i], p.window_size);

        tensor_name norm_layer = format<tensor_name>("norm{}", i);
        tensor out = layer_norm(m[norm_layer], r.x_out);
        out = ggml_reshape_4d(m, out, p.layers[i].n_features, r.w_out, r.h_out, b);
        outs[i] = out;
    }
    return outs;
}

constexpr int32_t bilinear_align_corners = GGML_SCALE_MODE_BILINEAR |
    (int)GGML_SCALE_FLAG_ALIGN_CORNERS;

tensor upscale_to_whcn(model_ref m, tensor x, tensor target) {
    return interpolate(m, x, {target->ne[0], target->ne[1]}, bilinear_align_corners);
}

tensor upscale_to(model_ref m, tensor x, tensor target) {
    auto [target_width, target_height, c, n] = nelements_whcn(m, target);
    x = contiguous_2d_to_whcn(m, x);
    x = interpolate(m, x, {target_width, target_height}, bilinear_align_corners);
    x = whcn_to_contiguous_2d(m, x);
    return x;
}

tensor downscale_by_whcn(model_ref m, tensor x, int f) {
    return interpolate(m, x, {x->ne[0] / f, x->ne[1] / f}, bilinear_align_corners);
}

tensor downscale_by(model_ref m, tensor x, int f) {
    x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    x = downscale_by_whcn(m, x, f);
    x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    return x;
}

swin_result encode_concat(model_ref m, swin_result& xs, swin_result& xs_low) {
    // TODO: implement cwhn upscale/interpolate which allows downscale & align_corners=True
    for (int i = 0; i < 4; ++i) {
        xs[i] = ggml_cont(m, permute_cwhn_to_whcn(m, xs[i]));
        xs_low[i] = permute_cwhn_to_whcn(m, xs_low[i]);
    }
    // clang-format off
    xs[0] = concat(m, {xs[0], upscale_to_whcn(m, xs_low[0], xs[0])}, 2);
    xs[1] = concat(m, {xs[1], upscale_to_whcn(m, xs_low[1], xs[1])}, 2);
    xs[2] = concat(m, {xs[2], upscale_to_whcn(m, xs_low[2], xs[2])}, 2);
    xs[3] = concat(m, {xs[3], upscale_to_whcn(m, xs_low[3], xs[3])}, 2);
    xs[3] = concat(m, {downscale_by_whcn(m, xs[0], 8),
                       downscale_by_whcn(m, xs[1], 4),
                       downscale_by_whcn(m, xs[2], 2),
                       xs[3]}, /*dim = */ 2);
    // clang-format on

    // whcn -> native
    for (int i = 0; i < 4; ++i) {
        xs[i] = whcn_to_contiguous_2d(m, xs[i]);
    }
    return xs;
}

swin_result encode(model_ref m, tensor x, swin_params const& p) {
    auto xs = swin_transformer(m["bb"], x, p);
    auto x_low = downscale_by(m, x, 2);
    auto xs_low = swin_transformer(m["bb"], x_low, p);
    encode_concat(m, xs, xs_low);
    return xs;
}

//
// Decoder
//

tensor conv_2d_batch_norm(model_ref m, tensor x, int stride = 1, int pad = 0) {
    return conv_2d(m, x, stride, pad); // batch_norm is fused into conv_2d at model conversion
}

tensor deformable_conv_2d(model_ref m, tensor x, int stride, int pad) {
    tensor offset = conv_2d(m["offset"], x, stride, pad);
    tensor modulator = conv_2d(m["modulator"], x, stride, pad);
    modulator = ggml_sigmoid_inplace(m, modulator);
    modulator = ggml_scale_inplace(m, modulator, 2.0f);

    x = conv_2d_deform(m, x, m.weights("conv.weight"), offset, modulator, stride, pad);
    return named(m, x);
}

tensor mean_2d(model_ref m, tensor x) {
    auto [w, h, c, n] = nelements_whcn(m, x);
    x = contiguous_2d_to_whcn(m, x);
    x = ggml_reshape_3d(m, x, w * h, c, n);
    x = ggml_mean(m, x);
    x = is_cwhn(m) ? ggml_reshape_4d(m, x, c, 1, 1, n) : ggml_reshape_4d(m, x, 1, 1, c, n);
    return x;
}

tensor global_avg_pool(model_ref m, tensor x) {
    x = mean_2d(m, x);
    x = conv_2d_batch_norm(m[1], x);
    x = ggml_relu_inplace(m, x);
    return named(m, x);
}

tensor aspp_module_deformable(model_ref m, tensor x, int padding) {
    x = deformable_conv_2d(m["conv"], x, 1, padding);
    x = batch_norm_2d(m["bn"], x);
    x = ggml_relu_inplace(m, x);
    return named(m, x);
}

tensor aspp_deformable(model_ref m, tensor x) {
    const int kernel_sizes[] = {1, 3, 7};
    const int channel_dim = is_cwhn(m) ? 0 : 2;

    tensor x1 = aspp_module_deformable(m["aspp1"], x);
    model_ref aspp_deforms = m["aspp_deforms"];
    tensor x_deforms[3];
    for (int i = 0; i < 3; ++i) {
        int padding = kernel_sizes[i] / 2;
        x_deforms[i] = aspp_module_deformable(aspp_deforms[i], x, padding);
    }
    tensor x5 = global_avg_pool(m["global_avg_pool"], x);
    auto [w1, h1, c, n] = nelements_whcn(m, x1);
    x5 = contiguous_2d_to_whcn(m, x5);
    x5 = interpolate(m, x5, {w1, h1}, bilinear_align_corners);
    x5 = whcn_to_contiguous_2d(m, x5);
    x = concat(m, {x1, x_deforms[0], x_deforms[1], x_deforms[2], x5}, channel_dim);

    x = conv_2d_batch_norm(m["conv1"], x);
    x = ggml_relu_inplace(m, x);
    return named(m, x);
}

tensor basic_decoder_block(model_ref m, tensor x) {
    x = conv_2d_batch_norm(m["conv_in"], x, 1, 1);
    x = ggml_relu_inplace(m, x);
    x = aspp_deformable(m["dec_att"], x);
    x = conv_2d_batch_norm(m["conv_out"], x, 1, 1);
    return named(m, x);
}

tensor simple_conv(model_ref m, tensor x) {
    x = conv_2d(m["conv1"], x, 1, 1);
    x = conv_2d(m["conv_out"], x, 1, 1);
    return named(m, x);
}

tensor image_to_patches(model_ref m, tensor x, int64_t out_w, int64_t out_h) {
    auto [w, h, c, b] = nelements(x);
    ASSERT(w % out_w == 0 && h % out_h == 0 && "Grid must divide image size");
    int64_t grid_w = w / out_w;
    int64_t grid_h = h / out_h;
    x = ggml_reshape_4d(m, x, out_w, grid_w, out_h, grid_h * c * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, out_w, out_h, grid_w * grid_h * c, b);
    return x;
}

tensor gdt_conv(model_ref m, tensor x) {
    x = conv_2d_batch_norm(m[0], x, 1, 1);
    x = ggml_relu_inplace(m, x);
    return x;
}

tensor decode(model_ref m, tensor x, swin_result const& features) {
    const int channel_dim = is_cwhn(m) ? 0 : 2;

    tensor x1 = features[0];
    tensor x2 = features[1];
    tensor x3 = features[2];
    tensor x4 = features[3];
    tensor x_whcn = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    if (is_whcn(m)) {
        x = x_whcn;
    }
    {
        auto [w, h, c, n] = nelements_whcn(m, x4);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk5"], patches);
        x4 = ggml_concat(m, x4, patches, channel_dim);
    }
    tensor p4 = basic_decoder_block(m["block4"], x4);
    tensor p4_gdt = gdt_conv(m["gdt_convs_4"], p4);
    tensor gdt_attn_4 = conv_2d(m["gdt_convs_attn_4.0"], p4_gdt);
    gdt_attn_4 = ggml_sigmoid(m, gdt_attn_4);
    p4 = ggml_mul(m, p4, gdt_attn_4);

    x3 = conv_2d(m["lateral_block4.conv"], x3);
    tensor _p4 = upscale_to(m, p4, x3);
    tensor _p3 = ggml_add_inplace(m, _p4, x3);

    {
        auto [w, h, c, n] = nelements_whcn(m, _p3);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk4"], patches);
        _p3 = ggml_concat(m, _p3, patches, channel_dim);
    }
    tensor p3 = basic_decoder_block(m["block3"], _p3);
    tensor p3_gdt = gdt_conv(m["gdt_convs_3"], p3);
    tensor gdt_attn_3 = conv_2d(m["gdt_convs_attn_3.0"], p3_gdt);
    gdt_attn_3 = ggml_sigmoid(m, gdt_attn_3);
    p3 = ggml_mul(m, p3, gdt_attn_3);

    _p3 = upscale_to(m, p3, x2);
    x2 = conv_2d(m["lateral_block3.conv"], x2);
    tensor _p2 = ggml_add_inplace(m, _p3, x2);

    {
        auto [w, h, c, n] = nelements_whcn(m, _p2);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk3"], patches);
        _p2 = ggml_concat(m, _p2, patches, channel_dim);
    }
    tensor p2 = basic_decoder_block(m["block2"], _p2);
    tensor p2_gdt = gdt_conv(m["gdt_convs_2"], p2);
    tensor gdt_attn2 = conv_2d(m["gdt_convs_attn_2.0"], p2_gdt);
    gdt_attn2 = ggml_sigmoid(m, gdt_attn2);
    p2 = ggml_mul(m, p2, gdt_attn2);

    _p2 = upscale_to(m, p2, x1);
    x1 = conv_2d(m["lateral_block2.conv"], x1);
    tensor _p1 = ggml_add_inplace(m, _p2, x1);

    {
        auto [w, h, c, n] = nelements_whcn(m, _p1);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk2"], patches);
        _p1 = ggml_concat(m, _p1, patches, channel_dim);
    }
    _p1 = basic_decoder_block(m["block1"], _p1);
    _p1 = upscale_to(m, _p1, x);
    tensor p1_ipt = simple_conv(m["ipt_blk1"], x);
    _p1 = ggml_concat(m, _p1, p1_ipt, channel_dim);

    tensor p1_out = conv_2d(m["conv_out1.0"], _p1);
    p1_out = ggml_sigmoid_inplace(m, p1_out);

    return named(m, p1_out);
}

} // namespace birefnet

tensor birefnet_predict(model_ref m, tensor image, birefnet_params const& p) {
    // Encoder
    birefnet::swin_result features = birefnet::encode(m, image, p.encoder);
    // Squeeze block
    features[3] = birefnet::basic_decoder_block(m["squeeze_module.0"], features[3]);
    // Decoder
    tensor scaled_preds = birefnet::decode(m["decoder"], image, features);

    return compute_graph_output(m, scaled_preds);
}

image_data birefnet_process_input(image_view image, birefnet_params const& p) {
    constexpr f32x4 mean = f32x4{0.485f, 0.456f, 0.406f, 0.f};
    constexpr f32x4 std = f32x4{0.229f, 0.224f, 0.225f, 1.f};

    image_data resized;
    if (image.extent != p.image_extent) {
        resized = image_scale(image, p.image_extent);
        image = image_view(resized);
    }

    return image_u8_to_f32(image, image_format::rgb_f32, -mean, 1.f / std);
}

image_data birefnet_process_output(
    span<float const> mask_data, i32x2 target_extent, birefnet_params const& p) {

    image_view mask_output(p.image_extent, mask_data);
    image_data mask_resized;
    if (p.image_extent != target_extent) {
        mask_resized = image_scale(mask_output, target_extent);
        mask_output = mask_resized;
    }
    return image_f32_to_u8(mask_output, image_format::alpha_u8);
}

birefnet_buffers birefnet_precompute(model_ref m, birefnet_params const& params) {
    int w = params.encoder.window_size;
    int width = params.image_extent[0] / 4;
    int height = params.image_extent[1] / 4;

    birefnet_buffers b;
    b[0] = birefnet::create_relative_position_index(m, w);
    for (int i = 0; i < swin_params::n_layers + 1; ++i) {
        b[i + 1] = birefnet::create_attention_mask(m, width >> i, height >> i, w);
    }
    return b;
}

// clang-format off
const swin_params swin_t_params = {
    .embed_dim = 96,
    .window_size = 7,
    .layers = {
        //       depth  n_heads   n_features   downsample
        swin_layer_t{2,    3,        96 * 1,     true},
        swin_layer_t{2,    6,        96 * 2,     true},
        swin_layer_t{6,    12,       96 * 4,     true},
        swin_layer_t{2,    24,       96 * 8,     false}}};

const swin_params swin_l_params = {
    .embed_dim = 192,
    .window_size = 12,
    .layers = {
        //       depth  n_heads   n_features   downsample
        swin_layer_t{2,    6,        192 * 1,     true},
        swin_layer_t{2,    12,       192 * 2,     true},
        swin_layer_t{18,   24,       192 * 4,     true},
        swin_layer_t{2,    48,       192 * 8,     false}}};
// clang-format on

swin_params swin_detect_params(model_file const& f) {
    int embed_dim = f.get_int("swin.embed_dim");
    if (embed_dim == 96) {
        return swin_t_params;
    } else if (embed_dim == 192) {
        return swin_l_params;
    } else {
        throw except("Unsupported Swin Transformer embed dim: {}", embed_dim);
    }
}

i32x2 birefnet_image_extent(i32x2 input_extent, birefnet_params const& p, size_t max_alloc) {
    i32x2 extent{p.image_size, p.image_size};
    if (p.image_size == -1) {
        ASSERT(input_extent[0] > 0 && input_extent[1] > 0);
        // largest layer in BiRefNet-dynamic is input for 240-channel conv-2d at full resolution
        size_t req_alloc = size_t(input_extent[0]) * input_extent[1] * 240ULL * sizeof(float);
        if (req_alloc > max_alloc) {
            float scale = std::sqrt(float(max_alloc) / float(req_alloc));
            input_extent = {
                std::max(1, int(input_extent[0] * scale) - p.image_multiple),
                std::max(1, int(input_extent[1] * scale) - p.image_multiple)};
        }
        extent = {
            next_multiple(input_extent[0], p.image_multiple),
            next_multiple(input_extent[1], p.image_multiple)};
    }
    return extent;
}

birefnet_params birefnet_detect_params(
    model_file const& f, i32x2 dynamic_extent, size_t max_alloc) {

    if (std::string_view arch = f.arch(); arch != "birefnet") {
        throw except("Architecture expected to be 'birefnet', but was '{}' ({})", arch, f.path);
    }
    birefnet_params p;
    p.image_size = f.get_int("birefnet.image_size");
    p.image_multiple = f.get_int("birefnet.image_multiple");
    p.image_extent = birefnet_image_extent(dynamic_extent, p, max_alloc);
    p.encoder = swin_detect_params(f);
    return p;
}

} // namespace visp
