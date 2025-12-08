#include "visp/arch/swin.h"
#include "util/string.h"
#include "visp/nn.h"

namespace visp {
namespace swin {

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

tensor window_attention(model_ref m, tensor x, tensor mask, int n_heads, int window) {
    auto [c, n, b, _] = nelements(x);

    tensor_name rel_pos_name = format<tensor_name>("window_attention_{}.rel_pos_index", window);
    tensor rel_pos_index = ggml_get_tensor(m, rel_pos_name.c_str());
    tensor rel_pos_table = m.weights("relative_position_bias_table");
    tensor rel_pos_bias = ggml_get_rows(m, rel_pos_table, rel_pos_index);
    rel_pos_bias = ggml_reshape_4d(m, rel_pos_bias, n_heads, n, n, 1);
    rel_pos_bias = ggml_permute(m, rel_pos_bias, 2, 0, 1, 3); // [n, n, n_heads, 1]
    rel_pos_bias = ggml_cast(m, rel_pos_bias, GGML_TYPE_F16); // get_rows result is always f32

    tensor attn_mask = rel_pos_bias;
    if (mask) {
        int64_t n_windows = mask->ne[2];
        if (b > n_windows) { // if there are multiple images in the batch
            mask = ggml_reshape_4d(m, mask, n, n, n_windows, 1);
            mask = ggml_repeat_4d(m, mask, n, n, n_windows, b / n_windows);
        }
        mask = ggml_reshape_4d(m, mask, n, n, 1, b);
        mask = ggml_repeat_4d(m, mask, n, n, n_heads, b); // can only broadcast one operand in add
        attn_mask = ggml_add(m, mask, attn_mask);         // [n, n, n_heads, b] + [n, n, n_heads, 1]
    }

    auto [q, k, v] = split_qkv(m["qkv"], x, n_heads, 2);
    float scale = 1.0f / std::sqrt(float(c / n_heads));
    x = attention(m, q, k, v, attn_mask, scale, m["proj"]);

    return named(m, x);
}

tensor block(model_ref m, tensor x, tensor mask, block_params const& p) {
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

constexpr uint16_t neg_inf_f16 = 0xfc00; // -infinity in IEEE 754 half-precision

void compute_attention_mask(span<byte> out_bytes, int64_t w, int64_t h, int window_size) {
    uint16_t* out = reinterpret_cast<uint16_t*>(out_bytes.data());
    int n = window_size;
    int n2 = n * n;
    int n4 = n2 * n2;
    int shift = window_size / 2;
    int64_t nw_x = (w + n - 1) / n;
    int64_t nw_y = (h + n - 1) / n;
    int64_t w_pad = nw_x * n;
    int64_t h_pad = nw_y * n;

    std::memset(out, 0, out_bytes.size());

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
                            // If not, set attention mask to -inf so it is ignored by softmax
                            if (!match_y || !match_x) {
                                int64_t idx = base + (y0 * n + x0) * n2 + (y1 * n + x1);
                                out[idx] = neg_inf_f16;
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
    auto result = tensor_alloc(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n * n, n * n, nw_x * nw_y));
    auto name = format<tensor_name>("swin_layer_{}x{}.attn_mask", w, h);
    compute_attention_mask(result.as_bytes(), w, h, window_size);
    make_constant(result.x, name);
    return result;
}

layer_result layer(
    model_ref m, tensor x, int64_t w, int64_t h, swin_layer_t const& p, int window, bool down) {
    // Attention masks need to be precomputed
    tensor_name attn_mask_name = format<tensor_name>("swin_layer_{}x{}.attn_mask", w, h);
    tensor attn_mask = ggml_get_tensor(m, attn_mask_name.c_str());

    model_ref blocks = m["blocks"];
    for (int i = 0; i < p.depth; ++i) {
        x = block(
            blocks[i], x, attn_mask,
            {.n_heads = p.n_heads,
             .window_size = window,
             .w = w,
             .h = h,
             .shift = i % 2 == 0 ? 0 : window / 2});
    }
    if (down) {
        tensor x_down = patch_merging(m["downsample"], x, w, h);
        return {x, w, h, x_down, (w + 1) / 2, (h + 1) / 2};
    }
    return {x, w, h, x, w, h};
}

swin_result encode(model_ref m, tensor x, swin_params const& p) {
    x = patch_embed(m["patch_embed"], x, 4);

    auto [c, w, h, b] = nelements(x);
    x = ggml_reshape_3d(m, x, c, w * h, b);

    layer_result r{x, w, h, x, w, h};
    swin_result outs = {};

    for (int i = 0; i < swin_n_layers; ++i) {
        bool downsample = (i < swin_n_layers - 1);
        r = layer(
            m["layers"][i], r.x_down, r.w_down, r.h_down, p.layers[i], p.window_size, downsample);

        tensor_name norm_layer = format<tensor_name>("norm{}", i);
        tensor out = layer_norm(m[norm_layer], r.x_out);
        out = ggml_reshape_4d(m, out, p.layers[i].n_features, r.w_out, r.h_out, b);
        outs[i] = out;
    }
    return outs;
}

} // namespace swin

// clang-format off
const swin_params swin_t_params = {
    .embed_dim = 96,
    .window_size = 7,
    .layers = {
        //       depth  n_heads   n_features
        swin_layer_t{2,    3,        96 * 1},
        swin_layer_t{2,    6,        96 * 2},
        swin_layer_t{6,    12,       96 * 4},
        swin_layer_t{2,    24,       96 * 8}}};

const swin_params swin_l_params = {
    .embed_dim = 192,
    .window_size = 12,
    .layers = {
        //       depth  n_heads   n_features
        swin_layer_t{2,    6,        192 * 1},
        swin_layer_t{2,    12,       192 * 2},
        swin_layer_t{18,   24,       192 * 4},
        swin_layer_t{2,    48,       192 * 8}}};
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

swin_buffers swin_precompute(model_ref m, i32x2 image_extent, swin_params const& p) {
    int w = p.window_size;
    int width = image_extent[0] / 4;
    int height = image_extent[1] / 4;

    swin_buffers b;
    b[0] = swin::create_relative_position_index(m, w);
    for (int i = 0; i < swin_n_layers + 1; ++i) {
        b[i + 1] = swin::create_attention_mask(m, width >> i, height >> i, w);
    }
    return b;
}

swin_result swin_encode(model_ref m, tensor image, swin_params const& p) {
    return swin::encode(m, image, p);
}

} // namespace visp