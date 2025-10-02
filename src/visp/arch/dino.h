#pragma once

#include "util/math.h"
#include "visp/image.h"
#include "visp/ml.h"
#include "visp/nn.h"

#include <vector>

#pragma optimize("", off)

namespace visp {
namespace dino {

inline tensor interpolate_pos_encoding(
    model_ref m, tensor x, int64_t w, int64_t h, int patch_size) {

    tensor pos_embed = ggml_cast(m, m.weights("pos_embed"), GGML_TYPE_F32);
    int64_t n_patch = x->ne[1] - 1;
    int64_t n = pos_embed->ne[1] - 1;
    if (n_patch == n && w == h) {
        return pos_embed;
    }

    tensor class_embed = slice(m, pos_embed, {}, {0}, {}, {});
    tensor patch_embed = slice(m, pos_embed, {}, {1, n + 1}, {}, {});
    int64_t dim = x->ne[0];
    i64x2 target = i64x2{w, h} / patch_size;
    int64_t sqrt_n = int64_t(std::sqrt(float(n)) + 0.01f);

    patch_embed = ggml_reshape_4d(m, patch_embed, dim, sqrt_n, sqrt_n, 1);
    patch_embed = ggml_cont(m, permute_cwhn_to_whcn(m, patch_embed));
    patch_embed = interpolate(m, patch_embed, target, GGML_SCALE_MODE_BICUBIC);
    patch_embed = ggml_cont(m, permute_whcn_to_cwhn(m, patch_embed));
    patch_embed = ggml_reshape_3d(m, patch_embed, dim, target[0] * target[1], 1);
    return concat(m, {class_embed, patch_embed}, 1);
}

inline tensor prepare_tokens(model_ref m, tensor x, int patch_size) {
    auto [c, w, h, n] = nelements(x);
    x = patch_embed(m["patch_embed"], x, patch_size);
    x = ggml_reshape_3d(m, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);

    tensor cls_token = m.weights("cls_token");
    if (cls_token->ne[2] != n) {
        cls_token = ggml_repeat_4d(m, cls_token, cls_token->ne[0], 1, n, 1);
    }
    x = concat(m, {cls_token, x}, 1);

    tensor pos_enc = interpolate_pos_encoding(m, x, w, h, patch_size);
    x = ggml_add_inplace(m, x, pos_enc);
    return x;
}

inline tensor layer_scale(model_ref m, tensor x) {
    return ggml_mul(m, x, m.weights("gamma"));
}

inline tensor mlp(model_ref m, tensor x) {
    x = linear(m["fc1"], x);
    x = ggml_gelu(m, x);
    x = linear(m["fc2"], x);
    return x;
}

inline tensor attention(model_ref m, tensor x, int n_heads, bool flash_attn) {
    auto [c, n, b, _] = nelements(x);
    float scale = 1.0f / std::sqrt(float(c) / float(n_heads));

    tensor qkv = linear(m["qkv"], x);
    qkv = ggml_reshape_4d(m, qkv, c / n_heads, n_heads, 3, n * b);
    qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 1, 3, 2));

    auto split = [=](tensor tensor, size_t index, bool transpose = false) mutable {
        tensor = slice(m, tensor, {}, {}, {}, index);
        tensor = ggml_reshape_4d(m, tensor, c / n_heads, n_heads, n, b);
        if (transpose) {
            tensor = ggml_cont(m, ggml_permute(m, tensor, 1, 2, 0, 3));
        } else {
            tensor = ggml_cont(m, ggml_permute(m, tensor, 0, 2, 1, 3));
        }
        return tensor;
    };
    tensor q = split(qkv, 0);
    tensor k = split(qkv, 1);
    tensor v = split(qkv, 2, !flash_attn);

    if (flash_attn) {
        int64_t c_pad = GGML_PAD(c, 4) - c;
        int64_t n_pad = GGML_PAD(n, 32) - n;
        q = ggml_pad(m, q, c_pad, n_pad, 0, 0);
        k = ggml_pad(m, k, c_pad, n_pad, 0, 0);
        v = ggml_pad(m, v, c_pad, n_pad, 0, 0);

        ggml_type dtype = m.weights("qkv.weight")->type;
        k = ggml_cast(m, k, dtype);
        v = ggml_cast(m, v, dtype);

        x = ggml_flash_attn_ext(m, q, k, v, nullptr, scale, 0.0f, 0.0f);
        x = slice(m, x, {}, {}, {0, n}, {});
    } else {
        q = ggml_scale_inplace(m, q, scale);

        tensor attn = ggml_mul_mat(m, k, q);
        attn = ggml_soft_max(m, attn);

        x = ggml_mul_mat(m, v, attn);
        x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
        x = ggml_reshape_3d(m, x, c, n, b);
    }

    x = linear(m["proj"], x);
    return named(m, x);
}

struct dino_params {
    int patch_size = 16;
    int embed_dim = 384;
    int n_blocks = 12;
    int n_heads = 6;
    int mlp_ratio = 4;
    bool flash_attention = false;
};

inline tensor block(model_ref m, tensor x, dino_params const& p) {
    tensor attn = x;
    attn = layer_norm(m["norm1"], attn);
    attn = attention(m["attn"], attn, p.n_heads, p.flash_attention);
    attn = layer_scale(m["ls1"], attn);
    x = ggml_add_inplace(m, x, attn);

    tensor ffn = x;
    ffn = layer_norm(m["norm2"], ffn);
    ffn = mlp(m["mlp"], ffn);
    ffn = layer_scale(m["ls2"], ffn);
    x = ggml_add_inplace(m, x, ffn);

    return named(m, x);
}

inline std::vector<tensor> get_intermediate_layers(
    model_ref m, tensor x, int n, dino_params const& p) {
        
    x = prepare_tokens(m, x, p.patch_size);

    std::vector<tensor> outputs;
    model_ref blocks = m["blocks"];
    for (int i = 0; i < p.n_blocks; ++i) {
        x = block(blocks[i], x, p);
        if (i >= p.n_blocks - n) {
            outputs.push_back(x);
        }
    }
    return outputs;
}

} // namespace dino
} // namespace visp
