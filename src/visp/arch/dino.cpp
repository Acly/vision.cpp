#include "visp/arch/dino.h"
#include "util/math.h"
#include "visp/ml.h"
#include "visp/nn.h"

namespace visp {
namespace dino {

tensor interpolate_pos_encoding(model_ref m, tensor x, int64_t w, int64_t h, int patch_size) {
    tensor pos_embed = m.weights("position_embeddings");
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

tensor prepare_tokens(model_ref m, tensor x, int patch_size) {
    auto [c, w, h, n] = nelements(x);
    x = patch_embed(m["patch_embeddings"], x, patch_size);
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

tensor layer_scale(model_ref m, tensor x) {
    return ggml_mul(m, x, m.weights("lambda1"));
}

tensor mlp(model_ref m, tensor x) {
    x = linear(m["fc1"], x);
    x = ggml_gelu(m, x);
    x = linear(m["fc2"], x);
    return x;
}

tensor self_attention(model_ref m, tensor x, int n_heads) {
    auto [c, n, b, _] = nelements(x);
    auto project = [&](model_ref m, tensor t) {
        t = linear(m, t);
        t = ggml_reshape_4d(m, t, c / n_heads, n_heads, n, b);
        return t;
    };

    tensor q = project(m["attention.query"], x);
    tensor k = project(m["attention.key"], x);
    tensor v = project(m["attention.value"], x);

    float scale = 1.0f / std::sqrt(float(c) / float(n_heads));
    x = attention(m, q, k, v, nullptr, scale, m["output.dense"]);
    return x;
}

tensor layer(model_ref m, tensor x, dino_params const& p) {
    tensor attn = x;
    attn = layer_norm(m["norm1"], attn, 1e-6f);
    attn = self_attention(m["attention"], attn, p.n_heads);
    attn = layer_scale(m["layer_scale1"], attn);
    x = ggml_add(m, x, attn);

    tensor ffn = x;
    ffn = layer_norm(m["norm2"], ffn, 1e-6f);
    ffn = mlp(m["mlp"], ffn);
    ffn = layer_scale(m["layer_scale2"], ffn);
    x = ggml_add(m, x, ffn);

    return named(m, x);
}

template <typename T>
bool contains(std::span<const T> r, T const& value) {
    return std::find(r.begin(), r.end(), value) != r.end();
}

std::vector<tensor> get_intermediate_layers(
    model_ref m, tensor x, std::span<const int> layers, dino_params const& p) {

    x = prepare_tokens(m["embeddings"], x, p.patch_size);

    std::vector<tensor> outputs;
    model_ref encoder = m["encoder.layer"];
    for (int i = 0; i < p.n_layers; ++i) {
        x = layer(encoder[i], x, p);

        if (contains(layers, i)) {
            tensor out = layer_norm(m["layernorm"], x, 1e-6f);
            ggml_format_name(out, "dino_layer_%d", i);
            ggml_build_forward_expand(m.graph, out);
            outputs.push_back(out);
        }
    }
    return outputs;
}

} // namespace dino

std::vector<tensor> dino_get_intermediate_layers(
    model_ref m, tensor x, std::span<const int> layers, dino_params const& p) {
    return dino::get_intermediate_layers(m, x, layers, p);
}

dino_params dino_detect_params(model_file const& file) {
    dino_params p{};
    p.patch_size = file.get_int("dino.patch_size");
    p.embed_dim = file.get_int("dino.embed_dim");
    p.n_heads = file.get_int("dino.n_heads");
    p.n_layers = file.get_int("dino.n_layers");
    return p;
}

} // namespace visp
