#include "esrgan.hpp"
#include "nn.hpp"
#include "util/string.hpp"
#include "visp/vision.hpp"

#include <charconv>
#include <string_view>

namespace visp {
namespace esrgan {

tensor upsample(model_ref m, tensor x) {
    auto [c, w, h, n] = nelements(x);
    x = ggml_interpolate(m, x, int(c), int(w * 2), int(h * 2), int(n), GGML_SCALE_MODE_NEAREST);
    x = conv_2d(m, x, 1, 1);
    x = ggml_leaky_relu(m, x, 0.2f, true);
    return named(m, x);
}

tensor conv_block(model_ref m, tensor x) {
    x = conv_2d(m[0], x, 1, 1);
    x = ggml_leaky_relu(m, x, 0.2f, true);
    return x;
}

tensor risidual_dense_block(model_ref m, tensor x) {
    tensor x1 = conv_block(m["conv1"], x);
    tensor c1 = concat(m, {x, x1}, 0);
    tensor x2 = conv_block(m["conv2"], c1);
    tensor c2 = concat(m, {c1, x2}, 0);
    tensor x3 = conv_block(m["conv3"], c2);
    tensor c3 = concat(m, {c2, x3}, 0);
    tensor x4 = conv_block(m["conv4"], c3);
    tensor c4 = concat(m, {c3, x4}, 0);
    tensor x5 = conv_2d(m["conv5.0"], c4, 1, 1);
    x5 = ggml_scale_inplace(m, x5, 0.2f);
    x = ggml_add(m, x, x5);
    return named(m, x);
}

tensor rrdb(model_ref m, tensor x) {
    tensor x_in = x;
    x = risidual_dense_block(m["RDB1"], x);
    x = risidual_dense_block(m["RDB2"], x);
    x = risidual_dense_block(m["RDB3"], x);
    x = ggml_scale_inplace(m, x, 0.2f);
    x = ggml_add(m, x, x_in);
    return named(m, x);
}

} // namespace esrgan

tensor esrgan_generate(model_ref m, tensor x, esrgan_params const& p) {
    m = m["model"];
    x = conv_2d(m[0], x, 1, 1);

    tensor sub = x;
    model_ref block = m[1]["sub"];
    for (int i = 0; i < p.n_blocks; ++i) {
        sub = esrgan::rrdb(block[i], sub);
    }
    sub = conv_2d(block[p.n_blocks], sub, 1, 1);
    x = ggml_add(m, x, sub);

    int seq = 2;
    for (int i = 0; i < log2(p.scale); ++i) {
        x = esrgan::upsample(m[seq + 1], x);
        seq += 3;
    }
    x = conv_2d(m[seq], x, 1, 1);
    x = ggml_leaky_relu(m, x, 0.2f, true);
    x = conv_2d(m[seq + 2], x, 1, 1);

    return compute_graph_output(m, x, "result");
}

esrgan_params esrgan_detect_params(model_ref m) {
    esrgan_params p;
    p.n_blocks = 0;
    int model_len = 0;

    ggml_context* ctx = m.weights_context;
    for (tensor t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto name = std::string_view(ggml_get_name(t));
        if (name.starts_with("model.")) {
            name = name.substr(6);
            int x = 0;
            auto r = std::from_chars(name.data(), name.data() + 2, x);
            model_len = std::max(model_len, x + 1);

            size_t i_dot = name.find('.');
            if (i_dot == std::string_view::npos) {
                continue;
            }
            name = name.substr(i_dot + 1, 11);
            if (name.starts_with("sub.") && (name.ends_with("RDB1") || name.ends_with("RDB1."))) {
                r = std::from_chars(name.data() + 4, name.data() + 6, x);
                p.n_blocks = std::max(p.n_blocks, x + 1);
            }
        }
    }
    // 3 layers per upscale block, each upscales x2, 5 blocks for the rest of the model
    p.scale = 1 << ((model_len - 5) / 3);
    if (p.scale < 2 || p.scale > 4) {
        throw except("Unsupported scale: {}", p.scale);
    }
    if (p.n_blocks < 1 || p.n_blocks > 23) {
        throw except("Invalid number of blocks: {}", p.n_blocks);
    }
    return p;
}

int esrgan_estimate_graph_size(esrgan_params const& p) {
    // worst-case estimate, exact number depends on how conv-2d is implemented for the backend
    return 512 + p.n_blocks * 192;
}

} // namespace visp