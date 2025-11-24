#include "visp/arch/esrgan.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <charconv>
#include <string_view>

namespace visp {
namespace esrgan {

tensor upsample(model_ref m, tensor x) {
    auto [w, h, c, n] = nelements_whcn(m, x);
    x = interpolate(m, x, {w * 2, h * 2}, GGML_SCALE_MODE_NEAREST);
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
    int dim = (m.flags & model_build_flag::cwhn) ? 0 : 2;
    tensor x1 = conv_block(m["conv1"], x);
    tensor c1 = concat(m, {x, x1}, dim);
    tensor x2 = conv_block(m["conv2"], c1);
    tensor c2 = concat(m, {c1, x2}, dim);
    tensor x3 = conv_block(m["conv3"], c2);
    tensor c3 = concat(m, {c2, x3}, dim);
    tensor x4 = conv_block(m["conv4"], c3);
    tensor c4 = concat(m, {c3, x4}, dim);
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
    x = cwhn_to_contiguous_2d(m, x);
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

    x = contiguous_2d_to_cwhn(m, x);
    return compute_graph_output(m, x, "result");
}

esrgan_params esrgan_detect_params(model_file const& f) {
    if (std::string_view arch = f.arch(); arch != "esrgan") {
        throw except("Architecture expected to be 'esrgan', but was '{}' ({})", arch, f.path);
    }
    esrgan_params p;
    p.scale = f.get_int("esrgan.scale");
    p.n_blocks = f.get_int("esrgan.block_count");
    
    if (p.scale < 1 || p.scale > 8) {
        throw except("ESRGAN: unsupported scale: {}", p.scale);
    }
    if (p.n_blocks < 1 || p.n_blocks > 23) {
        throw except("ESRGAN: invalid number of blocks: {}", p.n_blocks);
    }
    return p;
}

int esrgan_estimate_graph_size(esrgan_params const& p) {
    // worst-case estimate, exact number depends on how conv-2d is implemented for the backend
    return 512 + p.n_blocks * 192;
}

} // namespace visp
