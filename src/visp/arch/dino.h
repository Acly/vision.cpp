#pragma once

#include "util/math.h"
#include "visp/ml.h"
#include "visp/vision.h"

#include <vector>

namespace visp::dino {

tensor interpolate_pos_encoding(model_ref m, tensor x, int64_t w, int64_t h, int patch_size);
tensor prepare_tokens(model_ref m, tensor x, int patch_size);
tensor layer_scale(model_ref m, tensor x);
tensor mlp(model_ref m, tensor x);
tensor self_attention(model_ref m, tensor x, int n_heads);
tensor layer(model_ref m, tensor x, dino_params const& p);

std::vector<tensor> get_intermediate_layers(
    model_ref m, tensor x, std::span<int const> layers, dino_params const& p);

} // namespace visp::dino
