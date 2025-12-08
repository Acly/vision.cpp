#pragma once

#include "visp/image.h"
#include "visp/ml.h"
#include "visp/vision.h"

#include <array>
#include <span>
#include <tuple>
#include <vector>

namespace visp::sam {

// Image encoder

struct tiny_vit_params {

    struct layer {
        int resolution;
        int embed_dim;
        int depth;
        int num_heads;
        int window_size;
        bool downsample;
    };

    static constexpr int num_layers = 4;

    int img_size = 1024;
    // clang-format off
    std::array<layer, num_layers> layers = {
        // resolution   dim     depth   attn heads  window size   downsample
        layer{256,      64,     2,      2,          7,              true},
        layer{128,      128,    2,      4,          7,              true},
        layer{64,       160,    6,      5,          14,             true},
        layer{64,       320,    2,      10,         7,              false}};
    // clang-format on
};

float resize_longest_side(i32x2 extent, int target_longest_side);

tensor patch_embed(model_ref m, tensor x);
tensor mb_conv(model_ref m, tensor x);
tensor patch_merging(model_ref m, tensor x);
tensor mlp(model_ref m, tensor x);
tensor attention_rel_bias(model_ref m, tensor x, int dim, int num_heads);
tensor window_partition(model_ref m, tensor x, int window);
tensor window_reverse(model_ref m, tensor x, int w, int h, int window);
tensor tiny_vit_block(
    model_ref m, tensor x, int input_resolution, int dim, int num_heads, int window_size);
tensor conv_layer(model_ref m, tensor x, tiny_vit_params::layer p);
tensor basic_layer(model_ref m, tensor x, tiny_vit_params::layer const& p);

tensor tiny_vit(model_ref m, tensor x, tiny_vit_params const& p);

// Prompt encoder

tensor embed_points(model_ref m, tensor coords);
tensor embed_box(model_ref m, tensor coords);
tensor no_mask_embed(model_ref m);

float transform_coord(int p, float scale, int image_size = 1024);
tensor position_embedding_random(model_ref m, tensor coords);

// Mask decoder

tensor mlp_block(model_ref m, tensor x);
tensor separate_attention_heads(model_ref m, tensor x, int num_heads);
tensor decoder_attention(model_ref m, tensor q, tensor k, tensor v, int num_heads);
std::tuple<tensor, tensor> two_way_attention_block(
    model_ref m,
    tensor queries,
    tensor keys,
    tensor query_pe,
    tensor key_pe,
    int num_heads,
    bool skip_first_layer_pe = false);
std::tuple<tensor, tensor> two_way_transformer(
    model_ref m,
    tensor image_embedding,
    tensor image_pe,
    tensor point_embedding,
    int depth,
    int num_heads);
tensor upscale_outputs(model_ref m, tensor x);
tensor hypernetwork_mlp(model_ref m, tensor x, int num_layers);

sam_prediction predict_masks(
    model_ref, tensor image_embed, tensor sparse_prompt, tensor dense_prompt);

} // namespace visp::sam