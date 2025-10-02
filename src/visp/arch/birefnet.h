#pragma once

#include "visp/ml.h"
#include "visp/image.h"

#include <array>

namespace visp {
namespace birefnet {

// SWIN Transformer

struct swin_block_params {
    int n_heads = 6;
    int window_size = 7;
    int64_t w = 0;
    int64_t h = 0;
    int shift = 0;
};

struct swin_layer_result {
    tensor x_out;
    int64_t w_out;
    int64_t h_out;
    tensor x_down;
    int64_t w_down;
    int64_t h_down;
};

using swin_result = std::array<tensor, swin_params::n_layers>;

void compute_relative_position_index(span<int32_t> dst, int window_size);
tensor_data create_relative_position_index(ggml_context* ctx, int window_size);
void compute_attention_mask(std::span<float> out, int64_t w, int64_t h, int window_size);
tensor_data create_attention_mask(ggml_context* ctx, int64_t w, int64_t h, int window_size);

tensor mlp(model_ref m, tensor x);
tensor patch_merging(model_ref m, tensor x, int64_t w, int64_t h);
tensor window_partition(model_ref m, tensor x, int window);
tensor window_reverse(model_ref m, tensor x, int w, int h, int window);
tensor window_attention(model_ref m, tensor x, tensor mask, int num_heads, int window);
tensor swin_block(model_ref m, tensor x, tensor mask, swin_block_params const&);
swin_layer_result swin_layer(
    model_ref m, tensor x, int64_t w, int64_t h, swin_layer_t const&, int window_size);
swin_result swin_transformer(model_ref m, tensor x, swin_params const& p);

// Encoder

swin_result encode_concat(model_ref m, swin_result& xs, swin_result& xs_low);
swin_result encode(model_ref m, tensor x, swin_params const& p);

// Decoder

tensor deformable_conv_2d(model_ref m, tensor x, int stride = 1, int pad = 0);
tensor mean_2d(model_ref m, tensor x);
tensor global_avg_pool(model_ref m, tensor x);
tensor aspp_module_deformable(model_ref m, tensor x, int padding = 0);
tensor aspp_deformable(model_ref m, tensor x);
tensor basic_decoder_block(model_ref m, tensor x);
tensor simple_conv(model_ref m, tensor x);
tensor image_to_patches(model_ref m, tensor x, int64_t out_w, int64_t out_h);
tensor gdt_conv(model_ref m, tensor x);
tensor decode(model_ref m, tensor x, swin_result const& features);

} // namespace birefnet
} // namespace visp