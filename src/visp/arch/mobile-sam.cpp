#include "visp/arch/mobile-sam.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <ggml.h>

#include <cmath>
#include <optional>

namespace visp {
namespace sam {

tensor conv_2d_batch_norm(model_ref m, tensor x, int stride = 1, int pad = 0) {
    // batch_norm is fused into conv_2d when converting the model
    return conv_2d(m["c"], x, stride, pad);
}

tensor conv_2d_depthwise_batch_norm(model_ref m, tensor x, int stride = 1, int pad = 0) {
    // batch_norm is fused into conv_2d_depthwise when converting the model
    return conv_2d_depthwise(m["c"], x, stride, pad);
}

tensor window_partition(model_ref m, tensor x, int window) {
    auto [c, w, h, b] = nelements(x);
    if (m.flags & model_build_flag::window_partition) {
        x = ggml_win_part(m, x, window);
        x = ggml_reshape_3d(m, x, c, window * window, x->ne[3]);
        return x;
    }
    int64_t px = (window - w % window) % window;
    int64_t py = (window - h % window) % window;
    int64_t npw = (w + px) / window;
    int64_t nph = (h + py) / window;

    if (px > 0 || py > 0) {
        x = ggml_pad(m, x, 0, int(px), int(py), 0);
    }
    x = ggml_reshape_4d(m, x, c * window, npw, window, nph * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, window * window, npw * nph * b);
    return x;
}

tensor window_reverse(model_ref m, tensor x, int w, int h, int window) {
    int64_t c = x->ne[0];
    int64_t b = x->ne[3];
    if (m.flags & model_build_flag::window_partition) {
        x = ggml_reshape_4d(m, x, c, window, window, x->ne[2]);
        x = ggml_win_unpart(m, x, w, h, window);
        return x;
    }
    int64_t px = (window - w % window) % window;
    int64_t py = (window - h % window) % window;
    int64_t npw = (w + px) / window;
    int64_t nph = (h + py) / window;

    x = ggml_reshape_4d(m, x, c * window, window, npw, nph * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, c, w + px, h + py, b);
    x = slice(m, x, {}, {0, w}, {0, h});
    x = ggml_cont(m, x);
    return x;
}

//
// Image encoder
//

tensor patch_embed(model_ref m, tensor x) {
    x = conv_2d_batch_norm(m["seq.0"], x, 2, 1);
    x = ggml_gelu_inplace(m, x);
    x = conv_2d_batch_norm(m["seq.2"], x, 2, 1);
    return named(m, x);
}

tensor mb_conv(model_ref m, tensor x) {
    tensor shortcut = x;

    x = conv_2d_batch_norm(m["conv1"], x);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_depthwise_batch_norm(m["conv2"], x, 1, 1);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_batch_norm(m["conv3"], x);
    x = ggml_add_inplace(m, x, shortcut);
    x = ggml_gelu_inplace(m, x);

    return named(m, x);
}

tensor patch_merging(model_ref m, tensor x) {
    x = conv_2d_batch_norm(m["conv1"], x);
    x = ggml_gelu_inplace(m, x);

    int c_out_dim = is_cwhn(m) ? 0 : 3;
    int c_out = int(m.weights("conv2.c.weight")->ne[c_out_dim]);
    int stride = (c_out == 320 || c_out == 448 || c_out == 576) ? 1 : 2;
    x = conv_2d_depthwise_batch_norm(m["conv2"], x, stride, 1);
    x = ggml_gelu_inplace(m, x);

    auto [w, h, c, b] = nelements_whcn(m, x);
    x = conv_2d_batch_norm(m["conv3"], x);
    if (is_whcn(m)) {
        x = ggml_reshape_3d(m, x, w * h, c, b);
        x = ggml_cont(m, ggml_permute(m, x, 1, 0, 2, 3));
    } else {
        x = ggml_reshape_3d(m, x, c, w * h, b);
    } // -> always [c, wh, b]
    return named(m, x);
}

tensor mlp(model_ref m, tensor x) {
    x = layer_norm(m["norm"], x);

    x = linear(m["fc1"], x);
    x = ggml_gelu_inplace(m, x);
    x = linear(m["fc2"], x);
    return named(m, x);
}

tensor attention_rel_bias(model_ref m, tensor x, int dim, int num_heads) {
    GGML_ASSERT(dim % num_heads == 0);
    int key_dim = dim / num_heads;
    auto [c, n, b, _] = nelements(x);

    x = layer_norm(m["norm"], x);

    tensor qkv = linear(m["qkv"], x);
    qkv = ggml_reshape_4d(m, qkv, key_dim, 3, num_heads * n, b);
    qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 3, 1, 2)); // ne = [key_dim, num_heads * n, b, 3]

    auto split = [=](model_ref m, tensor tensor, int64_t index) {
        tensor = slice(m, tensor, {}, {}, {}, index);
        tensor = ggml_reshape_4d(m, tensor, key_dim, num_heads, n, b);
        return tensor;
    };

    tensor q = split(m, qkv, 0);
    tensor k = split(m, qkv, 1);
    tensor v = split(m, qkv, 2);
    tensor mask = m.weights("attention_biases_indexed");
    float scale = 1.0f / std::sqrt(float(key_dim));

    if (m.flags & model_build_flag::flash_attention) {
        q = ggml_cont(m, ggml_permute(m, q, 0, 2, 1, 3));
        k = ggml_cast(m, ggml_permute(m, k, 0, 2, 1, 3), GGML_TYPE_F16);
        v = ggml_cast(m, ggml_permute(m, v, 0, 2, 1, 3), GGML_TYPE_F16);
        if (mask->type != GGML_TYPE_F16) {
            mask = ggml_cast(m, mask, GGML_TYPE_F16);
        }

        x = ggml_flash_attn_ext(m, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(x, GGML_PREC_F32);
    } else {
        q = ggml_cont(m, ggml_permute(m, q, 0, 2, 1, 3));
        k = ggml_cont(m, ggml_permute(m, k, 0, 2, 1, 3));
        v = ggml_cont(m, ggml_permute(m, v, 1, 2, 0, 3)); // transpose for mul_mat later

        tensor attn = ggml_mul_mat(m, k, q); // q @ k (k is transposed in mul_mat)
        attn = ggml_soft_max_ext(m, attn, mask, scale, 0.0f);

        x = ggml_mul_mat(m, v, attn);                     // attn @ v
        x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3)); // transpose(1, 2)
    }
    x = ggml_reshape_3d(m, x, key_dim * num_heads, n, b);
    x = linear(m["proj"], x);

    return named(m, x);
}

tensor tiny_vit_block(
    model_ref m, tensor x, int input_resolution, int dim, int num_heads, int window_size) {
    int h = input_resolution;
    int w = input_resolution;
    auto [c, spatial, b, _] = nelements(x);
    GGML_ASSERT(spatial == h * w);
    GGML_ASSERT(h != window_size && w != window_size);

    tensor res_x = x;
    x = ggml_reshape_4d(m, x, c, w, h, b);

    x = window_partition(m, x, window_size);
    x = attention_rel_bias(m["attn"], x, dim, num_heads);
    x = window_reverse(m, x, w, h, window_size);

    x = ggml_reshape_3d(m, x, c, spatial, b);
    x = ggml_add_inplace(m, x, res_x);

    model_ref local_conv = m["local_conv"];
    local_conv.flags |= model_build_flag::cwhn;
    x = ggml_reshape_4d(m, x, c, w, h, b);
    x = conv_2d_depthwise_batch_norm(local_conv, x, 1, 1);
    x = ggml_reshape_3d(m, x, c, spatial, b);

    tensor x_mlp = mlp(m["mlp"], x);
    x = ggml_add_inplace(m, x, x_mlp);
    return named(m, x);
}

tensor conv_layer(model_ref m, tensor x, tiny_vit_params::layer p) {
    auto block = m["blocks"];
    for (int i = 0; i < p.depth; ++i) {
        x = mb_conv(block[i], x);
    }
    x = patch_merging(m["downsample"], x);
    return named(m, x);
}

tensor basic_layer(model_ref m, tensor x, tiny_vit_params::layer const& p) {
    auto blocks = m["blocks"];
    for (int i = 0; i < p.depth; ++i) {
        x = tiny_vit_block(blocks[i], x, p.resolution, p.embed_dim, p.num_heads, p.window_size);
    }
    if (p.downsample) {
        x = ggml_reshape_4d(m, x, x->ne[0], p.resolution, p.resolution, x->ne[2]);
        x = cwhn_to_contiguous_2d(m, x);
        x = patch_merging(m["downsample"], x);
    }
    return named(m, x);
}

tensor tiny_vit(model_ref m, tensor x, tiny_vit_params const& p) {
    x = cwhn_to_contiguous_2d(m, x);
    x = patch_embed(m["patch_embed"], x);
    x = conv_layer(m["layers.0"], x, p.layers[0]);

    auto layers = m["layers"];
    for (int i = 1; i < p.num_layers; ++i) {
        x = basic_layer(layers[i], x, p.layers[i]);
    }

    x = ggml_reshape_4d(m, x, x->ne[0], 64, 64, x->ne[2]);

    // neck
    x = cwhn_to_contiguous_2d(m, x);
    x = conv_2d(m["neck.0"], x);
    x = contiguous_2d_to_cwhn(m, x);
    x = layer_norm(m["neck.1"], x);
    x = cwhn_to_contiguous_2d(m, x);
    x = conv_2d(m["neck.2"], x, 1, 1);
    x = contiguous_2d_to_cwhn(m, x);
    x = layer_norm(m["neck.3"], x);

    return x;
}

//
// Prompt encoder
//

float transform_coord(int p, float scale, int image_size) {
    float center_normalized = (float(p) * scale + 0.5f) / float(image_size);
    return 2.f * center_normalized - 1.f;
}

// Transforms a point from coordinates in the original input image (any resolution)
// to input in [-1, 1] expected by the prompt encoder.
f32x4 preprocess_point(i32x2 point, i32x2 input_image_extent, sam_params const& p) {
    float scale = resize_longest_side(input_image_extent, p.image_size);
    float x = transform_coord(point[0], scale, p.image_size);
    float y = transform_coord(point[1], scale, p.image_size);
    return f32x4{x, y, 0.f, 0.f};
}

f32x4 preprocess_box(
    i32x2 top_left, i32x2 bot_right, i32x2 input_image_extent, sam_params const& p) {
    float scale = resize_longest_side(input_image_extent, p.image_size);
    float x0 = transform_coord(top_left[0], scale, p.image_size);
    float y0 = transform_coord(top_left[1], scale, p.image_size);
    float x1 = transform_coord(bot_right[0], scale, p.image_size);
    float y1 = transform_coord(bot_right[1], scale, p.image_size);
    return f32x4{x0, y0, x1, y1};
}

tensor position_embedding_random(model_ref m, tensor coords) {
    constexpr float pi = 3.14159265358979323846f;

    tensor pe = m.weights("positional_encoding_gaussian_matrix");
    pe = ggml_cont(m, ggml_transpose(m, pe));
    coords = ggml_mul_mat(m, pe, coords);
    coords = ggml_scale_inplace(m, coords, 2.f * pi);
    tensor coords_sin = ggml_sin(m, coords);
    tensor coords_cos = ggml_cos(m, coords);
    return ggml_concat(m, coords_sin, coords_cos, 0);
}

tensor embed_points(model_ref m, tensor coords) {
    int64_t count = coords->ne[1] - 1; // last element is sentinel
    tensor x = position_embedding_random(m["pe_layer"], coords);

    // Write "not_a_point_embed" value into the last coordinate
    tensor label_end = slice(m, x, {}, count);
    label_end = ggml_cpy(m, m.weights("not_a_point_embed.weight"), label_end);
    ggml_build_forward_expand(m.graph, label_end);

    // Add point_embeddings[1] weight to all foreground points (prior coordinates)
    tensor label_one = slice(m, x, {}, {0, count});
    label_one = ggml_add_inplace(m, label_one, m.weights("point_embeddings.1.weight"));
    ggml_build_forward_expand(m.graph, label_one);

    // NOTE: background points are not handled
    return x;
}

tensor embed_box(model_ref m, tensor coords) {
    // Handles a box defined by two points
    coords = ggml_reshape_3d(m, coords, 2, 2, 1);
    tensor x = position_embedding_random(m["pe_layer"], coords);

    // Add point_embeddings[2] to the first corner and point_embeddings[3] to the second corner
    tensor corner1 = slice(m, x, {}, 0);
    corner1 = ggml_add_inplace(m, corner1, m.weights("point_embeddings.2.weight"));
    ggml_build_forward_expand(m.graph, corner1);

    tensor corner2 = slice(m, x, {}, 1);
    corner2 = ggml_add_inplace(m, corner2, m.weights("point_embeddings.3.weight"));
    ggml_build_forward_expand(m.graph, corner2);

    ggml_set_name(x, "box_embed");
    return x;
}

tensor no_mask_embed(model_ref m) {
    return m.weights("no_mask_embed.weight");
}

//
// Mask Decoder
//

tensor mlp_block(model_ref m, tensor x) {
    x = linear(m["lin1"], x);
    x = ggml_relu_inplace(m, x);
    x = linear(m["lin2"], x);
    return x;
}

tensor separate_attention_heads(model_ref m, tensor x, int num_heads) {
    x = ggml_reshape_4d(m, x, x->ne[0] / num_heads, num_heads, x->ne[1], x->ne[2]);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    return x;
}

tensor attention(model_ref m, tensor q, tensor k, tensor v, int num_heads) {
    q = linear(m["q_proj"], q);
    k = linear(m["k_proj"], k);
    v = linear(m["v_proj"], v);

    q = separate_attention_heads(m, q, num_heads);
    k = separate_attention_heads(m, k, num_heads);
    v = ggml_reshape_4d(m, v, v->ne[0] / num_heads, num_heads, v->ne[1], v->ne[2]);
    v = ggml_cont(m, ggml_permute(m, v, 1, 2, 0, 3)); // already transposed for mul_mat

    tensor attn = ggml_mul_mat(m, k, q);
    attn = ggml_scale_inplace(m, attn, 1.0f / std::sqrt(float(q->ne[0])));
    attn = ggml_soft_max(m, attn);

    tensor out = ggml_mul_mat(m, v, attn);
    out = ggml_cont(m, ggml_permute(m, out, 0, 2, 1, 3));
    out = ggml_reshape_3d(m, out, out->ne[0] * out->ne[1], out->ne[2], out->ne[3]);
    out = linear(m["out_proj"], out);
    return out;
}

auto two_way_attention_block(
    model_ref m,
    tensor queries,
    tensor keys,
    tensor query_pe,
    tensor key_pe,
    int num_heads,
    bool skip_first_layer_pe) -> std::tuple<tensor, tensor> {
    // Self attention block
    if (skip_first_layer_pe) {
        queries = attention(m["self_attn"], queries, queries, queries, num_heads);
    } else {
        tensor q = ggml_add(m, queries, query_pe);
        tensor attn_out = attention(m["self_attn"], q, q, queries, num_heads);
        queries = ggml_add(m, queries, attn_out);
    }
    queries = layer_norm(m["norm1"], queries);

    // Cross attention block, tokens attending to image embedding
    tensor q = ggml_add(m, queries, query_pe);
    tensor k = ggml_add(m, keys, key_pe);
    tensor attn_out = attention(m["cross_attn_t2i"], q, k, keys, num_heads);
    queries = ggml_add_inplace(m, queries, attn_out);
    queries = layer_norm(m["norm2"], queries);

    // MLP block
    tensor mlp_out = mlp_block(m["mlp"], queries);
    queries = ggml_add_inplace(m, queries, mlp_out);
    queries = layer_norm(m["norm3"], queries);

    // ???: without this queries is overwritten by keys = keys + attn_out
    queries = ggml_cont(m, queries);

    // Cross attention block, image embedding attending to tokens
    q = ggml_add(m, queries, query_pe);
    // k = ggml_add(m, keys, key_pe); // redundant, same as above
    attn_out = attention(m["cross_attn_i2t"], k, q, queries, num_heads);
    keys = ggml_add_inplace(m, keys, attn_out);
    keys = layer_norm(m["norm4"], keys);

    return {queries, keys};
}

auto two_way_transformer(
    model_ref m,
    tensor image_embedding,
    tensor image_pe,
    tensor point_embedding,
    int depth,
    int num_heads) -> std::tuple<tensor, tensor> {

    auto [c, w, h, b] = nelements(image_embedding);
    image_embedding = ggml_reshape_3d(m, image_embedding, c, w * h, b);
    image_pe = ggml_reshape_3d(m, image_pe, c, w * h, b);

    tensor queries = point_embedding;
    tensor keys = image_embedding;

    // Apply transformer blocks and final layer norm
    model_ref layers = m["layers"];
    for (int i = 0; i < depth; ++i) {
        bool skip_first_layer_pe = i == 0;
        std::tie(queries, keys) = two_way_attention_block(
            layers[i], queries, keys, point_embedding, image_pe, num_heads, skip_first_layer_pe);
    }

    // Apply the final attention layer from the points to the image
    tensor q = ggml_add(m, queries, point_embedding);
    tensor k = ggml_add(m, keys, image_pe);
    tensor attn_out = attention(m["final_attn_t2i"], q, k, keys, num_heads);
    queries = ggml_add_inplace(m, queries, attn_out);
    queries = layer_norm(m["norm_final_attn"], queries);

    return {queries, keys};
}

tensor upscale_outputs(model_ref m, tensor x) {
    m.flags |= model_build_flag::cwhn;
    x = conv_transpose_2d(m[0], x, 2);
    x = layer_norm(m[1], x);
    x = ggml_gelu_inplace(m, x);
    x = conv_transpose_2d(m[3], x, 2);
    x = ggml_gelu_inplace(m, x);
    return x;
}

tensor hypernetwork_mlp(model_ref m, tensor x, int num_layers) {
    model_ref layers = m["layers"];
    for (int i = 0; i < num_layers; ++i) {
        x = linear(layers[i], x);
        if (i < num_layers - 1) {
            x = ggml_relu_inplace(m, x);
        }
    }
    return x;
}

sam_prediction predict_masks(
    model_ref m, tensor image_embeddings, tensor sparse_prompt, tensor dense_prompt) {
    const int num_heads = 8;
    const int transformer_depth = 2;
    const int num_mask_tokens = 4; // num_multimask_outputs + 1

    tensor iou_token = m.weights("iou_token.weight");
    tensor mask_tokens = m.weights("mask_tokens.weight");

    // Concatenate output tokens
    int64_t prompt_size = sparse_prompt->ne[2];
    tensor output_tokens = ggml_concat(m, iou_token, mask_tokens, 1);
    output_tokens = ggml_repeat(
        m, output_tokens,
        ggml_new_tensor_3d(
            m, GGML_TYPE_F32, output_tokens->ne[0], output_tokens->ne[1], prompt_size));
    tensor tokens = ggml_concat(m, output_tokens, sparse_prompt, 1);

    // Expand per-image data in batch direction to be per-mask
    auto [ie0, ie1, ie2, ie3] = nelements(image_embeddings);
    tensor src = ggml_new_tensor_4d(m, GGML_TYPE_F32, ie0, ie1, ie2, tokens->ne[2]);
    src = ggml_repeat(m, image_embeddings, src);
    src = ggml_add_inplace(m, src, dense_prompt);

    tensor image_pe = m.weights("dense_positional_embedding");
    auto [pe0, pe1, pe2, pe3] = nelements(image_pe);
    tensor pos_src = ggml_new_tensor_4d(m, GGML_TYPE_F32, pe0, pe1, pe2, tokens->ne[3]);
    pos_src = ggml_repeat(m, image_pe, pos_src);

    // Run the transformer
    auto [c_low, w_low, h_low, b] = nelements(src);
    auto [hs, out] = two_way_transformer(
        m["transformer"], src, pos_src, tokens, transformer_depth, num_heads);
    tensor iou_token_out = ggml_view_2d(m, hs, hs->ne[0], hs->ne[2], hs->nb[2], 0);
    tensor mask_tokens_out = slice(m, hs, {}, {1, num_mask_tokens + 1}, {});

    // Upscale mask embeddings and predict masks using the mask tokens
    out = ggml_reshape_4d(m, out, c_low, w_low, h_low, b);
    tensor upscaled_embedding = upscale_outputs(m["output_upscaling"], out);
    auto [c, w, h, b2] = nelements(upscaled_embedding);
    upscaled_embedding = ggml_reshape_3d(m, upscaled_embedding, c, w * h, b);

    model_ref mlps = m["output_hypernetworks_mlps"];
    int64_t transformer_out_dim = mlps.weights("3.layers.2.weight")->ne[1];
    tensor hyper_in = ggml_new_tensor_3d(
        m, GGML_TYPE_F32, transformer_out_dim, num_mask_tokens, mask_tokens_out->ne[2]);
    for (int i = 0; i < num_mask_tokens; ++i) {
        tensor mask_slice = slice(m, mask_tokens_out, {}, i);
        mask_slice = hypernetwork_mlp(mlps[i], mask_slice, 3);
        tensor dest_slice = slice(m, hyper_in, {}, i);
        dest_slice = ggml_cpy(m, mask_slice, dest_slice);
        ggml_build_forward_expand(m.graph, dest_slice);
    }
    tensor masks = ggml_mul_mat(m, upscaled_embedding, hyper_in);
    masks = ggml_reshape_4d(m, masks, w, h, masks->ne[1], b);

    // Generate mask quality predictions
    tensor iou_pred = hypernetwork_mlp(m["iou_prediction_head"], iou_token_out, 3);

    return {masks, iou_pred};
}

template <typename Tsrc, typename Tdst, typename Fconvert>
void interpolate_bilinear(
    Tsrc const* src_pixels,
    i32x2 src_extent,
    int src_stride,
    Tdst* dst_pixels,
    i32x2 dst_extent,
    int dst_stride,
    Fconvert&& convert) {

    float scale_x = float(src_extent[0]) / float(dst_extent[0]);
    float scale_y = float(src_extent[1]) / float(dst_extent[1]);

    for (int y = 0; y < dst_extent[1]; ++y) {
        for (int x = 0; x < dst_extent[0]; ++x) {
            float src_xf = std::max((x + 0.5f) * scale_x - 0.5f, 0.0f);
            float src_yf = std::max((y + 0.5f) * scale_y - 0.5f, 0.0f);
            int src_x0 = std::max(int(src_xf), 0);
            int src_y0 = std::max(int(src_yf), 0);
            int src_x1 = std::min(src_x0 + 1, src_extent[0] - 1);
            int src_y1 = std::min(src_y0 + 1, src_extent[1] - 1);

            float v00 = src_pixels[src_y0 * src_stride + src_x0];
            float v01 = src_pixels[src_y0 * src_stride + src_x1];
            float v10 = src_pixels[src_y1 * src_stride + src_x0];
            float v11 = src_pixels[src_y1 * src_stride + src_x1];

            float wx = src_xf - src_x0;
            float wy = src_yf - src_y0;
            float v0 = (1 - wx) * v00 + wx * v01;
            float v1 = (1 - wx) * v10 + wx * v11;
            float value = (1 - wy) * v0 + wy * v1;

            dst_pixels[y * dst_stride + x] = convert(value);
        }
    }
}

float resize_longest_side(i32x2 extent, int target_longest_side) {
    int longest_side = std::max(extent[0], extent[1]);
    return float(target_longest_side) / float(longest_side);
}

int scale_coord(int coord, float scale) {
    return int(coord * scale + 0.5f);
}

i32x2 scale_extent(i32x2 extent, float scale) {
    return i32x2{scale_coord(extent[0], scale), scale_coord(extent[1], scale)};
}

} // namespace sam

image_data sam_process_input(image_view image, sam_params const& p) {
    constexpr f32x4 mean = f32x4{0.485f, 0.456f, 0.406f, 0.f};
    constexpr f32x4 std = f32x4{0.229f, 0.224f, 0.225f, 1.f};

    std::optional<image_data> resized;
    float scale = sam::resize_longest_side(image.extent, p.image_size);
    if (scale != 1) {
        resized = image_scale(image, sam::scale_extent(image.extent, scale));
        image = image_view(*resized);
    }

    image_data result = image_alloc({p.image_size, p.image_size}, image_format::rgb_f32);
    image_u8_to_f32(image, result, -mean, 1.f / std);
    return result;
}

f32x4 sam_process_point(i32x2 point, i32x2 input_image_extent, sam_params const& p) {
    return sam::preprocess_point(point, input_image_extent, p);
}

f32x4 sam_process_box(box_2d box, i32x2 input_image_extent, sam_params const& p) {
    return sam::preprocess_box(box.top_left, box.bottom_right, input_image_extent, p);
}

image_data sam_process_mask(
    std::span<float const> mask_data, int mask_index, i32x2 target_extent, sam_params const& p) {

    mask_data = mask_data.subspan(mask_index * sqr(p.mask_size));
    ASSERT(int(mask_data.size()) >= sqr(p.mask_size));

    float scale = sam::resize_longest_side(target_extent, p.image_size);
    i32x2 scaled_extent = sam::scale_extent(target_extent, scale);
    image_data mask = image_alloc(target_extent, image_format::alpha_u8);

    auto scaled = std::vector<float>(p.image_size * p.image_size);
    auto id = [](float x) {
        return x;
    };
    sam::interpolate_bilinear(
        mask_data.data(), {p.mask_size, p.mask_size}, p.mask_size, scaled.data(),
        {p.image_size, p.image_size}, p.image_size, id);

    auto threshold = [](float x) {
        return uint8_t(x > 0.0f ? 255 : 0);
    };
    sam::interpolate_bilinear(
        scaled.data(), scaled_extent, p.image_size, mask.data.get(), target_extent,
        target_extent[0], threshold);

    return mask;
}

tensor sam_encode_points(model_ref m, tensor coords) {
    return sam::embed_points(m["prompt_encoder"], coords);
}

tensor sam_encode_box(model_ref m, tensor coords) {
    return sam::embed_box(m["prompt_encoder"], coords);
}

tensor sam_encode_image(model_ref m, tensor image, sam_params const&) {
    return sam::tiny_vit(m["enc"], image, sam::tiny_vit_params{});
}

sam_prediction sam_predict_mask(model_ref m, tensor image_embed, tensor prompt_embed) {
    tensor dense_prompt = sam::no_mask_embed(m["prompt_encoder"]);
    auto [masks, iou] = sam::predict_masks(m["dec"], image_embed, prompt_embed, dense_prompt);

    return {compute_graph_output(m, masks, "masks"), compute_graph_output(m, iou, "iou")};
}

} // namespace visp
