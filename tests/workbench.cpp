#include "util/string.h"
#include "visp/arch/birefnet.h"
#include "visp/arch/depth-anything.h"
#include "visp/arch/dino.h"
#include "visp/arch/esrgan.h"
#include "visp/arch/migan.h"
#include "visp/arch/mobile-sam.h"
#include "visp/nn.h"

#include <ggml-blas.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <ggml.h>

#include <cassert>
#include <cmath>
#include <exception>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace std::literals;

//
// Workbench - environment for comparing Python torch and C++ ggml implementation
//
// 1) Create the NN module to test in Python
// 2) Initialize state-dict with random weights
// 3) Create some input tensors and compute `forward` to get the expected output
// 4) Create an entry function here in C++ which calls the ggml implementation
// 5) In Python, call `workbench.invoke_test("entry_function_name", input_tensor, state_dict)`
// 6) Compare the result with the expected output (eg. `torch.allclose`)

namespace visp {

enum class param_type { int32, float32, string };

struct test_param {
    std::string_view name;
    param_type type;
    union {
        int32_t i;
        float f;
        char const* s;
    } value;
};

struct param_dict {
    std::vector<test_param> params;

    test_param const* find(char const* name) const;
    int get(char const* name, int) const;
    float get(char const* name, float) const;
    char const* get(char const* name, char const*) const;
};

using test_function = std::vector<tensor> (*)(model_ref, std::span<tensor>, param_dict const&);

backend_device const& workbench_backend();
void workbench_add_test(char const* name, test_function func);

struct test_case_def {
    test_case_def(test_function func, char const* name) { workbench_add_test(name, func); }
};

#define DEF(name)                                                                                  \
    std::vector<tensor> _test_func_##name(model_ref, std::span<tensor>, param_dict const&);        \
    const test_case_def _test_def_##name(_test_func_##name, #name);                                \
    std::vector<tensor> _test_func_##name

//
// Test entry points
//

DEF(conv_2d_depthwise_nchw)(model_ref m, span<tensor> input, param_dict const& p) {
    int stride = p.get("stride", 1);
    int pad = p.get("pad", 0);
    int dil = p.get("dilation", 1);
    tensor weight = m.weights("weight");
    return {ggml_conv_2d_dw_direct(m, weight, input[0], stride, stride, pad, pad, dil, dil)};
}

DEF(conv_2d_depthwise_nhwc)(model_ref m, span<tensor> input, param_dict const& p) {
    return {conv_2d_depthwise(m, input[0], p.get("stride", 1), p.get("pad", 0))};
}

DEF(conv_2d_channels)(model_ref m, span<tensor> input, param_dict const& p) {
    return {conv_2d(m, input[0], p.get("stride", 1), p.get("pad", 0))};
}

DEF(conv_transpose_2d)(model_ref m, span<tensor> input, param_dict const& p) {
    return {conv_transpose_2d(m, input[0], p.get("stride", 1))};
}

DEF(conv_2d_deform)(model_ref m, span<tensor> input, param_dict const& p) {
    tensor weight = m.weights("weight");
    tensor offset = m.weights("offset");
    tensor mask = m.find("mask");
    int padding = p.get("padding", 1);
    return {conv_2d_deform(m, input[0], weight, offset, mask, 1, padding)};
}

DEF(batch_norm_2d)(model_ref m, span<tensor> input, param_dict const& p) {
    return {batch_norm_2d(m, input[0])};
}

DEF(roll)(model_ref m, span<tensor> input, param_dict const& p) {
    return {ggml_roll(m, input[0], p.get("s0", 0), p.get("s1", 0), p.get("s2", 0), p.get("s3", 0))};
}

DEF(layer_norm)(model_ref m, span<tensor> input, param_dict const& p) {
    return {layer_norm(m, input[0])};
}

DEF(linear)(model_ref m, span<tensor> input, param_dict const& p) {
    return {linear(m, input[0])};
}

DEF(interpolate)(model_ref m, span<tensor> input, param_dict const& p) {
    int w = p.get("w", 8);
    int h = p.get("h", 8);
    uint32_t mode = p.get("mode", "bilinear") == "bilinear"sv ? GGML_SCALE_MODE_BILINEAR
                                                              : GGML_SCALE_MODE_BICUBIC;
    if (p.get("align_corners", 0)) {
        mode |= GGML_SCALE_FLAG_ALIGN_CORNERS;
    }
    return {ggml_interpolate(m, input[0], w, h, input[0]->ne[2], input[0]->ne[3], mode)};
}

//
// Mobile SAM

DEF(sam_conv_2d_batch_norm)(model_ref m, span<tensor> input, param_dict const& p) {
    return {conv_2d(m["c"], input[0], 2, 1)}; // fused conv_2d + batch_norm
}

DEF(sam_patch_embed)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::patch_embed(m, input[0])};
}

DEF(sam_mb_conv)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::mb_conv(m, input[0])};
}

DEF(sam_patch_merging)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::patch_merging(m, input[0])};
}

DEF(sam_mlp)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::mlp(m, input[0])};
}

DEF(sam_attention_rel_bias)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::attention_rel_bias(m, input[0], 4, 2)};
}

DEF(sam_window_partition)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::window_partition(m, input[0], 3)};
}

DEF(sam_tiny_vit_block)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::tiny_vit_block(
        m, input[0], 8, /*dim*/ 4, /*num_heads*/ 2,
        /*window_size*/ 5)};
}

DEF(sam_tiny_vit)(model_ref m, span<tensor> input, param_dict const& p) {
    sam::tiny_vit_params params;
    return {sam::tiny_vit(m, input[0], params)};
}

DEF(sam_position_embedding_random)(model_ref m, span<tensor> input, param_dict const& p) {
    float* input_data = reinterpret_cast<float*>(input[0]->data);
    for (int i = 0; i < ggml_nelements(input[0]); ++i) {
        input_data[i] = (input_data[i] / 64.f) * 2.f - 1.f;
    }
    return {sam::position_embedding_random(m, input[0])};
}

DEF(sam_embed_points)(model_ref m, span<tensor> input, param_dict const& p) {
    float* input_data = reinterpret_cast<float*>(input[0]->data);
    for (int i = 0; i < ggml_nelements(input[0]) - 2; ++i) {
        input_data[i] = sam::transform_coord(input_data[i], 1.0f, 64);
    }
    return {sam::embed_points(m, input[0])};
}

DEF(sam_embed_box)(model_ref m, span<tensor> input, param_dict const& p) {
    float* input_data = reinterpret_cast<float*>(input[0]->data);
    for (int i = 0; i < ggml_nelements(input[0]); ++i) {
        input_data[i] = sam::transform_coord(input_data[i], 1.0f, 64);
    }
    return {sam::embed_box(m, input[0])};
}

DEF(sam_attention)(model_ref m, span<tensor> input, param_dict const& p) {
    tensor q = input[0];
    tensor k = m.weights("input_k");
    tensor v = m.weights("input_v");
    return {sam::attention(m, q, k, v, 2)};
}

DEF(sam_two_way_attention_block)(model_ref m, span<tensor> input, param_dict const& p) {
    tensor queries = input[0];
    tensor keys = m.weights("input_keys");
    tensor query_pe = m.weights("input_query_pe");
    tensor key_pe = m.weights("input_key_pe");
    bool skip_first_layer_pe = p.get("mode", "default") == "skip_first_layer_pe"sv;
    auto [result_queries, result_keys] = sam::two_way_attention_block(
        m, queries, keys, query_pe, key_pe, 2, skip_first_layer_pe);
    return {result_queries, result_keys};
}

DEF(sam_two_way_transformer)(model_ref m, span<tensor> input, param_dict const& p) {
    tensor image_embedding = input[0];
    tensor image_pe = m.weights("input_image_pe");
    tensor point_embedding = m.weights("input_point_embedding");
    auto [result_queries, result_keys] = sam::two_way_transformer(
        m, image_embedding, image_pe, point_embedding, 2, 2);
    return {result_queries, result_keys};
}

DEF(sam_hypernetwork_mlp)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::hypernetwork_mlp(m, input[0], 2)};
}

DEF(sam_output_upscaling)(model_ref m, span<tensor> input, param_dict const& p) {
    return {sam::upscale_outputs(m, input[0])};
}

DEF(sam_predict_masks)(model_ref m, span<tensor> input, param_dict const& p) {
    tensor image_embeddings = input[0];
    tensor sparse_prompt = m.weights("input_sparse_prompt");
    tensor dense_prompt = m.weights("input_dense_prompt");
    auto [masks, iou] = sam::predict_masks(m, image_embeddings, sparse_prompt, dense_prompt);
    return {masks, iou};
}

//
// BiRefNet

DEF(biref_patch_embed)(model_ref m, span<tensor> input, param_dict const& p) {
    return {patch_embed(m, input[0], 4)};
}

DEF(biref_relative_position_index)(model_ref m, span<tensor> input, param_dict const& p) {
    auto dst = span(reinterpret_cast<int32_t*>(input[0]->data), ggml_nelements(input[0]));
    birefnet::compute_relative_position_index(dst, 3);
    return {input[0]};
}

DEF(biref_window_attention)(model_ref m, span<tensor> input, param_dict const& p) {
    int window_size = 3;
    tensor mask = m.find("mask");
    auto rel_pos_index = birefnet::create_relative_position_index(m, window_size);
    ggml_backend_alloc_ctx_tensors(m, workbench_backend());
    transfer_to_backend(rel_pos_index);
    return {birefnet::window_attention(m, input[0], mask, 2, window_size)};
}

DEF(biref_swin_block)(model_ref m, span<tensor> input, param_dict const& p) {
    birefnet::swin_block_params block;
    block.n_heads = 2;
    block.window_size = 3;
    block.w = 6;
    block.h = 6;
    block.shift = 0;
    tensor mask = m.find("mask");
    auto rel_pos_index = birefnet::create_relative_position_index(m, 3);
    ggml_backend_alloc_ctx_tensors(m, workbench_backend());
    transfer_to_backend(rel_pos_index);
    return {birefnet::swin_block(m, input[0], mask, block)};
}

DEF(biref_patch_merging)(model_ref m, span<tensor> input, param_dict const& p) {
    return {birefnet::patch_merging(m, input[0], 6, 4)};
}

DEF(biref_attention_mask)(model_ref m, span<tensor> input, param_dict const& p) {
    auto dst = span((float*)input[0]->data, ggml_nelements(input[0]));
    birefnet::compute_attention_mask(dst, 18, 18, 6);
    return {input[0]};
}

DEF(biref_swin_layer)(model_ref m, span<tensor> input, param_dict const& p) {
    swin_layer_t layer;
    layer.depth = 2;
    layer.n_heads = 2;
    layer.n_features = 8;
    layer.downsample = true;
    auto rel_pos_index = birefnet::create_relative_position_index(m, 3);
    auto attn_mask = birefnet::create_attention_mask(m, 6, 6, 3);
    ggml_backend_alloc_ctx_tensors(m, workbench_backend());
    transfer_to_backend(rel_pos_index);
    transfer_to_backend(attn_mask);
    auto result = birefnet::swin_layer(m, input[0], 6, 6, layer, 3);
    ASSERT(result.w_down == 3 && result.h_down == 3);
    return {result.x_down};
}

DEF(biref_swin_transformer)(model_ref m, span<tensor> input, param_dict const& p) {
    swin_params swinp = {
        .embed_dim = 8,
        .window_size = 3,
        .layers = {
            swin_layer_t{2, 2, 8 * 1, true},
            swin_layer_t{2, 2, 8 * 2, true},
            swin_layer_t{2, 4, 8 * 4, true},
            swin_layer_t{2, 2, 8 * 8, false},
        }};
    auto rel_pos_index = birefnet::create_relative_position_index(m, 3);
    auto attn_masks = std::array{
        birefnet::create_attention_mask(m, 8, 8, 3), birefnet::create_attention_mask(m, 4, 4, 3),
        birefnet::create_attention_mask(m, 2, 2, 3), birefnet::create_attention_mask(m, 1, 1, 3)};
    ggml_backend_alloc_ctx_tensors(m, workbench_backend());
    transfer_to_backend(rel_pos_index);
    for (auto&& attn_mask : attn_masks) {
        transfer_to_backend(attn_mask);
    }
    auto result = birefnet::swin_transformer(m, input[0], swinp);
    return {result[0], result[1], result[2], result[3]};
}

DEF(biref_encode)(model_ref m, span<tensor> input, param_dict const& p) {
    birefnet::swin_result xs, xs_low;
    for (int i = 0; i < 4; ++i) {
        xs[i] = m.find(format<tensor_name>("input{}", i).c_str());
        xs_low[i] = m.find(format<tensor_name>("input_low{}", i).c_str());
    }
    birefnet::encode_concat(m, xs, xs_low);
    return std::vector{xs[0], xs[1], xs[2], xs[3]};
}

DEF(biref_deformable_conv_2d)(model_ref m, span<tensor> input, param_dict const& p) {
    return {birefnet::deformable_conv_2d(m, input[0], 1, 1)};
}

DEF(biref_global_avg_pool)(model_ref m, span<tensor> input, param_dict const& p) {
    return {birefnet::global_avg_pool(m, input[0])};
}

DEF(biref_aspp_deformable)(model_ref m, span<tensor> input, param_dict const& p) {
    return {birefnet::aspp_deformable(m, input[0])};
}

DEF(biref_basic_dec_blk)(model_ref m, span<tensor> input, param_dict const& p) {
    return {birefnet::basic_decoder_block(m, input[0])};
}

DEF(biref_image_to_patches_2)(model_ref m, span<tensor> input, param_dict const& p) {
    return {birefnet::image_to_patches(m, input[0], 4, 4)};
}

DEF(biref_decode)(model_ref m, span<tensor> input, param_dict const& p) {
    birefnet::swin_result features;
    for (int i = 0; i < 4; ++i) {
        features[i] = m.find(format<tensor_name>("x{}", i + 1).c_str());
    }
    return {birefnet::decode(m, input[0], features)};
}

//
// MI-GAN

DEF(migan_lrelu_agc)(model_ref m, span<tensor> input, param_dict const& p) {
    return {migan::lrelu_agc(m, input[0], 0.2f, std::sqrt(2), 1.0f)};
}

DEF(migan_downsample_2d)(model_ref m, span<tensor> input, param_dict const& p) {
    return {migan::downsample_2d(m, input[0])};
}

DEF(migan_upsample_2d)(model_ref m, span<tensor> input, param_dict const& p) {
    return {ggml_cont(m, migan::upsample_2d(m, input[0]))};
}

DEF(migan_separable_conv_2d)(model_ref m, span<tensor> input, param_dict const& p) {
    auto flags = migan::conv::noise | migan::conv::activation;
    return {migan::separable_conv_2d(m, input[0], flags)};
}

DEF(migan_encoder)(model_ref m, span<tensor> input, param_dict const& p) {
    return {migan::encode(m, input[0], 16).first};
}

DEF(migan_synthesis)(model_ref m, span<tensor> input, param_dict const& p) {
    migan::features feats;
    feats[0] = m.weights("feat16");
    feats[1] = m.weights("feat8");
    feats[2] = m.weights("feat4");
    return {migan::synthesis(m, input[0], feats, 16)};
}

//
// ESRGAN

DEF(esrgan_upconv)(model_ref m, span<tensor> input, param_dict const& p) {
    return {esrgan::upsample(m[1], input[0])};
}

DEF(esrgan_residual_dense_block)(model_ref m, span<tensor> input, param_dict const& p) {
    return {esrgan::risidual_dense_block(m, input[0])};
}

DEF(esrgan_rrdb)(model_ref m, span<tensor> input, param_dict const& p) {
    return {esrgan::rrdb(m, input[0])};
}

DEF(esrgan_rrdbnet)(model_ref m, span<tensor> input, param_dict const& p) {
    esrgan_params params;
    params.n_blocks = 2;
    params.scale = 2;
    return {esrgan_generate(m, input[0], params)};
}

//
// DINO

DEF(dino_interpolate_pos_encoding)(model_ref m, span<tensor> input, param_dict const& p) {
    int s = p.get("img_size", 64);
    int patch_size = p.get("patch_size", 16);
    return {dino::interpolate_pos_encoding(m, input[0], s, s, patch_size)};
}

DEF(dino_prepare_tokens)(model_ref m, span<tensor> input, param_dict const& p) {
    return {dino::prepare_tokens(m, input[0], 4)};
}

DEF(dino_attention)(model_ref m, span<tensor> input, param_dict const& p) {
    return {dino::attention(m, input[0], p.get("n_heads", 8), p.get("flash_attn", 0) != 0)};
}

DEF(dino_block)(model_ref m, span<tensor> input, param_dict const& p) {
    dino::dino_params params{};
    params.n_heads = p.get("n_heads", 8);
    params.flash_attention = p.get("flash_attn", 0) != 0;
    return {dino::block(m, input[0], params)};
}

//
// Depth Anything

//
// Workbench implementation
//

struct raw_param {
    char const* name;
    char const* value;
    int32_t type;
};

param_dict build_dict(span<raw_param const> raw_params) {
    param_dict dict;
    for (const auto& raw : raw_params) {
        test_param param;
        param.name = raw.name;

        switch (param_type(raw.type)) {
            case param_type::int32:
                param.type = param_type::int32;
                param.value.i = std::stoi(raw.value);
                break;
            case param_type::float32:
                param.type = param_type::float32;
                param.value.f = std::stof(raw.value);
                break;
            case param_type::string:
                param.type = param_type::string;
                param.value.s = raw.value;
                break;
            default: throw except("Unknown parameter type");
        }
        dict.params.push_back(param);
    }
    return dict;
}

test_param const* param_dict::find(char const* name) const {
    auto it = std::find_if(
        params.begin(), params.end(), [name](test_param const& p) { return p.name == name; });
    return it != params.end() ? &(*it) : nullptr;
}

int param_dict::get(char const* name, int default_value) const {
    if (auto param = find(name)) {
        ASSERT(param->type == param_type::int32);
        return param->value.i;
    }
    return default_value;
}

float param_dict::get(char const* name, float default_value) const {
    if (auto param = find(name)) {
        ASSERT(param->type == param_type::float32);
        return param->value.f;
    }
    return default_value;
}

char const* param_dict::get(char const* name, char const* default_value) const {
    if (auto param = find(name)) {
        ASSERT(param->type == param_type::string);
        return param->value.s;
    }
    return default_value;
}

struct raw_tensor {
    char const* name;
    float* data;
    int32_t type_;
    int32_t ne[4];

    ggml_type type() const { return ggml_type(type_); }
    size_t size() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    size_t size_bytes() const { return size() * ggml_type_size(type()); }
};

struct test_case {
    char const* name;
    test_function func;
};

struct workbench {
    std::vector<test_case> tests;
    std::vector<raw_tensor> outputs;
    std::vector<byte> data; // for storing output tensor data
    backend_device current_backend;
};

workbench& get_workbench() {
    static workbench w;
    return w;
}

backend_device const& workbench_backend() {
    return get_workbench().current_backend;
}

void workbench_add_test(char const* name, test_function func) {
    auto& w = get_workbench();
    w.tests.push_back(test_case{name, func});
}

test_case const& workbench_find_test(std::string_view name) {
    auto& w = get_workbench();
    auto it = std::find_if(
        w.tests.begin(), w.tests.end(), [name](test_case const& t) { return t.name == name; });
    if (it != w.tests.end()) {
        return *it;
    }
    throw except("Test case not found: {}", name);
}

void workbench_run(
    std::string_view test_name,
    span<raw_tensor const> tensors,
    span<raw_param const> params,
    backend_type backend_type) {

    workbench& w = get_workbench();
    w.current_backend = backend_init(backend_type);
    model_weights weights = model_init(tensors.size() + 10);
    weights.buffer_type = backend_type;
    compute_graph graph = compute_graph_init(1024);
    model_ref m(weights, graph);

    std::vector<tensor> inputs;
    for (raw_tensor const& raw : tensors) {
        auto tensor = ggml_new_tensor_4d(
            m.weights_context, raw.type(), raw.ne[0], raw.ne[1], raw.ne[2], raw.ne[3]);
        if (raw.name && raw.name[0] != '\0' && raw.name != std::string_view("input")) {
            ggml_set_name(tensor, raw.name);
        } else {
            inputs.push_back(tensor);
        }
    }

    model_allocate(weights, w.current_backend);
    for (raw_tensor const& raw : tensors) {
        transfer_to_backend(m.weights(raw.name), span(raw.data, raw.size()));
    }

    param_dict test_params = build_dict(params);
    std::string_view memory_layout = test_params.get("memory_layout", "whcn");
    if (memory_layout == "cwhn" || memory_layout == "nhwc") {
        m.flags |= model_build_flag::cwhn;
    }

    test_case const& test = workbench_find_test(test_name);
    std::vector<tensor> outputs = test.func(m, inputs, test_params);
    for (tensor& out : outputs) {
        out = compute_graph_output(m, ggml_cont(m, out));
    }

    ASSERT(!outputs.empty(), "Test function must return at least one output tensor");

    compute_graph_allocate(graph, w.current_backend);
    compute(graph, w.current_backend);

    size_t output_size = std::accumulate(
        outputs.begin(), outputs.end(), size_t(0),
        [](size_t sum, tensor t) { return sum + ggml_nbytes(t); });
    w.data.resize(output_size);
    size_t offset = 0;

    std::vector<raw_tensor>& output_raw = w.outputs;
    output_raw.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        byte* data_ptr = w.data.data() + offset;
        offset += ggml_nbytes(outputs[i]);
        ggml_backend_tensor_get(outputs[i], data_ptr, 0, ggml_nbytes(outputs[i]));

        output_raw[i].name = ggml_get_name(outputs[i]);
        output_raw[i].data = reinterpret_cast<float*>(data_ptr);
        output_raw[i].type_ = int32_t(outputs[i]->type);
        output_raw[i].ne[0] = outputs[i]->ne[0];
        output_raw[i].ne[1] = outputs[i]->ne[1];
        output_raw[i].ne[2] = outputs[i]->ne[2];
        output_raw[i].ne[3] = outputs[i]->ne[3];
    }
}

} // namespace visp

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
__declspec(dllexport)
#endif
int32_t
visp_workbench(
    char const* testcase,
    visp::raw_tensor const* inputs,
    int32_t n_inputs,
    visp::raw_param const* params,
    int32_t n_params,
    visp::raw_tensor const** outputs,
    int32_t* n_outputs,
    int32_t backend) {

    try {
        visp::workbench_run(
            testcase, std::span(inputs, n_inputs), std::span(params, n_params),
            visp::backend_type(backend));

        *outputs = visp::get_workbench().outputs.data();
        *n_outputs = int32_t(visp::get_workbench().outputs.size());

    } catch (std::exception const& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }
    return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif