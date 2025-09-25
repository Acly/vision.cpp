#include "visp/ml.h"
#include "util/string.h"
#include "visp/platform.h"

#include <algorithm>
#include <array>
#include <thread>
#include <vector>

namespace visp {

//
// backend

enum ggml_backend_dev_type convert(backend_type type) {
    switch (type) {
        case backend_type::cpu: return GGML_BACKEND_DEVICE_TYPE_CPU;
        case backend_type::gpu: return GGML_BACKEND_DEVICE_TYPE_GPU;
        default:
            ASSERT(false, "Unsupported backend type");
            return GGML_BACKEND_DEVICE_TYPE_CPU; // fallback
    }
}

bool load_ggml_backends() {
    static const bool loaded = []() {
        if (ggml_backend_reg_count() > 0) {
            return true; // already loaded
        }
        ggml_backend_load_all();
        if (ggml_backend_reg_count() == 0) {
            if (path dir = current_library_path(); !dir.empty()) {
                auto str = dir.parent_path().u8string();
                ggml_backend_load_all_from_path((char const*)str.c_str());
            }
        }
        return true;
    }();
    return loaded;
}

bool backend_is_available(backend_type type) {
    load_ggml_backends();
    return ggml_backend_dev_by_type(convert(type)) != nullptr;
}

backend_device backend_init() {
    load_ggml_backends();
    backend_device b;
    b.handle.reset(ggml_backend_init_best());
    b.device = ggml_backend_get_device(b.handle.get());
    ASSERT(b.handle, "Failed to initialize backend");
    return b;
}

backend_device backend_init(backend_type type) {
    load_ggml_backends();

    backend_device b;
    b.handle.reset(ggml_backend_init_by_type(convert(type), nullptr));
    if (!b.handle) {
        throw except("Failed to initialize backend, no suitable device available");
    }
    b.device = ggml_backend_get_device(b.handle.get());

    int nthreads = std::max(1, (int)std::thread::hardware_concurrency() - 2);
    backend_set_n_threads(b, nthreads);
    return b;
}

backend_type backend_device::type() const {
    ggml_backend_dev_t dev = ggml_backend_get_device(handle.get());
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return backend_type::cpu;
        case GGML_BACKEND_DEVICE_TYPE_GPU: return backend_type::gpu;
        default: ASSERT(false, "Unsupported backend device type"); return backend_type::cpu;
    }
}

typedef bool (*ggml_backend_dev_supports_f16_t)(ggml_backend_dev_t);

ggml_type backend_device::preferred_float_type() const {
    if (type() == backend_type::cpu) {
        return GGML_TYPE_F32; // not all operations support F16
    } else {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (void* f = ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_supports_f16")) {
            bool supports_f16 = ((ggml_backend_dev_supports_f16_t)f)(device);
            if (!supports_f16) {
                return GGML_TYPE_F32;
            }
        }
    }
    return GGML_TYPE_COUNT; // no preference, use float type of model weights
}

tensor_data_layout backend_device::preferred_layout() const {
    if (type() == backend_type::cpu) {
        return tensor_data_layout::cwhn;
    }
    return tensor_data_layout::unknown; // no preference, keep model weight layout
}

size_t backend_device::total_memory() const {
    ggml_backend_dev_t dev = ggml_backend_get_device(handle.get());
    size_t free, total;
    ggml_backend_dev_memory(dev, &free, &total);
    return total;
}

size_t backend_device::max_alloc() const {
    const size_t vulkan_max = 4 * 1024 * 1024 * 1024ULL; // TODO: query from backend
    return type() == backend_type::cpu ? SIZE_MAX : vulkan_max;
}

void backend_set_n_threads(backend_device& b, int n_threads) {
    if (b.type() != backend_type::cpu) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(b.device);
    ggml_backend_set_n_threads_t set_n_threads =
        (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_set_n_threads");
    ASSERT(set_n_threads, "Failed to get backend set_n_threads function");
    set_n_threads(b.handle.get(), n_threads);
}

//
// model_build_flags

model_build_flags backend_default_flags(backend_type type) {
    using enum model_build_flag;
    switch (type) {
        case backend_type::cpu:
            return conv_2d_direct_cwhn | concat_n | f16_conv_transpose | window_partition;
        case backend_type::gpu: return {};
    }
    return {};
}

model_build_flags model_get_build_flags(model_file const& file) {
    fixed_string<64> str;
    std::string_view arch = file.arch();
    model_build_flags flags;

    int64_t key = gguf_find_key(file.gguf.get(), format(str, "{}.tensor_data_layout", arch));
    if (key != -1) {
        std::string_view layout = gguf_get_val_str(file.gguf.get(), key);
        if (layout == "cwhn") {
            flags |= model_build_flag::cwhn;
        }
    }
    return flags;
}

//
// model_file

model_file model_load(char const* filepath) {
    ggml_context* data_ctx;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &data_ctx;

    gguf_context_ptr gguf_ctx(gguf_init_from_file(filepath, params));
    if (!gguf_ctx) {
        throw except("Failed to load GGUF model: {}", filepath);
    }
    return model_file{std::move(gguf_ctx), ggml_context_ptr(data_ctx), filepath};
}

int64_t model_file::n_tensors() const {
    return gguf_get_n_tensors(gguf.get());
}

int64_t model_file::key(char const* name) const {
    int64_t key_id = gguf_find_key(gguf.get(), name);
    if (key_id == -1) {
        throw except("Can't find key '{}' in model file {}", name, path);
    }
    return key_id;
}

std::string_view model_file::get_string(char const* key_name) const {
    return gguf_get_val_str(gguf.get(), key(key_name));
}

int model_file::get_int(char const* key_name) const {
    return gguf_get_val_i32(gguf.get(), key(key_name));
}

std::string_view model_file::arch() const {
    return get_string("general.architecture");
}

tensor_data_layout model_file::tensor_layout() const {
    fixed_string<64> str;
    int64_t key = gguf_find_key(gguf.get(), format(str, "{}.tensor_data_layout", arch()));
    if (key != -1) {
        std::string_view layout = gguf_get_val_str(gguf.get(), key);
        if (layout == "cwhn") {
            return tensor_data_layout::cwhn;
        } else if (layout == "whcn") {
            return tensor_data_layout::whcn;
        }
    }
    return tensor_data_layout::unknown;
}

//
// model_weights

model_weights model_init(size_t size) {
    ggml_init_params params{};
    params.mem_size = size * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));

    model_weights w{};
    w.context = std::move(ctx);
    w.buffer_type = backend_type::cpu;
    return w;
}

bool model_allocate(model_weights& m, backend_device const& b) {
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(m.context.get(), b.handle.get()));
    if (!buffer) {
        return false; // context contains nothing to allocate
    }
    m.buffer_type = b.type();
    m.extra_buffers.push_back(std::move(buffer));
    return true;
}

namespace {

bool is_float_type(ggml_type t) {
    return t != GGML_TYPE_I8 && t != GGML_TYPE_I16 && t != GGML_TYPE_I32 && t != GGML_TYPE_I64;
}

int64_t max_tensor_elements(ggml_context* ctx) {
    int64_t result = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        result = std::max(result, ggml_nelements(t));
    }
    return result;
}

ggml_type detect_float_type(ggml_context* ctx) {
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (is_float_type(t->type)) {
            return t->type;
        }
    }
    return GGML_TYPE_F32;
}

template <typename T>
void permute_whcn_to_cwhn(T* n, bool depthwise) {
    if (depthwise) { // wh1c -> c1wh
        T perm[] = {n[3], n[2], n[0], n[1]};
        std::copy(perm, perm + 4, n);
    } else {
        std::swap(n[0], n[2]); // -> chwn
        std::swap(n[1], n[2]); // -> cwhn
    }
}

struct tensor_converter {
    ggml_type src_type;
    ggml_type dst_type;
    ggml_backend_ptr backend;
    ggml_context_ptr ctx;
    ggml_cgraph* graph;
    ggml_gallocr_ptr gallocr;
    ggml_tensor convert_src{};
    ggml_tensor* convert_dst;

    tensor_converter(ggml_context* weights, ggml_type target_type, bool whcn_to_cwhn)
        : dst_type(target_type) {

        if (dst_type == GGML_TYPE_COUNT && !whcn_to_cwhn) {
            return;
        }
        src_type = detect_float_type(weights);
        if (src_type == dst_type && !whcn_to_cwhn) {
            return;
        }
        if (dst_type == GGML_TYPE_COUNT) {
            dst_type = src_type;
        }

        ggml_init_params ctx_params{
            .mem_size = ggml_tensor_overhead() + ggml_graph_overhead(),
            .mem_buffer = nullptr,
            .no_alloc = true};
        ctx.reset(ggml_init(ctx_params));

        size_t max_elem = max_tensor_elements(weights);
        graph = ggml_new_graph_custom(ctx.get(), 2, false);
        convert_src.type = src_type;
        convert_src.ne[0] = max_elem;
        convert_src.nb[0] = ggml_type_size(src_type);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            convert_src.ne[i] = 1;
            convert_src.nb[i] = convert_src.nb[i - 1] * convert_src.ne[i - 1];
        }
        convert_dst = ggml_cast(ctx.get(), &convert_src, dst_type);
        ggml_set_output(convert_dst);
        ggml_build_forward_expand(graph, convert_dst);

        gallocr.reset(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
        ggml_gallocr_reserve(gallocr.get(), graph);

        backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    }

    ggml_type target_type(ggml_tensor const* t) const {
        if (dst_type == GGML_TYPE_COUNT || !is_float_type(t->type)) {
            return t->type;
        }
        return dst_type;
    }

    void const* operator()(ggml_tensor const* src, ggml_tensor const* dst, bool whcn_to_cwhn) {
        bool need_type_conv = is_float_type(src->type) && src->type != dst_type;
        if (dst_type == GGML_TYPE_COUNT || !(need_type_conv || whcn_to_cwhn)) {
            return src->data;
        }
        ASSERT(ctx, "Weights contain tensors that would require conversion");

        convert_src.type = src->type;
        convert_src.data = src->data;
        std::copy(src->ne, src->ne + GGML_MAX_DIMS, convert_src.ne);
        std::copy(src->nb, src->nb + GGML_MAX_DIMS, convert_src.nb);
        if (whcn_to_cwhn) {
            bool depthwise = convert_src.ne[2] == 1;
            permute_whcn_to_cwhn(convert_src.ne, depthwise);
            permute_whcn_to_cwhn(convert_src.nb, depthwise);
        }

        ASSERT(convert_dst->type == dst->type);
        std::copy(dst->ne, dst->ne + GGML_MAX_DIMS, convert_dst->ne);
        std::copy(dst->nb, dst->nb + GGML_MAX_DIMS, convert_dst->nb);

        bool alloc_ok = ggml_gallocr_alloc_graph(gallocr.get(), graph);
        ASSERT(alloc_ok);

        ggml_backend_graph_compute(backend.get(), graph);
        return convert_dst->data;
    }
};

span<int32_t const> find_conv2d_weight_indices(model_file const& f) {
    gguf_context* gguf = f.gguf.get();
    auto name = format<fixed_string<64>>("{}.conv2d_weights", f.arch());
    int64_t key = gguf_find_key(gguf, name.c_str());
    if (key != -1 && gguf_get_arr_type(gguf, key) == GGUF_TYPE_INT32) {
        size_t n = gguf_get_arr_n(gguf, key);
        int32_t const* a = reinterpret_cast<int32_t const*>(gguf_get_arr_data(gguf, key));
        return span(a, n);
    }
    return {};
}

} // namespace

void model_transfer(
    ggml_context* const& src_ctx,
    model_weights& weights,
    backend_device const& device,
    ggml_type float_type,
    tensor_data_layout src_layout,
    tensor_data_layout dst_layout,
    span<int32_t const> conv2d_weights) {

    ggml_context* dst_ctx = weights.context.get();
    bool to_cwhn = src_layout == tensor_data_layout::whcn && dst_layout == tensor_data_layout::cwhn;
    tensor_converter convert(src_ctx, float_type, to_cwhn);

    tensor orig = ggml_get_first_tensor(src_ctx);
    for (int64_t i = 0, conv2d_idx = 0; orig;) {
        if (strncmp(orig->name, "GGUF", 4) == 0) {
            orig = ggml_get_next_tensor(src_ctx, orig); // skip "GGUF tensor data binary blob"
            continue; // (why is there no way to iterate over GGUF tensors directly?)
        }
        auto ne = nelements(orig);
        if (to_cwhn && conv2d_idx < ssize(conv2d_weights) && conv2d_weights[conv2d_idx] == i) {
            permute_whcn_to_cwhn(ne.data(), ne[2] == 1);
            ++conv2d_idx;
        }
        tensor dup = ggml_new_tensor(dst_ctx, convert.target_type(orig), GGML_MAX_DIMS, ne.data());
        ggml_set_name(dup, ggml_get_name(orig));
        orig = ggml_get_next_tensor(src_ctx, orig);
        ++i;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(dst_ctx, device);
    weights.weights_buffer = ggml_backend_buffer_ptr(buffer);
    weights.buffer_type = device.type();
    if (to_cwhn) {
        weights.flags |= model_build_flag::cwhn;
    }

    tensor src = ggml_get_first_tensor(src_ctx);
    tensor dst = ggml_get_first_tensor(dst_ctx);
    for (int i = 0, conv2d_idx = 0; src && dst;) {
        if (strncmp(src->name, "GGUF", 4) == 0) {
            src = ggml_get_next_tensor(src_ctx, src);
            continue; // skip "GGUF tensor data binary blob"
        }
        bool is_2d = conv2d_idx < int(conv2d_weights.size()) && conv2d_weights[conv2d_idx] == i;
        if (is_2d) {
            ++conv2d_idx;
        }
        void const* data = convert(src, dst, is_2d && to_cwhn);
        ggml_backend_tensor_set(dst, data, 0, ggml_nbytes(dst));
        src = ggml_get_next_tensor(src_ctx, src);
        dst = ggml_get_next_tensor(dst_ctx, dst);
        ++i;
    }
}

void model_transfer(
    model_file const& file,
    model_weights& weights,
    backend_device const& device,
    ggml_type float_type,
    tensor_data_layout layout) {

    weights.flags = model_get_build_flags(file);
    model_transfer(
        file.data.get(), weights, device, float_type, file.tensor_layout(), layout,
        find_conv2d_weight_indices(file));
}

ggml_type model_weights::float_type() const {
    for (ggml_tensor* t = ggml_get_first_tensor(context.get()); t != nullptr;
         t = ggml_get_next_tensor(context.get(), t)) {
        if (is_float_type(t->type)) {
            return t->type; // return first float type found
        }
    }
    return GGML_TYPE_COUNT;
}

//
// compute_graph

compute_graph compute_graph_init(size_t size) {
    ggml_init_params graph_ctx_params{};
    graph_ctx_params.mem_size = size * ggml_tensor_overhead() + ggml_graph_overhead();
    graph_ctx_params.no_alloc = true;
    ggml_context* ctx = ggml_init(graph_ctx_params);
    ggml_context_ptr ctx_ptr(ctx);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, size, false);
    return compute_graph{std::move(ctx_ptr), graph, nullptr};
}

bool compute_graph_allocate(compute_graph& g, backend_device const& backend) {
    if (!g.allocr) {
        g.allocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    }
    bool result = ggml_gallocr_alloc_graph(g.allocr.get(), g.graph);
    if (!result) {
        throw std::runtime_error("Failed to allocate buffer for graph");
    }
    return result;
}

void compute(compute_graph const& g, backend_device const& b) {
    ggml_backend_graph_compute(b, g.graph);
}

//
// model_ref

model_ref::model_ref(model_weights& m)
    : weights_context(m.context.get()),
      graph_context(m.context.get()),
      graph(nullptr),
      flags(m.flags | backend_default_flags(m.buffer_type)) {}

model_ref::model_ref(model_weights& m, compute_graph& g)
    : weights_context(m.context.get()),
      graph_context(g.context.get()),
      graph(g.graph),
      flags(m.flags | backend_default_flags(m.buffer_type)) {}

model_ref::model_ref(
    ggml_context* weights_context,
    ggml_context* graph_context,
    ggml_cgraph* graph,
    model_build_flags flags,
    tensor_name prefix)
    : weights_context(weights_context),
      graph_context(graph_context ? graph_context : weights_context),
      graph(graph),
      flags(flags),
      prefix(prefix) {}

tensor model_ref::find(char const* name) const {
    auto full_name = tensor_name();
    if (prefix) {
        name = format(full_name, "{}.{}", prefix.c_str(), name);
    }
    return ggml_get_tensor(weights_context, name);
}

tensor model_ref::weights(char const* name) const {
    if (tensor result = find(name)) {
        return result;
    }
    throw except("tensor not found: {}.{}", prefix.view(), name);
}

model_ref model_ref::with_prefix(tensor_name new_prefix) const {
    return model_ref{weights_context, graph_context, graph, flags, new_prefix};
}

template <typename Stringable>
model_ref chain_prefix(model_ref const& m, Stringable sub_module) {
    if (m.prefix) {
        return m.with_prefix(format<tensor_name>("{}.{}", m.prefix.view(), sub_module));
    } else {
        return m.with_prefix(format<tensor_name>("{}", sub_module));
    }
}

model_ref model_ref::operator[](char const* sub_module) const {
    return chain_prefix(*this, sub_module);
}
model_ref model_ref::operator[](tensor_name sub_module) const {
    return chain_prefix(*this, sub_module.view());
}
model_ref model_ref::operator[](int sub_module) const {
    return chain_prefix(*this, sub_module);
}

tensor named(model_ref const& m, tensor tensor) {
    ggml_set_name(tensor, m.prefix.c_str());
    return tensor;
}

//
// tensor creation and data handling

tensor compute_graph_input(model_ref const& m, ggml_type type, i64x4 shape, tensor_name name) {
    tensor x = ggml_new_tensor_4d(m, type, shape[0], shape[1], shape[2], shape[3]);
    ggml_set_name(x, name.c_str());
    ggml_set_input(x);
    return x;
}

tensor compute_graph_output(model_ref const& m, tensor x, tensor_name name) {
    ggml_set_name(x, name.c_str());
    ggml_set_output(x);
    ggml_build_forward_expand(m.graph, x);
    return x;
}

tensor_data tensor_alloc(tensor x) {
    return {x, std::unique_ptr<byte[]>(new byte[ggml_nbytes(x)])};
}

tensor_data tensor_load(tensor x, char const* filepath) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        throw except("Failed to open file: {}", filepath);
    }
    tensor_data result = tensor_alloc(x);
    size_t read = fread(result.data.get(), 1, ggml_nbytes(x), file);
    fclose(file);
    if (read != ggml_nbytes(x)) {
        throw except("Failed to read data from file: {}", filepath);
    }
    return result;
}

std::span<float> tensor_data::as_f32() {
    ASSERT(x->type == GGML_TYPE_F32);
    return span(reinterpret_cast<float*>(data.get()), ggml_nelements(x));
}

std::span<float const> tensor_data::as_f32() const {
    ASSERT(x->type == GGML_TYPE_F32);
    return span(reinterpret_cast<float const*>(data.get()), ggml_nelements(x));
}

std::span<int32_t> tensor_data::as_i32() {
    ASSERT(x->type == GGML_TYPE_I32);
    return span(reinterpret_cast<int32_t*>(data.get()), ggml_nelements(x));
}

std::span<int32_t const> tensor_data::as_i32() const {
    ASSERT(x->type == GGML_TYPE_I32);
    return span(reinterpret_cast<int32_t const*>(data.get()), ggml_nelements(x));
}

void transfer_to_backend(tensor_data const& d) {
    ggml_backend_tensor_set(d.x, d.data.get(), 0, ggml_nbytes(d.x));
}

void transfer_to_backend(tensor x, std::span<const byte> data) {
    ASSERT(ggml_nbytes(x) == data.size_bytes());
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));
}

void transfer_to_backend(tensor x, std::span<const float> data) {
    ASSERT(ggml_nbytes(x) == data.size_bytes());
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));
}

void transfer_to_backend(tensor x, image_view const& img) {
    ASSERT(ggml_nbytes(x) == n_bytes(img));
    ggml_backend_tensor_set(x, img.data, 0, ggml_nbytes(x));
}

tensor_data transfer_from_backend(tensor x) {
    tensor_data result = tensor_alloc(x);
    ggml_backend_tensor_get(x, result.data.get(), 0, ggml_nbytes(x));
    return result;
}

void transfer_from_backend(tensor x, span<float> dst, size_t offset) {
    size_t size = std::min(dst.size_bytes(), ggml_nbytes(x) - offset);
    ggml_backend_tensor_get(x, dst.data(), offset, size);
}

void transfer_from_backend(tensor x, image_span const& dst) {
    ASSERT(ggml_nbytes(x) == n_bytes(dst));
    ggml_backend_tensor_get(x, dst.data, 0, ggml_nbytes(x));
}

//
// tensor operations

tensor slice(model_ref const& m, tensor x, slice_t s0, slice_t s1, slice_t s2, slice_t s3) {
    ASSERT(s0.step == 1 && "Slice step must be 1 for the begin dimension");

    auto ne = std::array{x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    auto nb = std::array{x->nb[0], x->nb[1], x->nb[2], x->nb[3]};
    auto slices = std::array{s0, s1, s2, s3};
    size_t offset = 0;

    for (int dim = 0; dim < 4; ++dim) {
        auto [begin, end, step] = slices[dim];
        end = end == slice_t::max ? x->ne[dim] : end;
        end = end < 0 ? x->ne[dim] + end : end;
        begin = begin < 0 ? x->ne[dim] + begin : begin;
        ASSERT(begin >= 0 && end <= x->ne[dim] && "Slice indices out of bounds");
        ASSERT(begin < end && "Begin index must be less than end index");

        ne[dim] = (end - begin + step - 1) / step;
        nb[dim] = x->nb[dim] * step;
        offset += begin * x->nb[dim];
    }
    return ggml_view_4d(m, x, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset);
}

tensor concat(model_ref const& m, std::array<tensor, GGML_MAX_SRC> src, int dim) {
    int n = (int)std::count_if(src.begin(), src.end(), [](tensor t) { return t != nullptr; });
    if (m.flags & model_build_flag::concat_n) {
        return ggml_concat_n(m, src.data(), n, dim);
    } else {
        tensor x = src[0];
        for (int i = 1; i < n; ++i) {
            x = ggml_concat(m, x, src[i], dim);
        }
        return x;
    }
}

tensor interpolate(model_ref const& m, tensor x, i64x2 target, int32_t mode) {
    if ((m.flags & model_build_flag::cwhn) && mode == GGML_SCALE_MODE_NEAREST) {
        return ggml_interpolate(m, x, x->ne[0], target[0], target[1], x->ne[3], mode);
    }
    // Bilinear interpolation requires WHCN layout!
    return ggml_interpolate(m, x, target[0], target[1], x->ne[2], x->ne[3], mode);
}

} // namespace visp
