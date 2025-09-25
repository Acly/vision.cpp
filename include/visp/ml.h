#pragma once

#include "visp/image.h"
#include "visp/util.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace visp {
using std::byte;
using std::span;
using tensor_name = fixed_string<GGML_MAX_NAME>;
using tensor = ggml_tensor*;

// Memory layout, especially for weights of 2D operations like convolutions
enum tensor_data_layout { unknown, whcn, cwhn };

//
// Backend device - represents the compute hardware

enum class backend_type { cpu = 1, gpu = 2 };

// True if the backend library is loaded and has at least one supported device.
VISP_API bool backend_is_available(backend_type);

struct backend_device {
    ggml_backend_ptr handle;
    ggml_backend_dev_t device;

    VISP_API backend_type type() const;
    VISP_API ggml_type preferred_float_type() const;
    VISP_API tensor_data_layout preferred_layout() const;
    VISP_API size_t total_memory() const;
    VISP_API size_t max_alloc() const;

    operator ggml_backend_t() const { return handle.get(); }
};

// Initialize a backend device, automatically tries to pick the "best" available.
VISP_API backend_device backend_init();

// Initialize the most suited device that matches the specified backend type.
VISP_API backend_device backend_init(backend_type);

// Set number of threads used by the backend (CPU only).
VISP_API void backend_set_n_threads(backend_device&, int n_threads);

//
// Model build flags - backend capabilities, model configuration and optimization

enum class model_build_flag {
    // clang-format off
    cwhn                = 1 << 0,
    conv_2d_direct_cwhn = 1 << 1,
    concat_n            = 1 << 2,
    f16_conv_transpose  = 1 << 3,
    window_partition    = 1 << 4
}; // clang-format on

using model_build_flags = flags<model_build_flag>;

VISP_API model_build_flags backend_default_flags(backend_type);

//
// Model file - holds the contents of a GGUF file

struct model_file {
    gguf_context_ptr gguf;
    ggml_context_ptr data;
    std::string path;

    VISP_API int64_t n_tensors() const;
    VISP_API std::string_view arch() const;
    VISP_API tensor_data_layout tensor_layout() const;

    VISP_API int64_t key(char const* name) const;
    VISP_API int get_int(char const* name) const;
    VISP_API std::string_view get_string(char const* name) const;
};

// Opens a .gguf file and reads its contents into memory.
VISP_API model_file model_load(char const* filepath);

//
// Model weights
//
// * stores the tensor descriptors of model weights
// * holds the backend buffers for model weight data
// * holds buffers for extra tensors such as pre-computed lookup tables

struct model_weights {
    ggml_context_ptr context;
    backend_type buffer_type = backend_type::cpu;
    ggml_backend_buffer_ptr weights_buffer;
    std::vector<ggml_backend_buffer_ptr> extra_buffers;
    model_build_flags flags;

    VISP_API ggml_type float_type() const;

    operator ggml_context*() const { return context.get(); }
};

// Creates a GGML context with storage for a fixed number of tensors.
// Does not allocate any backend buffers.
VISP_API model_weights model_init(size_t n_tensors);

// Allocates backend buffers for the model weights if needed. Does not transfer data.
// Returns false and does nothing if all tensors already have an associated backend buffer.
VISP_API bool model_allocate(model_weights&, backend_device const&);

// Adds model weights contained in `file` to `weights`. Allocates backend buffers for the
// weights on `device` and transfers the data to the device buffer.
// Optionally converts float weights to the specified data type during transfer.
VISP_API void model_transfer(
    model_file const& file,
    model_weights& weights,
    backend_device const& device,
    ggml_type float_type = GGML_TYPE_COUNT,
    tensor_data_layout = tensor_data_layout::unknown);

VISP_API void model_transfer(
    ggml_context* const& src_ctx,
    model_weights& weights,
    backend_device const& device,
    ggml_type float_type = GGML_TYPE_COUNT,
    tensor_data_layout src_layout = tensor_data_layout::unknown,
    tensor_data_layout dst_layout = tensor_data_layout::unknown,
    span<int32_t const> conv2d_weights = {});

//
// Compute graph - wrapper for ggml_cgraph and its associated backend memory

struct compute_graph {
    ggml_context_ptr context;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_ptr allocr;

    operator ggml_cgraph*() const { return graph; }
};

// Initializes a compute graph and associated backend allocator.
VISP_API compute_graph compute_graph_init(size_t size = GGML_DEFAULT_GRAPH_SIZE);

// Allocates memory for inputs, outputs and computations on the backend.
VISP_API bool compute_graph_allocate(compute_graph&, backend_device const&);

// Runs inference. Blocks until done.
VISP_API void compute(compute_graph const&, backend_device const&);

//
// Model ref - represents a ML model
//
// * helper for building compute graphs
// * allows access to the model's weights by name, with an optional name prefix
//   to support nested modules
// * pass anywhere ggml_context* is expected while building the graph

struct VISP_API model_ref {
    ggml_context* weights_context = nullptr;
    ggml_context* graph_context = nullptr;
    ggml_cgraph* graph = nullptr;
    model_build_flags flags;
    tensor_name prefix;

    model_ref() = default;
    model_ref(model_weights&);
    model_ref(model_weights&, compute_graph&);

    explicit model_ref(
        ggml_context* weights_context,
        ggml_context* graph_context = nullptr,
        ggml_cgraph* graph = nullptr,
        model_build_flags flags = {},
        tensor_name prefix = {});

    // Find weights tensor by name, prepends the current prefix.
    tensor find(char const* name) const;    // returns null if not found
    tensor weights(char const* name) const; // asserts if not found

    model_ref with_prefix(tensor_name new_prefix) const;

    // Returns a model_ref with prefix set to <current prefix>.<sub_module>
    model_ref operator[](char const* sub_module) const;
    model_ref operator[](tensor_name sub_module) const;
    model_ref operator[](int sub_module) const;

    operator ggml_context*() const { return graph_context; }
};

// Sets the name of a tensor to the current model prefix.
VISP_API tensor named(model_ref const&, tensor);

// Creates a new tensor as part of the model graph where input data can be stored.
VISP_API tensor compute_graph_input(model_ref const&, ggml_type, i64x4 ne, tensor_name = "input");

// Marks a tensor as an output of the compute graph.
VISP_API tensor compute_graph_output(model_ref const&, tensor, tensor_name = "output");

//
// Tensor data and transfer to backend device

struct VISP_API tensor_data {
    tensor x;
    std::unique_ptr<byte[]> data;

    span<float> as_f32();
    span<int32_t> as_i32();
    span<float const> as_f32() const;
    span<int32_t const> as_i32() const;
};

// Allocates data for a tensor in main memory, outside of context and backend buffers.
VISP_API tensor_data tensor_alloc(tensor x);

// Loads tensor data from a file storing raw numbers as binary.
VISP_API tensor_data tensor_load(tensor x, char const* filepath);

// Copies data to the tensor's backend buffer (which should already be allocated).
VISP_API void transfer_to_backend(tensor_data const&);
VISP_API void transfer_to_backend(tensor x, span<byte const> data);
VISP_API void transfer_to_backend(tensor x, span<float const> data);
VISP_API void transfer_to_backend(tensor x, image_view const& data);

// Copies tensor data from the backend buffer to main memory.
VISP_API tensor_data transfer_from_backend(tensor x);
VISP_API void transfer_from_backend(tensor x, span<float> dst, size_t offset = 0);
VISP_API void transfer_from_backend(tensor x, image_span const& dst);

//
// Tensor operations

// Returns tensor shape. Allows structured binding: `auto [c, w, h, n] = nelements(t);`
inline std::array<int64_t, 4> nelements(tensor t) {
    return {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
}

struct slice_t {
    int64_t begin;
    int64_t end;
    int64_t step;

    static constexpr int64_t max = std::numeric_limits<int64_t>::max();

    // Default: selects the entire range for a dimension (ie. `tensor[:]`)
    constexpr slice_t() : begin(0), end(max), step(1) {}

    // Selects a single slice of a dimension (ie. `tensor[index]`)
    constexpr slice_t(int64_t index) : begin(index), end(index + 1), step(1) {}

    // Selects a range [begin, end) with an optional step (ie. `tensor[begin:end:step]`)
    constexpr slice_t(int64_t begin, int64_t end, int64_t step = 1)
        : begin(begin), end(end), step(step) {}
};

// Slice a tensor along one or more dimensions similar to numpy/torch. Returns a view.
// Example: `x[0, 0:64, 16:32, :]` becomes `slice(m, x, {}, {16, 32}, {0, 64}, 0)`
VISP_API tensor slice(
    model_ref const&, tensor x, slice_t s0, slice_t s1 = {}, slice_t s2 = {}, slice_t s3 = {});

// Concatenate multiple tensors along a specified dimension.
VISP_API tensor concat(model_ref const&, std::array<tensor, GGML_MAX_SRC> src, int dim);

// Up- or downsample a 2D tensor (WHCN) to target width x height.
VISP_API tensor interpolate(model_ref const&, tensor x, i64x2 target, int32_t mode);

//
// SWIN Transformer

struct swin_layer_t {
    int depth;
    int n_heads;
    int n_features;
    bool downsample;
};

struct swin_params {
    static constexpr int n_layers = 4;

    int embed_dim;
    int window_size;
    std::array<swin_layer_t, n_layers> layers;
};

extern swin_params const swin_t_params;
extern swin_params const swin_l_params;
VISP_API swin_params swin_detect_params(model_file const&);

//
// implementation

constexpr model_build_flags operator|(model_build_flag lhs, model_build_flag rhs) {
    return model_build_flags(uint32_t(lhs) | uint32_t(rhs));
}

constexpr model_build_flags operator~(model_build_flag f) {
    return ~model_build_flags(f);
}

} // namespace visp
