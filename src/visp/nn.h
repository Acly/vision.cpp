#pragma once

#include "visp/ml.h"
#include "visp/util.h"

// Common neural network building blocks

namespace visp {

tensor linear(model_ref, tensor x);
tensor layer_norm(model_ref, tensor x, float eps = 1e-5f);

// Permute between CWHN and WHCN tensor dimension ordering. Does not rewrite tensor data.
tensor permute_cwhn_to_whcn(model_ref m, tensor x);
tensor permute_whcn_to_cwhn(model_ref m, tensor x);

// "Contiguous 2D" refers to the layout configured in `m` model flags, ie. the preferred
// memory layout for 2D operations like convolution.
inline bool is_whcn(model_ref m) { return !(m.flags & model_build_flag::cwhn); }
inline bool is_cwhn(model_ref m) { return !!(m.flags & model_build_flag::cwhn); }

// These functions convert between memory layouts, ie. they rewrite tensor data.
tensor cwhn_to_contiguous_2d(model_ref m, tensor x);
tensor whcn_to_contiguous_2d(model_ref m, tensor x);
tensor contiguous_2d_to_cwhn(model_ref m, tensor x);
tensor contiguous_2d_to_whcn(model_ref m, tensor x);

// Always returns number of elements of tensor in width-height-channels-batch order,
// even if that's not how they're stored in memory.
std::array<int64_t, 4> nelements_whcn(model_ref const&, tensor t);

// 2D (convolution) functions
// Input and weight are expected to be in "contiguous 2D" layout as configured in `m`.
tensor conv_2d(model_ref m, tensor x, int stride = 1, int pad = 0);
tensor conv_2d_depthwise(model_ref m, tensor x, int stride = 1, int pad = 0);
tensor conv_2d_deform(
    model_ref m, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad);
tensor conv_transpose_2d(model_ref m, tensor x, int stride);
tensor batch_norm_2d(model_ref, tensor x);

// 2D image to patch embedding using convolution and optional norm. CWHN input and output.
tensor patch_embed(model_ref, tensor x, int patch_size);

} // namespace visp
