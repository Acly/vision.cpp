#include "visp/vision.h"
#include "util/math.h"
#include "util/string.h"

namespace visp {

//
// Mobile SAM

sam_model sam_load_model(char const* filepath, backend_device const& dev) {
    sam_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = sam_params{};
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

void sam_encode(sam_model& model, image_view image) {
    if (!model.encoder) {
        model.encoder = compute_graph_init();
        model_ref m = model_ref(model.weights, model.encoder);

        int res = model.params.image_size;
        model.input_image = compute_graph_input(m, GGML_TYPE_F32, {3, res, res, 1});
        tensor embeds = sam_encode_image(m, model.input_image, model.params);
        model.output_embed = compute_graph_output(m, embeds);
        compute_graph_allocate(model.encoder, *model.backend);
    }

    model.image_extent = image.extent;
    image_data img_data = sam_process_input(image, model.params);
    transfer_to_backend(model.input_image, img_data);
    compute(model.encoder, *model.backend);
}

image_data sam_compute_impl(sam_model& model, i32x2 point1, i32x2 point2) {
    ASSERT(model.image_extent[0] > 0, "Missing image embeds, call sam_encode() first");
    bool is_point = point2 == i32x2{-1, -1};

    if (!model.decoder || model.is_point_prompt != is_point) {
        model.is_point_prompt = is_point;

        model.decoder = compute_graph_init();
        model_ref m(model.weights, model.decoder);
        model.input_embed = compute_graph_input(m, GGML_TYPE_F32, {256, 64, 64, 1});
        model.input_prompt = compute_graph_input(m, GGML_TYPE_F32, {2, 2, 1, 1}, "input_prompt");
        tensor prompt_embed = is_point ? sam_encode_points(m, model.input_prompt)
                                       : sam_encode_box(m, model.input_prompt);
        model.output = sam_predict_mask(m, model.input_embed, prompt_embed);

        compute_graph_allocate(model.decoder, *model.backend);
    }

    f32x4 prompt_data = is_point
        ? sam_process_point(point1, model.image_extent, model.params)
        : sam_process_box({point1, point2}, model.image_extent, model.params);
    transfer_to_backend(model.input_prompt, span(prompt_data.v, 4));
    ggml_backend_tensor_copy(model.output_embed, model.input_embed);

    compute(model.decoder, *model.backend);

    tensor_data iou_data = transfer_from_backend(model.output.iou);
    tensor_data mask_data = transfer_from_backend(model.output.masks);
    auto iou = iou_data.as_f32().subspan(0, 3);
    int idx = int(std::max_element(iou.begin(), iou.end()) - iou.begin());
    return sam_process_mask(mask_data.as_f32(), idx, model.image_extent, model.params);
}

image_data sam_compute(sam_model& model, i32x2 point) {
    return sam_compute_impl(model, point, i32x2{-1, -1});
}

image_data sam_compute(sam_model& model, box_2d box) {
    return sam_compute_impl(model, box.top_left, box.bottom_right);
}

//
// BiRefNet

birefnet_model birefnet_load_model(char const* filepath, backend_device const& dev) {
    birefnet_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = birefnet_detect_params(file, {1024, 1024});
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

image_data birefnet_compute(birefnet_model& model, image_view image) {
    i32x2 res = birefnet_image_extent(image.extent, model.params, model.backend->max_alloc());
    if (!model.graph || res != model.params.image_extent) {
        model.params.image_extent = res;
        model.graph = compute_graph_init(6 * 1024);

        model_ref m(model.weights, model.graph);
        birefnet_buffers buffers = birefnet_precompute(m, model.params);
        model.input = compute_graph_input(m, GGML_TYPE_F32, {3, res[0], res[1], 1});
        model.output = birefnet_predict(m, model.input, model.params);

        compute_graph_allocate(model.graph, *model.backend);
        for (tensor_data const& buf : buffers) {
            transfer_to_backend(buf);
        }
    }

    image_data img_data = birefnet_process_input(image, model.params);
    transfer_to_backend(model.input, img_data);

    compute(model.graph, *model.backend);

    tensor_data mask_data = transfer_from_backend(model.output);
    return birefnet_process_output(mask_data.as_f32(), image.extent, model.params);
}

//
// MI-GAN

migan_model migan_load_model(char const* filepath, backend_device const& dev) {
    migan_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = migan_detect_params(file);
    model.params.invert_mask = true; // inpaint opaque areas
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

image_data migan_compute(migan_model& model, image_view image, image_view mask) {
    if (!model.graph) {
        model.graph = compute_graph_init();
        model_ref m(model.weights, model.graph);

        int res = model.params.resolution;
        model.input = compute_graph_input(m, GGML_TYPE_F32, {4, res, res, 1});
        model.output = migan_generate(m, model.input, model.params);
        compute_graph_allocate(model.graph, *model.backend);
    }

    image_data input_data = migan_process_input(image, mask, model.params);
    transfer_to_backend(model.input, input_data);

    compute(model.graph, *model.backend);

    tensor_data output_data = transfer_from_backend(model.output);
    image_data output = migan_process_output(output_data.as_f32(), image.extent, model.params);
    image_set_alpha(output, mask);
    return output;
}

//
// ESRGAN

constexpr int esrgan_default_tile_size = 224;

esrgan_model esrgan_load_model(char const* filepath, backend_device const& dev) {
    esrgan_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = esrgan_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

image_data esrgan_compute(esrgan_model& model, image_view image) {
    tile_layout tiles(image.extent, esrgan_default_tile_size, 16);
    if (!model.graph || model.tile_size != tiles.tile_size) {
        model.tile_size = tiles.tile_size;
        model.graph = compute_graph_init(esrgan_estimate_graph_size(model.params));

        model_ref m(model.weights, model.graph);
        i64x4 input_shape = {3, tiles.tile_size[0], tiles.tile_size[1], 1};
        model.input = compute_graph_input(m, GGML_TYPE_F32, input_shape);
        model.output = esrgan_generate(m, model.input, model.params);

        compute_graph_allocate(model.graph, *model.backend);
    }

    tile_layout tiles_out = tile_scale(tiles, model.params.scale);
    image_data input_tile = image_alloc(tiles.tile_size, image_format::rgb_f32);
    image_data output_tile = image_alloc(tiles_out.tile_size, image_format::rgb_f32);
    image_data output_image = image_alloc(image.extent * model.params.scale, image_format::rgb_f32);
    image_clear(output_image);

    for (int t = 0; t < tiles.total(); ++t) {
        i32x2 tile_coord = tiles.coord(t);
        i32x2 tile_offset = tiles.start(tile_coord);

        image_u8_to_f32(image, input_tile, f32x4{0, 0, 0, 0}, f32x4{1, 1, 1, 1}, tile_offset);
        transfer_to_backend(model.input, input_tile);

        compute(model.graph, *model.backend);

        transfer_from_backend(model.output, output_tile);
        tile_merge(output_tile, output_image, tile_coord, tiles_out);
    }
    return image_f32_to_u8(output_image, image_format::rgba_u8);
}

} // namespace visp