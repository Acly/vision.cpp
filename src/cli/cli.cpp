#include "util/math.hpp"
#include "util/string.hpp"
#include "visp/vision.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace visp {
using std::filesystem::path;

enum class cli_command { sam, birefnet, migan, esrgan };

struct cli_args {
    cli_command command;
    std::vector<char const*> inputs;   // -i --input
    char const* output = "output.png"; // -o --output
    char const* model = nullptr;       // -m --model
    std::vector<char const*> prompt;   // -p --prompt
    // int threads = -1; // -t --threads
    // bool verbose = false; // -v --verbose
    std::optional<backend_type> backend_type; // -b --backend
    // std::string_view device = 0; // -d --device
    // ggml_type float_type = GGML_TYPE_COUNT; // -f32 -f16

    char const* composite = nullptr; // --composite
    int tile_size = -1;              // --tile
};

char const* next_arg(int argc, char** argv, int& i) {
    if (++i < argc) {
        return argv[i];
    } else {
        throw error("Missing argument after {}", argv[i - 1]);
    }
}

std::vector<char const*> collect_args(int argc, char** argv, int& i, char delim = '-') {
    std::vector<char const*> r;
    do {
        r.push_back(next_arg(argc, argv, i));
    } while (i + 1 < argc && argv[i + 1][0] != delim);
    if (r.empty()) {
        throw error("Missing argument after {}", argv[i - 1]);
    }
    return r;
}

int parse_int(std::string_view arg) {
    int value = 0;
    auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (ec != std::errc()) {
        throw error("Invalid integer argument: {}", arg);
    }
    return value;
}

char const* validate_path(char const* arg) {
    if (!exists(path(arg))) {
        throw error("File not found: {}", arg);
    }
    return arg;
}

void require_inputs(std::span<char const* const> inputs, int n_required, char const* names) {
    if (inputs.size() != size_t(n_required)) {
        throw error(
            "Expected -i to be followed by {} inputs: {} - but found {}.", n_required, names,
            inputs.size());
    }
}

cli_args cli_parse(int argc, char** argv) {
    cli_args r;
    if (argc < 2) {
        throw error("Missing command.\nUsage: {} <command> [options]", argv[0]);
    }

    std::string_view arg1 = argv[1];
    if (arg1 == "sam") {
        r.command = cli_command::sam;
    } else if (arg1 == "birefnet") {
        r.command = cli_command::birefnet;
    } else if (arg1 == "migan") {
        r.command = cli_command::migan;
    } else if (arg1 == "esrgan") {
        r.command = cli_command::esrgan;
    } else {
        throw error("Unknown command: {}", arg1);
    }

    for (int i = 2; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            r.inputs = collect_args(argc, argv, i);
            for_each(r.inputs.begin(), r.inputs.end(), validate_path);
        } else if (arg == "-o" || arg == "--output") {
            r.output = next_arg(argc, argv, i);
        } else if (arg == "-m" || arg == "--model") {
            r.model = validate_path(next_arg(argc, argv, i));
        } else if (arg == "-p" || arg == "--prompt") {
            r.prompt = collect_args(argc, argv, i, '-');
        } else if (arg == "-b" || arg == "--backend") {
            std::string_view backend_arg = next_arg(argc, argv, i);
            if (backend_arg == "cpu") {
                r.backend_type = backend_type::cpu;
            } else if (backend_arg == "gpu") {
                r.backend_type = backend_type::gpu;
            } else {
                throw error("Unknown backend type '{}', must be one of: cpu, gpu", backend_arg);
            }
        } else if (arg == "--composite") {
            r.composite = next_arg(argc, argv, i);
        } else if (arg == "--tile") {
            r.tile_size = parse_int(next_arg(argc, argv, i));
        } else if (arg.starts_with("-")) {
            throw error("Unknown argument: {}", arg);
        }
    }
    return r;
}

void run_sam(cli_args const&);
void run_birefnet(cli_args const&);
void run_migan(cli_args const&);
void run_esrgan(cli_args const&);

} // namespace visp

//
// main

int main(int argc, char** argv) {
    using namespace visp;
    try {
        ggml_time_init();

        cli_args args = cli_parse(argc, argv);
        switch (args.command) {
            case cli_command::sam: run_sam(args); break;
            case cli_command::birefnet: run_birefnet(args); break;
            case cli_command::migan: run_migan(args); break;
            case cli_command::esrgan: run_esrgan(args); break;
        }

    } catch (std::exception const& e) {
        printf("Error: %s\n", e.what());
        return 1;
    } catch (...) {
        return -1;
    }
    return 0;
}

namespace visp {

struct timer {
    int64_t start;
    fixed_string<16> string;

    timer() : start(ggml_time_us()) {}

    int64_t elapsed() const { return ggml_time_us() - start; }
    float elapsed_ms() const { return float(elapsed()) / 1000.0f; }

    char const* elapsed_str() {
        format(string, "{:.1f} ms", elapsed_ms());
        return string.c_str();
    }
};

//
// Common helpers

backend backend_init(cli_args const& args) {
    timer t;
    printf("Initializing backend... ");

    backend b;
    if (args.backend_type) {
        b = backend_init(*args.backend_type);
    } else {
        b = backend_init();
    }
    printf("done (%s)\n", t.elapsed_str());

    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    char const* dev_name = ggml_backend_dev_name(dev);
    char const* dev_desc = ggml_backend_dev_description(dev);
    printf("- device: %s - %s\n", dev_name, dev_desc);
    return b;
}

model_weights load_model_weights(
    cli_args const& args, backend const& b, char const* default_model, int n_tensors = 0) {

    timer t;
    char const* model_path = args.model ? args.model : default_model;
    printf("Loading model weights from '%s'... ", model_path);

    model_load_params load_params = {
        .float_type = b.preferred_float_type(),
        .n_extra_tensors = n_tensors,
    };
    model_weights weights = model_load(model_path, b, load_params);

    printf("done (%s)\n", t.elapsed_str());
    printf("- float type: %s\n", ggml_type_name(weights.float_type()));
    return weights;
}

void compute_timed(compute_graph const& g, backend const& b) {
    timer t;
    printf("Running inference... ");
    compute(g, b);
    printf("complete (%s)\n", t.elapsed_str());
}

void composite_image_with_mask(image_view image, image_view mask, char const* output_path) {
    if (!output_path) {
        return;
    }

    image_data_f32 image_f32 = image_alloc_f32(image.extent, 4);
    image_u8_to_f32(image, image_f32);

    image_data_f32 mask_f32 = image_alloc_f32(mask.extent, 1);
    image_u8_to_f32(mask, mask_f32);

    image_data_f32 foreground = image_estimate_foreground(image_f32, mask_f32);
    image_data output = image_alloc(image.extent, image_format::rgba);
    image_f32_to_u8(foreground.as_span().elements(), span(output.data.get(), n_bytes(output)));
    image_save(output, output_path);
    printf("-> image composited and saved to %s\n", output_path);
}

//
// SAM

struct sam_prompt {
    i32x2 point1 = {-1, -1};
    i32x2 point2 = {-1, -1};

    bool is_point() const { return point2[0] == -1 || point2[1] == -1; }
    bool is_box() const { return !is_point(); }
};

sam_prompt sam_parse_prompt(std::span<char const* const> args, i32x2 extent) {
    if (args.empty()) {
        throw error(
            "SAM requires a prompt with coordinates for a point or box"
            "eg. '--prompt 100 200' to pick the point at pixel (x=100, y=200)");
    }
    if (args.size() < 2 || args.size() > 4) {
        throw error(
            "Invalid number of arguments for SAM prompt. Expected 2 (point) or 4 (box) numbers, "
            "got {}",
            args.size());
    }
    i32x2 a{-1, -1};
    if (args.size() >= 2) {
        a = {parse_int(args[0]), parse_int(args[1])};
        if (a[0] < 0 || a[1] < 0 || a[0] >= extent[0] || a[1] >= extent[1]) {
            throw error("Invalid image coordinates: ({}, {})", a[0], a[1]);
        }
    }
    i32x2 b{-1, -1};
    if (args.size() == 4) {
        b = {parse_int(args[2]), parse_int(args[3])};
        if (b[0] < 0 || b[1] < 0 || b[0] >= extent[0] || b[1] >= extent[1]) {
            throw error("Invalid image coordinates: ({}, {})", b[0], b[1]);
        }
        if (a[0] >= b[0] || a[1] >= b[1]) {
            throw error("Invalid box coordinates: ({}, {}) to ({}, {})", a[0], a[1], b[0], b[1]);
        }
    }
    return sam_prompt{a, b};
};

void run_sam(cli_args const& args) {
    backend backend = backend_init(args);
    model_weights weights = load_model_weights(args, backend, "models/mobile-sam.gguf");
    sam_params params{};

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    image_data_f32 image_data_ = sam_process_input(image, params);

    sam_prompt prompt = sam_parse_prompt(args.prompt, image.extent);
    f32x4 prompt_data = prompt.is_point()
        ? sam_process_point(prompt.point1, image.extent, params)
        : sam_process_box(prompt.point1, prompt.point2, image.extent, params);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    tensor image_tensor = create_input(m, GGML_TYPE_F32, {3, 1024, 1024, 1}, "image");
    tensor point_tensor = create_input(m, GGML_TYPE_F32, {2, 2, 1, 1}, "points");

    tensor image_embed = sam_encode_image(m, image_tensor, params);
    tensor prompt_embed = prompt.is_point() ? sam_encode_points(m, point_tensor)
                                            : sam_encode_box(m, point_tensor);

    sam_prediction output = sam_predict(m, image_embed, prompt_embed);

    allocate(graph, backend);
    transfer_to_backend(image_tensor, image_data_);
    transfer_to_backend(point_tensor, span(prompt_data.v, 4));

    compute_timed(graph, backend);

    timer t_post;
    printf("Postprocessing output... ");

    tensor_data iou = transfer_from_backend(output.iou);
    tensor_data mask_data = transfer_from_backend(output.masks);

    image_data mask = sam_process_mask(mask_data.as_f32(), 2, image.extent, params);
    printf("complete (%s)\n", t_post.elapsed_str());

    image_save(mask, args.output);

    auto ious = iou.as_f32();
    printf("-> estimated accuracy (IoU): %f, %f, %f\n", ious[0], ious[1], ious[2]);
    printf("-> mask saved to %s\n", args.output);

    composite_image_with_mask(image, mask, args.composite);
}

//
// BirefNet

void run_birefnet(cli_args const& args) {
    backend backend = backend_init(args);
    model_weights weights = load_model_weights(args, backend, "models/birefnet.gguf", 6);
    birefnet_params params = birefnet_detect_params(weights);
    int img_size = params.image_size;

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    image_data_f32 input_data = birefnet_process_input(image, params);

    birefnet_buffers buffers = birefnet_precompute(model_ref(weights), params);
    allocate(weights, backend);
    for (tensor_data const& buf : buffers) {
        transfer_to_backend(buf);
    }

    compute_graph graph = compute_graph_init(6 * 1024);
    model_ref m(weights, graph);

    tensor input = create_input(m, GGML_TYPE_F32, {3, img_size, img_size, 1});
    tensor output = birefnet_predict(m, input, params);

    allocate(graph, backend);
    transfer_to_backend(input, input_data);

    compute_timed(graph, backend);

    tensor_data mask_data = transfer_from_backend(output);
    image_data mask = image_alloc({img_size, img_size}, image_format::alpha);
    image_f32_to_u8(mask_data.as_f32(), span(mask.data.get(), n_bytes(mask)));
    image_data mask_resized = image_resize(mask, image.extent);
    image_save(mask_resized, args.output);
    printf("-> mask saved to %s\n", args.output);

    composite_image_with_mask(image, mask_resized, args.composite);
}

//
// MI-GAN

void run_migan(cli_args const& args) {
    backend backend = backend_init(args);
    model_weights weights = load_model_weights(args, backend, "models/migan_512_places2-f16.gguf");
    migan_params params = migan_detect_params(weights);
    params.invert_mask = true; // -> inpaint opaque areas

    require_inputs(args.inputs, 2, "<image> <mask>");
    image_data image = image_load(args.inputs[0]);
    image_data mask = image_load(args.inputs[1]);
    image_data_f32 input_data = migan_process_input(image, mask, params);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    tensor input = create_input(m, GGML_TYPE_F32, {4, params.resolution, params.resolution, 1});
    tensor output = migan_generate(m, input, params);

    allocate(graph, backend);
    transfer_to_backend(input, input_data);

    compute_timed(graph, backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data output_image = migan_process_output(output_data.as_f32(), image.extent, params);
    image_data mask_resized = image_resize(mask, image.extent);
    image_data composited = image_alloc(image.extent, image_format::rgb);
    image_alpha_composite(output_image, image, mask_resized, composited.data.get());
    image_save(composited, args.output);
    printf("-> output image saved to %s\n", args.output);
}

//
// ESRGAN

void run_esrgan(cli_args const& args) {
    backend backend = backend_init(args);
    model_weights weights = load_model_weights(args, backend, "models/RealESRGAN_x4.gguf");
    esrgan_params params = esrgan_detect_params(weights);

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    int tile_size = args.tile_size > 0 ? args.tile_size : 224;

    tile_layout tiles = tile_layout(image.extent, tile_size, 16);
    tile_layout tiles_out = tile_scale(tiles, params.scale);
    image_data_f32 input_tile = image_alloc_f32(tiles.tile_size, 3);
    image_data_f32 output_tile = image_alloc_f32(tiles_out.tile_size, 3);
    image_data_f32 output_image = image_alloc_f32(image.extent * params.scale, 3);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    tensor input = create_input(m, GGML_TYPE_F32, {3, tiles.tile_size[0], tiles.tile_size[1], 1});
    tensor output = esrgan_generate(m, input, params);

    allocate(graph, backend);

    timer total;
    printf(
        "Using tile size %d with %d overlap -> %dx%d tiles\n", //
        tile_size, tiles.overlap[0], tiles.n_tiles[0], tiles.n_tiles[1]);

    for (int t = 0; t < tiles.total(); ++t) {
        printf("\rRunning inference... tile %d of %d", t + 1, tiles.total());
        i32x2 tile_coord = tiles.coord(t);
        i32x2 tile_offset = tiles.start(tile_coord);
        
        image_u8_to_f32(image, input_tile, f32x4{0, 0, 0, 0}, f32x4{1, 1, 1, 1}, tile_offset);
        transfer_to_backend(input, input_tile);

        compute(graph, backend);

        transfer_from_backend(output, output_tile.as_span().elements());
        tile_merge(output_tile, output_image, tile_coord, tiles_out);
    }
    printf("\rRunning inference... complete (%s)\n", total.elapsed_str());

    image_data output_u8 = image_alloc(image.extent * params.scale, image_format::rgb);
    image_f32_to_u8(
        output_image.as_span().elements(), span(output_u8.data.get(), n_bytes(output_u8)));
    image_save(output_u8, args.output);
    printf("-> output image saved to %s\n", args.output);
}

} // namespace visp