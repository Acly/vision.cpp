#include "testing.h"
#include "visp/image.h"
#include "visp/ml.h"
#include "visp/util.h"
#include "visp/vision.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace visp {
using clock = std::chrono::high_resolution_clock;
using duration_ms = std::chrono::duration<double, std::milli>;

char const* usage = R"(
Usage: vision-bench [-m <model1>] [-m <model2> ...] [options]

Run benchmarks on one or more vision models and print results as table.
If no model is specified, benchmarks all supported models.

Options:
    -m, --model <arch>       Model architecture (sam, birefnet, depthany, ...)
    -m, --model <arch:file>  Specific model file, eg. "birefnet:BiRefNet-F16.gguf"
    -b, --backend <cpu|gpu>  Backend type (default: all backends)
    --timeout <seconds>      Benchmark timeout in seconds (default: 10)
    --min-iterations <n>     Minimum benchmark iterations (default: 4)
    --max-iterations <n>     Maximum benchmark iterations (default: 100)
)";

struct bench_args {
    duration_ms timeout = duration_ms(10000);
    int min_iterations = 4;
    int max_iterations = 100;
};

struct bench_timings {
    duration_ms total;
    duration_ms mean;
    duration_ms stdev;
    int iterations = 0;
};

struct input_transfer {
    tensor x;
    span<byte const> data;

    input_transfer(tensor x, span<byte const> data) : x(x), data(data) {}
    input_transfer(tensor x, image_view img) : x(x), data((byte const*)img.data, n_bytes(img)) {}
};

bench_timings run_benchmark(
    compute_graph& graph,
    backend_device& backend,
    bench_args const& args,
    std::vector<input_transfer> const& transfers = {}) {

    std::vector<double> timings;
    timings.reserve(args.max_iterations);

    compute(graph, backend); // Warm-up

    auto start = clock::now();
    int i = 0;
    for (i = 0; i < args.max_iterations; ++i) {
        auto start_iteration = clock::now();

        for (const auto& transfer : transfers) {
            transfer_to_backend(transfer.x, transfer.data);
        }
        compute(graph, backend);

        auto end = clock::now();
        duration_ms elapsed = end - start_iteration;
        timings.push_back(elapsed.count());

        if (i >= args.min_iterations && (end - start) >= args.timeout) {
            i++; // loop counter -> total runs
            break;
        }
    }

    duration_ms total = clock::now() - start;
    double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / timings.size() - mean * mean);
    return {total, duration_ms(mean), duration_ms(stdev), i};
}

bench_timings benchmark_sam(path model_path, backend_device& backend, bench_args const& args) {
    path input_path = test_dir().input / "cat-and-hat.jpg";

    sam_model model = sam_load_model(model_path.string().c_str(), backend);
    image_data input = image_load(input_path.string().c_str());
    image_data input_data = sam_process_input(input, model.params);

    sam_encode(model, image_view(input));
    bench_timings encoder_timings = run_benchmark(
        model.encoder, backend, args, {{model.input_image, input_data}});

    sam_compute(model, i32x2{200, 300});
    bench_timings decoder_timings = run_benchmark(model.decoder, backend, args);

    return {
        encoder_timings.total + decoder_timings.total, encoder_timings.mean + decoder_timings.mean,
        duration_ms(
            std::sqrt(
                encoder_timings.stdev.count() * encoder_timings.stdev.count() +
                decoder_timings.stdev.count() * decoder_timings.stdev.count())),
        encoder_timings.iterations};
}

bench_timings benchmark_birefnet(path model_path, backend_device& backend, bench_args const& args) {
    path input_path = test_dir().input / "wardrobe.jpg";

    birefnet_model model = birefnet_load_model(model_path.string().c_str(), backend);
    image_data input = image_load(input_path.string().c_str());
    image_data input_data = birefnet_process_input(input, model.params);

    birefnet_compute(model, input);
    return run_benchmark(model.graph, backend, args, {{model.input, input_data}});
}

bench_timings benchmark_depth_anything(
    path model_path, backend_device& backend, bench_args const& args) {
    path input_path = test_dir().input / "wardrobe.jpg";

    depthany_model model = depthany_load_model(model_path.string().c_str(), backend);
    image_data input = image_load(input_path.string().c_str());
    depthany_compute(model, input);

    image_data input_data = depthany_process_input(input, model.params);
    return run_benchmark(model.graph, backend, args, {{model.input, input_data}});
}

bench_timings benchmark_migan(path model_path, backend_device& backend, bench_args const& args) {
    path image_path = test_dir().input / "bench-image.jpg";
    path mask_path = test_dir().input / "bench-mask.png";

    migan_model model = migan_load_model(model_path.string().c_str(), backend);
    image_data image = image_load(image_path.string().c_str());
    image_data mask = image_load(mask_path.string().c_str());
    image_data input_data = migan_process_input(image, mask, model.params);

    migan_compute(model, image, mask);
    return run_benchmark(model.graph, backend, args, {{model.input, input_data}});
}

bench_timings benchmark_esrgan(path model_path, backend_device& backend, bench_args const& args) {
    path input_path = test_dir().input / "vase-and-bowl.jpg";

    esrgan_model model = esrgan_load_model(model_path.string().c_str(), backend);
    image_data input = image_load(input_path.string().c_str());
    image_data input_data = image_u8_to_f32(input, image_format::rgb_f32);

    compute_graph graph = compute_graph_init(esrgan_estimate_graph_size(model.params));
    model_ref m(model.weights, graph);
    i64x4 input_shape = {3, input.extent[0], input.extent[1], 1};
    model.input = compute_graph_input(m, GGML_TYPE_F32, input_shape);
    model.output = esrgan_generate(m, model.input, model.params);

    compute_graph_allocate(graph, backend);
    return run_benchmark(graph, backend, args, {{model.input, input_data}});
}

backend_device initialize_backend(std::string_view backend_type) {
    if (backend_type == "cpu") {
        backend_device cpu = backend_init(backend_type::cpu);
        backend_set_n_threads(cpu, (int)std::thread::hardware_concurrency());
        return cpu;
    } else if (backend_type == "vulkan") {
        return backend_init(backend_type::vulkan);
    } else if (backend_type == "gpu") {
        return backend_init(backend_type::gpu);
    } else {
        throw std::invalid_argument("Invalid backend type. Use 'cpu', 'gpu' or 'vulkan'.");
    }
}

struct bench_result {
    std::string_view arch;
    std::string_view model;
    std::string_view backend;
    bench_timings time;
};

bench_result benchmark_model(
    std::string_view arch,
    std::string_view model,
    backend_device& backend,
    bench_args const& args) {

    bench_result result;
    result.arch = arch;
    result.model = model;
    result.backend = to_string(backend.type());

    auto select_model = [&](std::string_view model, std::string_view fallback) {
        if (model.empty()) {
            result.model = fallback;
            return test_dir().models / fallback;
        }
        path p = path(model);
        if (!exists(p)) {
            fprintf(stderr, "Model file not found: %s\n", p.string().c_str());
            result.model = fallback;
            return test_dir().models / fallback;
        }
        return p;
    };

    if (arch == "sam") {
        path model_path = select_model(model, "MobileSAM-F16.gguf");
        result.time = benchmark_sam(model_path, backend, args);

    } else if (arch == "birefnet") {
        path model_path = select_model(model, "BiRefNet-lite-F16.gguf");
        result.time = benchmark_birefnet(model_path, backend, args);

    } else if (arch == "depthany") {
        path model_path = select_model(model, "Depth-Anything-V2-Small-F16.gguf");
        result.time = benchmark_depth_anything(model_path, backend, args);

    } else if (arch == "migan") {
        path model_path = select_model(model, "MIGAN-512-places2-F16.gguf");
        result.time = benchmark_migan(model_path, backend, args);

    } else if (arch == "esrgan") {
        path model_path = select_model(model, "RealESRGAN-x4plus_anime-6B-F16.gguf");
        result.time = benchmark_esrgan(model_path, backend, args);

    } else {
        fprintf(stderr, "Unknown model architecture: %s\n", arch.data());
    }
    return result;
}

char const* next_arg(int argc, char** argv, int& i) {
    if (++i < argc) {
        return argv[i];
    } else {
        throw except("Missing argument after {}", argv[i - 1]);
    }
}

void print(fixed_string<128> const& str) {
    printf("%s", str.c_str());
}

} // namespace visp

int main(int argc, char** argv) {
    using namespace visp;
    std::vector<std::pair<std::string_view, std::string_view>> models;
    std::vector<std::string_view> backends;
    bench_args args;

    try {

        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);
            if (arg == "-h" || arg == "--help") {
                printf("%s", usage);
                return 0;
            } else if (arg == "-m" || arg == "--model") {
                std::string_view text = next_arg(argc, argv, i);
                auto p = text.find(':');
                if (p == std::string_view::npos) {
                    models.push_back({text, ""});
                } else {
                    std::string_view arch = text.substr(0, p);
                    std::string_view model = text.substr(p + 1);
                    models.emplace_back(arch, model);
                }
            } else if (arg == "-b" || arg == "--backend") {
                backends.push_back(next_arg(argc, argv, i));
            } else if (arg == "--timeout") {
                args.timeout = duration_ms(std::stod(next_arg(argc, argv, i)) * 1000);
            } else if (arg == "--min-iterations") {
                args.min_iterations = std::stoi(next_arg(argc, argv, i));
            } else if (arg == "--max-iterations") {
                args.max_iterations = std::stoi(next_arg(argc, argv, i));
            } else {
                throw std::invalid_argument("Unknown argument: " + std::string(arg));
            }
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    if (models.empty()) {
        models = {{"sam", ""}, {"birefnet", ""}, {"migan", ""}, {"esrgan", ""}};
    }

    if (backends.empty()) {
        backends = {"cpu", "gpu"};
    }

    try {
        fixed_string<128> line;
        size_t n_tests = models.size() * backends.size();
        std::vector<bench_result> results;
        results.reserve(n_tests);

        int i = 0;
        for (auto&& backend : backends) {
            backend_device backend_device = initialize_backend(backend);
            for (auto&& model : models) {
                print(format(
                    line, "[{: <2}/{: <2}] Running {} on {}...", ++i, n_tests, model.first,
                    backend));

                bench_result result = benchmark_model(
                    model.first, model.second, backend_device, args);

                print(format(
                    line, " finished {} runs in {:.1f} s\n", result.time.iterations,
                    result.time.total.count() / 1000.0));
                results.push_back(result);
            }
        }

        printf("\n");
        print(format(
            line, "| {: <10} | {: <30} | {: <6} | {: >11} | {: >6} |\n", "Arch", "Model", "Device",
            "Avg", "Dev"));
        printf("|:-----------|:-------------------------------|:-------|------------:|-------:|\n");
        for (const auto& result : results) {
            auto model = result.model.substr(std::max(int(result.model.length()) - 35, 0));
            model = model.substr(0, model.find_last_of('.'));
            print(format(
                line, "| {: <10} | {: <30} | {: <6} | {:8.1f} ms | {:6.1f} |\n", result.arch, model,
                result.backend, result.time.mean.count(), result.time.stdev.count()));
        }
        printf("\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
