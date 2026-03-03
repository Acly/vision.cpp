#include <visp/image.h>
#include <visp/vision.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

using namespace visp;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>\n";
        return 2;
    }

    std::string const model_path = argv[1];

    try {
        backend_device backend = backend_init(backend_type::cpu);
        depthany_model model = depthany_load_model(model_path.c_str(), backend);

        image_data input = image_alloc({64, 64}, image_format::rgb_u8);
        image_clear(input);

        image_data output = depthany_compute(model, input);
        if (output.extent != input.extent) {
            std::cerr << "Unexpected output extent: " << output.extent[0] << "x" << output.extent[1]
                      << " (expected " << input.extent[0] << "x" << input.extent[1] << ")\n";
            return 1;
        }

        std::span<float const> depth = image_view{output}.as_floats();
        double const mean = std::accumulate(depth.begin(), depth.end(), 0.0) / depth.size();
        if (!std::isfinite(mean)) {
            std::cerr << "Depth output mean is not finite\n";
            return 1;
        }
        return 0;
    } catch (std::exception const& ex) {
        std::cerr << "pkg-check failed: " << ex.what() << "\n";
        return 1;
    }
}
