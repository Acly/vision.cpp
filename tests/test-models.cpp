#include "util/string.hpp"
#include "visp/vision.hpp"

#include "testing.hpp"

namespace visp {

void compare_images(std::string_view name, image_view result, float tolerance = 0.01f) {
    path reference_path = test_dir().reference / name;
    path result_path = test_dir().results / name;

    image_save(result, result_path.string().c_str());
    image_data reference = image_load(reference_path.string().c_str());

    test_set_info(format(
        "while comparing images {} and {}", relative(result_path).string(),
        relative(reference_path).string()));
    test_with_tolerance with(tolerance);
    CHECK_IMAGES_EQUAL(result, reference);
}

VISP_BACKEND_TEST(test_mobile_sam)(backend_type bt) {
    path model_path = test_dir().models / "MobileSAM-F16.gguf";
    path input_path = test_dir().input / "cat-and-hat.jpg";

    backend_device b = backend_init(bt);
    sam_model model = sam_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    sam_encode(model, input);
    image_data mask_box = sam_compute(model, box_2d{{180, 110}, {505, 330}});
    image_data mask_point = sam_compute(model, i32x2{200, 300});

    char const* suffix = bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";
    float tolerance = bt == backend_type::cpu ? 0.01f : 0.015f;
    compare_images(format("mobile_sam-box{}", suffix), mask_box, tolerance);
    compare_images(format("mobile_sam-point{}", suffix), mask_point, tolerance);
}

VISP_BACKEND_TEST(test_birefnet)(backend_type bt) {
    path model_path = test_dir().models / "BiRefNet-lite-F16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";
    std::string name = "birefnet";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    birefnet_model model = birefnet_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    image_data output = birefnet_compute(model, input);

    float tolerance = bt == backend_type::cpu ? 0.01f : 0.3f; // TODO: GPU is non-deterministic
    compare_images(name, output, tolerance);
}

VISP_BACKEND_TEST(test_migan)(backend_type bt) {
    path model_path = test_dir().models / "MIGAN-512-places2-F16.gguf";
    path image_path = test_dir().input / "bench-image.jpg";
    path mask_path = test_dir().input / "bench-mask.png";
    std::string name = "migan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    migan_model model = migan_load_model(model_path.string().c_str(), b);
    image_data image = image_load(image_path.string().c_str());
    image_data mask = image_load(mask_path.string().c_str());
    image_data output = migan_compute(model, image, mask);
    image_data composited = image_alpha_composite(output, image, mask);

    compare_images(name, composited);
}

VISP_BACKEND_TEST(test_esrgan)(backend_type bt) {
    path model_path = test_dir().models / "RealESRGAN-x4plus_anime-6B-F16.gguf";
    path input_path = test_dir().input / "vase-and-bowl.jpg";
    std::string name = "esrgan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    esrgan_model model = esrgan_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    image_data output = esrgan_compute(model, input);

    compare_images(name, output);
}

} // namespace visp