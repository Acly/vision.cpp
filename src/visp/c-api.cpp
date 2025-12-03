#include "util/string.h"
#include "visp/vision.h"


using namespace visp;

thread_local fixed_string<512> _error_string{};

void set_error(std::exception const& e) {
    _error_string = e.what();
}

template <typename F>
int32_t handle_errors(F&& f) {
    try {
        f();
    } catch (std::exception const& e) {        
        set_error(e);
        return 0;
    }
    return 1;
}

extern "C" {

VISP_API char const* visp_get_last_error() {
    return _error_string.c_str();
}

// image

struct visp_image_view {
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t format;
    void* data;
};

void put_image(visp_image_view* out, image_view const& img) {
    out->width = img.extent[0];
    out->height = img.extent[1];
    out->stride = img.stride;
    out->format = int32_t(img.format);
    out->data = (void*)img.data;
}

void return_image(image_data** out_data, visp_image_view* out_image, image_data&& img) {
    *out_data = new image_data(std::move(img));
    put_image(out_image, **out_data);
}

VISP_API void visp_image_destroy(image_data* img) {
    delete img;
}

// device

VISP_API int32_t visp_device_init(int32_t type, backend_device** out_device) {
    return handle_errors([&]() {
        if (type == 0) {
            *out_device = new backend_device(backend_init());
        } else {
            *out_device = new backend_device(backend_init(backend_type(type)));
        }
    });
}

VISP_API void visp_device_destroy(backend_device* d) {
    delete d;
}

VISP_API int32_t visp_device_type(backend_device const* d) {
    return int32_t(d->type());
}

VISP_API char const* visp_device_name(backend_device const* d) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(d->device, &props);
    return props.name;
}

VISP_API char const* visp_device_description(backend_device const* d) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(d->device, &props);
    return props.description;
}

// models

struct any_model {};

VISP_API int32_t visp_model_detect_family(char const* filepath, int32_t* out_family) {
    return handle_errors([&]() {
        model_file file = model_load(filepath);
        model_family family = model_detect_family(file);
        *out_family = int32_t(family);
    });
}

VISP_API int32_t visp_model_load(
    char const* filepath, backend_device const* dev, int32_t arch, any_model** out) {

    return handle_errors([&]() {
        model_family family = model_family(arch);
        if (family == model_family::count) {
            model_file file = model_load(filepath);
            family = model_detect_family(file);
        }
        switch (family) {
            case model_family::sam: {
                sam_model model = sam_load_model(filepath, *dev);
                *out = reinterpret_cast<any_model*>(new sam_model(std::move(model)));
                break;
            }
            case model_family::birefnet: {
                birefnet_model model = birefnet_load_model(filepath, *dev);
                *out = reinterpret_cast<any_model*>(new birefnet_model(std::move(model)));
                break;
            }
            case model_family::depth_anything: {
                depthany_model model = depthany_load_model(filepath, *dev);
                *out = reinterpret_cast<any_model*>(new depthany_model(std::move(model)));
                break;
            }
            case model_family::migan: {
                migan_model model = migan_load_model(filepath, *dev);
                *out = reinterpret_cast<any_model*>(new migan_model(std::move(model)));
                break;
            }
            case model_family::esrgan: {
                esrgan_model model = esrgan_load_model(filepath, *dev);
                *out = reinterpret_cast<any_model*>(new esrgan_model(std::move(model)));
                break;
            }
            default: throw visp::exception("Invalid model family");
        }
    });
}

VISP_API void visp_model_destroy(any_model* model, int32_t arch) {
    model_family family = model_family(arch);
    switch (family) {
        case model_family::sam: delete reinterpret_cast<sam_model*>(model); break;
        case model_family::birefnet: delete reinterpret_cast<birefnet_model*>(model); break;
        case model_family::depth_anything: delete reinterpret_cast<depthany_model*>(model); break;
        case model_family::migan: delete reinterpret_cast<migan_model*>(model); break;
        case model_family::esrgan: delete reinterpret_cast<esrgan_model*>(model); break;
        default: fprintf(stderr, "Invalid model family: %d\n", int(family)); break;
    }
}

VISP_API int32_t visp_esrgan_compute(
    esrgan_model* model, image_view in_image, visp_image_view* out_image, image_data** out_data) {

    return handle_errors([&]() {
        image_data result = esrgan_compute(*model, in_image);
        return_image(out_data, out_image, std::move(result));
    });
}

} // extern "C"