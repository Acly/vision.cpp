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

void expect_images(span<image_view> images, size_t count) {
    if (images.size() != count) {
        throw except("Expected {} input images, but got {}.", count, images.size());
    }
}

template <model_family f>
struct model_funcs {};

template <>
struct model_funcs<model_family::sam> {
    using model_t = sam_model;

    static sam_model load(char const* filepath, backend_device const& dev) {
        return sam_load_model(filepath, dev);
    }
    static image_data compute(sam_model& m, span<image_view> inputs, span<int> prompt) {
        expect_images(inputs, 1);
        sam_encode(m, inputs[0]);
        if (prompt.size() == 2) {
            return sam_compute(m, i32x2{prompt[0], prompt[1]});
        } else if (prompt.size() == 4) {
            return sam_compute(m, box_2d{i32x2{prompt[0], prompt[1]}, i32x2{prompt[2], prompt[3]}});
        } else {
            throw except("sam: bad number of arguments ({}), must be 2 or 4", prompt.size());
        }
    }
};

template <>
struct model_funcs<model_family::birefnet> {
    using model_t = birefnet_model;

    static birefnet_model load(char const* filepath, backend_device const& dev) {
        return birefnet_load_model(filepath, dev);
    }
    static image_data compute(birefnet_model& m, span<image_view> inputs, span<int>) {
        expect_images(inputs, 1);
        return birefnet_compute(m, inputs[0]);
    }
};

template <>
struct model_funcs<model_family::depth_anything> {
    using model_t = depthany_model;

    static depthany_model load(char const* filepath, backend_device const& dev) {
        return depthany_load_model(filepath, dev);
    }
    static image_data compute(depthany_model& m, span<image_view> inputs, span<int>) {
        expect_images(inputs, 1);
        image_data result_f32 = depthany_compute(m, inputs[0]);
        image_data normalized = image_normalize(result_f32);
        return image_f32_to_u8(normalized, image_format::alpha_u8);
    }
};

template <>
struct model_funcs<model_family::migan> {
    using model_t = migan_model;

    static migan_model load(char const* filepath, backend_device const& dev) {
        return migan_load_model(filepath, dev);
    }
    static image_data compute(migan_model& m, span<image_view> inputs, span<int>) {
        expect_images(inputs, 2);
        if (inputs[1].format != image_format::alpha_u8) {
            throw except("migan: second input image (mask) must be alpha_u8 format");
        }
        return migan_compute(m, inputs[0], inputs[1]);
    }
};

template <>
struct model_funcs<model_family::esrgan> {
    using model_t = esrgan_model;

    static esrgan_model load(char const* filepath, backend_device const& dev) {
        return esrgan_load_model(filepath, dev);
    }
    static image_data compute(esrgan_model& m, span<image_view> inputs, span<int>) {
        expect_images(inputs, 1);
        return esrgan_compute(m, inputs[0]);
    }
};

template <typename F>
void dispatch_model(model_family family, F&& f) {
    switch (family) {
        case model_family::sam: f(model_funcs<model_family::sam>{}); break;
        case model_family::birefnet: f(model_funcs<model_family::birefnet>{}); break;
        case model_family::depth_anything: f(model_funcs<model_family::depth_anything>{}); break;
        case model_family::migan: f(model_funcs<model_family::migan>{}); break;
        case model_family::esrgan: f(model_funcs<model_family::esrgan>{}); break;
        default: throw visp::exception("Unsupported model family");
    }
}

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

//
// public C interface

extern "C" {

VISP_API char const* visp_get_last_error() {
    return _error_string.c_str();
}

// image

VISP_API void visp_image_destroy(image_data* img) {
    delete img;
}

// device

VISP_API int32_t visp_backend_load_all(char const* dir) {
    ggml_backend_load_all_from_path(dir);
    return (int32_t)ggml_backend_reg_count();
}

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

struct any_model;

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
        dispatch_model(family, [&](auto funcs) {
            using model_t = typename decltype(funcs)::model_t;
            *out = reinterpret_cast<any_model*>(new model_t(funcs.load(filepath, *dev)));
        });
    });
}

VISP_API void visp_model_destroy(any_model* model, int32_t arch) {
    model_family family = model_family(arch);
    dispatch_model(family, [&](auto funcs) {
        using model_t = typename decltype(funcs)::model_t;
        delete reinterpret_cast<model_t*>(model);
    });
}

VISP_API int32_t visp_model_compute(
    any_model* model,
    int32_t family,
    image_view* inputs,
    int32_t n_inputs,
    int32_t* args,
    int32_t n_args,
    visp_image_view* out_image,
    image_data** out_data) {

    return handle_errors([&]() {
        span<image_view> input_views(inputs, n_inputs);
        span<int32_t> input_args(args, n_args);

        dispatch_model(model_family(family), [&](auto funcs) {
            using model_t = typename decltype(funcs)::model_t;
            model_t& m = *reinterpret_cast<model_t*>(model);
            image_data result = funcs.compute(m, input_views, input_args);
            return_image(out_data, out_image, std::move(result));
        });
    });
}

} // extern "C"