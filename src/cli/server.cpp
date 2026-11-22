//
// vision-server - a small HTTP server for running vision.cpp models
//
// Serves one or more GGUF models over HTTP with a minimal JSON API.
// Designed to run on the same machine as the models (127.0.0.1 by default)
// and to be cheap enough to live next to other GPU tenants: on a Vulkan
// backend the requested arena is probed against the device before the
// server starts, so it fails at boot instead of stealing memory later.
//
// POSIX only for now (Linux, macOS). A Windows port is a follow-up.

#include "util/json.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/vision.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace visp {

//
// Limits and defaults

constexpr size_t MAX_HEADER_BYTES = 64 * 1024;
constexpr size_t MAX_BODY_BYTES = 64 * 1024 * 1024;
constexpr int DEFAULT_PORT = 8080;
constexpr size_t DEFAULT_MAX_ARENA_MB = 4096;

struct timer {
    int64_t start;

    timer() : start(ggml_time_us()) {}

    float elapsed_ms() const { return float(ggml_time_us() - start) / 1000.0f; }

    // Formats into a caller-provided buffer - returning a pointer to a local
    // string would dangle the moment this function returns.
    void elapsed_str(fixed_string<16>& s) const { format(s, "{:.1f} ms", elapsed_ms()); }
};

// Accepts plain base64 and data: URL prefixes (eg. from browser clients)
std::string_view base64_of(std::string_view in) {
    if (in.starts_with("data:")) {
        size_t comma = in.find(',');
        if (comma == std::string_view::npos) {
            throw except("Malformed data: URL");
        }
        return in.substr(comma + 1);
    }
    return in;
}

//
// HTTP

struct http_request {
    std::string method;
    std::string path;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;

    std::string const* header(char const* name) const {
        for (auto const& [k, v] : headers) {
            if (k == name) {
                return &v;
            }
        }
        return nullptr;
    }
};

ssize_t write_all(int fd, void const* data, size_t size) {
    size_t sent = 0;
    while (sent < size) {
        ssize_t n = ::write(fd, static_cast<char const*>(data) + sent, size - sent);
        if (n <= 0) {
            return -1;
        }
        sent += size_t(n);
    }
    return ssize_t(sent);
}

bool read_all(int fd, void* data, size_t size) {
    size_t got = 0;
    while (got < size) {
        ssize_t n = ::read(fd, static_cast<char*>(data) + got, size - got);
        if (n <= 0) {
            return false;
        }
        got += size_t(n);
    }
    return true;
}

// Reads one HTTP request. If the client sends "Expect: 100-continue", a
// "100 Continue" response is sent before the body is read, so large uploads
// (base64 images) don't stall behind clients that wait for the signal.
http_request http_read(int fd) {
    http_request req;

    std::string head;
    head.reserve(1024);
    char buf[4096];
    bool complete = false;
    while (!complete && head.size() < MAX_HEADER_BYTES) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n <= 0) {
            throw except("Connection closed while reading request");
        }
        head.append(buf, size_t(n));
        complete = head.find("\r\n\r\n") != std::string::npos;
    }
    if (!complete) {
        throw except("Request header too large");
    }
    size_t head_end = head.find("\r\n\r\n");
    std::string_view head_sv = std::string_view(head).substr(0, head_end);

    // Request line
    size_t line_end = head_sv.find("\r\n");
    std::string_view line = head_sv.substr(
        0, line_end == std::string_view::npos ? head_sv.size() : line_end);
    {
        size_t sp1 = line.find(' ');
        size_t sp2 = line.rfind(' ');
        if (sp1 == std::string_view::npos || sp2 == sp1) {
            throw except("Malformed request line");
        }
        req.method = std::string(line.substr(0, sp1));
        req.path = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));
    }

    // Headers
    size_t pos = line_end + 2;
    while (pos < head_sv.size()) {
        size_t eol = head_sv.find("\r\n", pos);
        std::string_view h = head_sv.substr(
            pos, eol == std::string_view::npos ? head_sv.size() - pos : eol - pos);
        size_t colon = h.find(':');
        if (colon != std::string_view::npos) {
            std::string name;
            for (size_t i = 0; i < colon; ++i) {
                name += char(std::tolower(uint8_t(h[i])));
            }
            size_t vs = colon + 1;
            while (vs < h.size() && h[vs] == ' ') {
                ++vs;
            }
            req.headers.emplace_back(std::move(name), std::string(h.substr(vs)));
        }
        if (eol == std::string_view::npos) {
            break;
        }
        pos = eol + 2;
    }

    // Body
    size_t content_length = 0;
    if (std::string const* cl = req.header("content-length")) {
        char* end = nullptr;
        unsigned long v = std::strtoul(cl->c_str(), &end, 10);
        if (end == cl->c_str() || *end != '\0' || v > MAX_BODY_BYTES) {
            throw except("Invalid Content-Length: {}", *cl);
        }
        content_length = v;
    }

    if (std::string const* expect = req.header("expect")) {
        if (*expect == "100-continue") {
            char const* cont = "HTTP/1.1 100 Continue\r\n\r\n";
            if (write_all(fd, cont, strlen(cont)) < 0) {
                throw except("Failed to send 100-continue");
            }
        }
    }

    // The reads above may have captured part of the body already - seed with
    // it, then read the rest.
    req.body.assign(head, head_end + 4);
    if (req.body.size() > content_length) {
        throw except("Request body larger than Content-Length");
    }
    while (req.body.size() < content_length) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n <= 0) {
            throw except("Connection closed while reading body");
        }
        req.body.append(buf, size_t(n));
        if (req.body.size() > MAX_BODY_BYTES) {
            throw except("Request body too large");
        }
    }
    req.body.resize(content_length);
    return req;
}

char const* http_status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

struct http_response {
    int status = 200;
    std::string body;

    void json(json_value const& v) { json_write(body, v); }

    void error(int status_, char const* msg) {
        status = status_;
        json_value v;
        v.set("error", std::string(msg));
        json(v);
    }
};

void http_write(int fd, http_response const& r) {
    std::string head = "HTTP/1.1 " + std::to_string(r.status) + " " + http_status_text(r.status) +
        "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(r.body.size()) +
        "\r\nConnection: close\r\n\r\n";
    if (write_all(fd, head.data(), head.size()) < 0) {
        return;
    }
    if (!r.body.empty()) {
        write_all(fd, r.body.data(), r.body.size());
    }
}

// Error with an HTTP status - thrown for client errors (4xx) and inference
// failures (5xx) alike; everything else (unexpected) stays a plain except.
struct http_error : exception {
    int status = 500;

    template <typename... Args>
    http_error(int status_, char const* fmt, Args&&... args)
        : exception(format<fixed_string<128>>(fmt, std::forward<Args>(args)...)), status(status_) {}
};

//
// Images in memory (stb)

image_data image_decode_rgb(std::span<byte const> bytes) {
    int w = 0, h = 0, ch = 0;
    unsigned char* pixels = stbi_load_from_memory(
        reinterpret_cast<unsigned char const*>(bytes.data()), int(bytes.size()), &w, &h, &ch, 3);
    if (!pixels) {
        throw except("Could not decode image ({} bytes)", bytes.size());
    }
    image_data img{.extent = {w, h}, .format = image_format::rgb_u8};
    img.data.reset(new unsigned char[size_t(w) * h * 3]);
    std::memcpy(img.data.get(), pixels, size_t(w) * h * 3);
    stbi_image_free(pixels);
    return img;
}

image_data image_decode_alpha(std::span<byte const> bytes) {
    int w = 0, h = 0, ch = 0;
    unsigned char* pixels = stbi_load_from_memory(
        reinterpret_cast<unsigned char const*>(bytes.data()), int(bytes.size()), &w, &h, &ch, 0);
    if (!pixels) {
        throw except("Could not decode mask image ({} bytes)", bytes.size());
    }
    image_data img{.extent = {w, h}, .format = image_format::alpha_u8};
    img.data.reset(new unsigned char[size_t(w) * h]);
    if (ch == 1) {
        std::memcpy(img.data.get(), pixels, size_t(w) * h);
    } else if (ch == 4) {
        for (int i = 0; i < w * h; ++i) {
            img.data[i] = pixels[i * 4 + 3];
        }
    } else {
        throw except("Mask image must be grayscale or have an alpha channel (got {} channels)", ch);
    }
    stbi_image_free(pixels);
    return img;
}

std::vector<byte> image_encode_png(image_view const& img) {
    std::vector<byte> out;
    auto write_cb = [](void* ctx, void* data, int size) {
        auto& vec = *static_cast<std::vector<byte>*>(ctx);
        vec.insert(
            vec.end(), static_cast<byte const*>(data), static_cast<byte const*>(data) + size);
    };
    if (!stbi_write_png_to_func(
            write_cb, &out, img.extent[0], img.extent[1], n_channels(img.format), img.data,
            img.extent[0] * n_bytes(img.format))) {
        throw except("PNG encoding failed");
    }
    return out;
}

json_value image_to_json(image_view const& img) {
    json_value v;
    v.set("width", int(img.extent[0]));
    v.set("height", int(img.extent[1]));
    v.set("image", base64_encode(image_encode_png(img)));
    return v;
}

//
// Models

char const* to_string(model_family f) {
    switch (f) {
        case model_family::sam: return "sam";
        case model_family::birefnet: return "birefnet";
        case model_family::depth_anything: return "depth";
        case model_family::migan: return "migan";
        case model_family::esrgan: return "esrgan";
        default: return "unknown";
    }
}

// FNV-1a, for the SAM embedding cache key
uint64_t hash_bytes(std::span<unsigned char const> data) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char b : data) {
        h ^= uint64_t(b);
        h *= 0x100000001b3ULL;
    }
    return h;
}

struct sam_embed_cache {
    bool valid = false;
    uint64_t key = 0;
    ggml_type embed_type = GGML_TYPE_F32;
    i64x4 embed_ne = {};
    std::optional<tensor_data> embed_host;
};

struct model_entry {
    model_family family = model_family::sam;
    std::string path;
    bool loaded = false;

    model_file file;
    model_weights weights;

    // SAM: the image encoder (TinyViT) is the expensive half of a predict.
    // The most recent encoder output is kept on the host and re-uploaded on
    // cache hits, so repeated prompts on the same image skip the encode.
    sam_params sam_p;
    sam_embed_cache sam_cache;

    migan_params migan_p;
    esrgan_params esrgan_p;
};

struct server_config {
    std::string listen = "127.0.0.1";
    int port = DEFAULT_PORT;
    std::optional<backend_type> backend;
    std::vector<char const*> models;
    size_t max_arena_mb = DEFAULT_MAX_ARENA_MB;
};

struct server {
    server_config cfg;
    backend_device backend{};
    std::vector<model_entry> models;
    std::mutex infer_mutex;
    std::atomic<bool> running{false};
    int listen_fd = -1;
};

// Proves the backend can hand out an arena of the requested size right now,
// and gives it back. On Vulkan this is a real device-memory check, so the
// server refuses to start on a card already committed to other tenants
// (eg. a big LLM server) instead of failing mid-request later.
void reserve_arena(backend_device const& b, size_t bytes) {
    if (bytes == 0 || b.type() != backend_type::vulkan) {
        return; // CPU memory is not the shared resource this protects
    }
    printf("Checking %zu MB can be reserved on the GPU... ", size_t(bytes) / (1024 * 1024));
    timer t;
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(b, bytes);
    if (!buf) {
        throw except(
            "Cannot reserve an arena of {} MB on {} - not enough free memory "
            "(in use by another client?). Lower --max-arena, free memory, or use --backend cpu.",
            size_t(bytes) / (1024 * 1024), ggml_backend_dev_name(ggml_backend_get_device(b)));
    }
    ggml_backend_buffer_free(buf);
    fixed_string<16> ts;
    t.elapsed_str(ts);
    printf("done (%s)\n", ts.c_str());
}

// The device budget for one request. This ggml-vulkan allocates per tensor
// (there is no arena to clamp), so the budget is checked *before* the graph
// allocates: a dry run of the graph allocator (no_alloc) reports exactly the
// buffer the real allocation would create - in-place ops included, which is
// why summing tensor sizes would overcount - and an over-budget request is
// refused with its measured figure instead of dying mid-allocation inside
// another client's headroom.
void check_device_budget(compute_graph const& g, server& s) {
    ggml_gallocr_t tmp = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));
    size_t need = 0;
    ggml_gallocr_reserve_n_size(tmp, g.graph, nullptr, nullptr, &need);
    ggml_gallocr_free(tmp);

    size_t const cap = s.cfg.max_arena_mb * 1024 * 1024;
    if (need > cap) {
        throw http_error(
            503,
            "this request would use ~{} MB of {} memory, more than the {} MB cap ({}); use a smaller image or raise the cap",
            need / (1024 * 1024),
            ggml_backend_dev_name(ggml_backend_get_device(s.backend)),
            s.cfg.max_arena_mb,
            s.cfg.max_arena_mb == DEFAULT_MAX_ARENA_MB ? "the default" : "--max-arena");
    }
}

model_entry& server_find_model(server& s, model_family family) {
    for (auto& m : s.models) {
        if (m.loaded && m.family == family) {
            return m;
        }
    }
    throw http_error(404, "No {} model loaded", to_string(family));
}

void server_load_models(server& s) {
    for (auto& m : s.models) {
        timer t;
        printf("Loading model weights from '%s'... ", m.path.c_str());
        m.file = model_load(m.path.c_str());
        m.family = model_detect_family(m.file);
        m.weights = model_init(m.file.n_tensors());
        model_transfer(
            m.file, m.weights, s.backend, s.backend.preferred_float_type(),
            s.backend.preferred_layout());
        m.loaded = true;

        switch (m.family) {
            case model_family::sam: m.sam_p = {}; break;
            case model_family::migan:
                m.migan_p = migan_detect_params(m.file);
                m.migan_p.invert_mask = true; // -> inpaint opaque areas
                break;
            case model_family::esrgan: m.esrgan_p = esrgan_detect_params(m.file); break;
            default:
                // sam/birefnet/depth params are detected per request so
                // dynamic-size variants work with any input resolution
                break;
        }
        fixed_string<16> ts;
        t.elapsed_str(ts);
        printf("done (%s)\n", ts.c_str());
        ggml_type ftype = m.file.float_type() != GGML_TYPE_COUNT ? m.file.float_type()
                                                                 : m.weights.float_type();
        printf("- family: %s - float type: %s\n", to_string(m.family), ggml_type_name(ftype));
    }
}

//
// Request handlers

json_value sam_predict(server& s, model_entry& m, image_data const& image, json_value const& req) {
    bool has_point = req.find("point") != nullptr;
    bool has_box = req.find("box") != nullptr;
    if (has_point == has_box) {
        throw http_error(400, "Provide exactly one of 'point' or 'box'");
    }

    f32x4 prompt_data;
    bool is_box = false;
    if (has_point) {
        json_value const& p = *req.find("point");
        i32x2 point{p.get_int("x"), p.get_int("y")};
        if (point[0] < 0 || point[1] < 0 || point[0] >= image.extent[0] ||
            point[1] >= image.extent[1]) {
            throw http_error(400, "Invalid image coordinates: ({}, {})", point[0], point[1]);
        }
        prompt_data = sam_process_point(point, image.extent, m.sam_p);
    } else {
        json_value const& b = *req.find("box");
        i32x2 a{b.get_int("x1"), b.get_int("y1")};
        i32x2 c{b.get_int("x2"), b.get_int("y2")};
        if (a[0] < 0 || a[1] < 0 || c[0] > image.extent[0] || c[1] > image.extent[1] ||
            a[0] >= c[0] || a[1] >= c[1]) {
            throw http_error(
                400, "Invalid box coordinates: ({}, {}) to ({}, {})", a[0], a[1], c[0], c[1]);
        }
        prompt_data = sam_process_box({a, c}, image.extent, m.sam_p);
        is_box = true;
    }

    compute_graph graph = compute_graph_init();
    model_ref model(m.weights, graph);

    uint64_t key = hash_bytes(
        std::span<unsigned char const>(
            image.data.get(), size_t(image.extent[0]) * image.extent[1] * 3));
    tensor image_embed;
    tensor image_tensor{};
    image_data input_data;
    bool cache_hit = false;
    if (m.sam_cache.valid && m.sam_cache.key == key) {
        // Same image as the previous request: re-upload the cached encoder
        // output instead of running TinyViT again. The raw image input is not
        // part of the graph on this path (and must not be transferred).
        image_embed = compute_graph_input(
            model, m.sam_cache.embed_type, m.sam_cache.embed_ne, "image_embed");
        cache_hit = true;
    } else {
        input_data = sam_process_input(image, m.sam_p);
        image_tensor = compute_graph_input(
            model, GGML_TYPE_F32, {3, m.sam_p.image_size, m.sam_p.image_size, 1}, "image");
        image_embed = sam_encode_image(model, image_tensor, m.sam_p);
    }

    tensor point_tensor = compute_graph_input(model, GGML_TYPE_F32, {2, 2, 1, 1}, "points");
    tensor prompt_embed = is_box ? sam_encode_box(model, point_tensor)
                                 : sam_encode_points(model, point_tensor);

    sam_prediction output = sam_predict_mask(model, image_embed, prompt_embed);

    // Allocate before the transfers: input tensors have no buffer until the
    // graph allocation runs (this is why the CLI allocates first).
    check_device_budget(graph, s);
    compute_graph_allocate(graph, s.backend);

    if (cache_hit) {
        // The cached host buffer outlives the graph that produced it, so the
        // tensor it came from is gone - size the span from the cached shape
        // and type instead of asking the (freed) tensor.
        int64_t const ne = m.sam_cache.embed_ne[0] * m.sam_cache.embed_ne[1] *
            m.sam_cache.embed_ne[2] * m.sam_cache.embed_ne[3];
        std::span<byte const> embed_bytes{
            m.sam_cache.embed_host->data.get(),
            size_t(ne) * ggml_type_size(m.sam_cache.embed_type)};
        transfer_to_backend(image_embed, embed_bytes);
    } else {
        transfer_to_backend(image_tensor, input_data);
    }
    transfer_to_backend(point_tensor, span(prompt_data.v, 4));

    compute(graph, s.backend);

    tensor_data iou_data = transfer_from_backend(output.iou);
    tensor_data mask_data = transfer_from_backend(output.masks);

    // Cache the encoder output for the next request on this image.
    auto ne = nelements(image_embed);
    m.sam_cache.valid = true;
    m.sam_cache.key = key;
    m.sam_cache.embed_type = image_embed->type;
    m.sam_cache.embed_ne = {ne[0], ne[1], ne[2], ne[3]};
    m.sam_cache.embed_host = transfer_from_backend(image_embed);

    auto ious = iou_data.as_f32();
    json_value result;
    result.set("width", int(image.extent[0]));
    result.set("height", int(image.extent[1]));

    json_value iou_arr;
    iou_arr.t = json_value::type::array;
    for (float iou : ious) {
        iou_arr.array.push_back(iou);
    }
    result.set("iou", std::move(iou_arr));

    json_value masks_arr;
    masks_arr.t = json_value::type::array;
    for (size_t i = 0; i < 4; ++i) {
        image_data mask = sam_process_mask(mask_data.as_f32(), int(i), image.extent, m.sam_p);
        masks_arr.array.push_back(base64_encode(image_encode_png(mask)));
    }
    result.set("masks", std::move(masks_arr));
    return result;
}

json_value birefnet_predict(server& s, model_entry& m, image_data const& image) {
    size_t max_alloc = std::min(size_t(4096) * 1024 * 1024, s.cfg.max_arena_mb * 1024 * 1024);
    birefnet_params p = birefnet_detect_params(m.file, image.extent, max_alloc);
    image_data input_data = birefnet_process_input(image, p);

    compute_graph graph = compute_graph_init(6 * 1024);
    model_ref model(m.weights, graph);

    // The precomputed buffers live in this request's graph context, so they
    // are rebuilt per request (cheap, host-side work).
    birefnet_buffers buffers = birefnet_precompute(model, p);

    tensor input = compute_graph_input(
        model, GGML_TYPE_F32, {3, p.image_extent[0], p.image_extent[1], 1});
    tensor output = birefnet_predict(model, input, p);

    check_device_budget(graph, s);
    compute_graph_allocate(graph, s.backend);
    transfer_to_backend(input, input_data);
    for (tensor_data const& buf : buffers) {
        transfer_to_backend(buf);
    }
    compute(graph, s.backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data mask = birefnet_process_output(output_data.as_f32(), image.extent, p);
    return image_to_json(mask);
}

json_value depthany_predict(server& s, model_entry& m, image_data const& image) {
    depthany_params p = depthany_detect_params(m.file, image.extent);
    image_data input_data = depthany_process_input(image, p);

    compute_graph graph = compute_graph_init();
    model_ref model(m.weights, graph);
    tensor input = compute_graph_input(
        model, GGML_TYPE_F32, {3, p.image_extent[0], p.image_extent[1], 1});
    tensor output = depthany_predict(model, input, p);

    check_device_budget(graph, s);
    compute_graph_allocate(graph, s.backend);
    transfer_to_backend(input, input_data);
    compute(graph, s.backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data depth_raw = depthany_process_output(output_data.as_f32(), image.extent, p);
    return image_to_json(image_f32_to_u8(depth_raw, image_format::alpha_u8));
}

json_value migan_predict(
    server& s, model_entry& m, image_data const& image, image_data const& mask) {
    image_data input_data = migan_process_input(image, mask, m.migan_p);

    compute_graph graph = compute_graph_init();
    model_ref model(m.weights, graph);
    i64x4 input_shape = {4, m.migan_p.resolution, m.migan_p.resolution, 1};
    tensor input = compute_graph_input(model, GGML_TYPE_F32, input_shape);
    tensor output = migan_generate(model, input, m.migan_p);

    check_device_budget(graph, s);
    compute_graph_allocate(graph, s.backend);
    transfer_to_backend(input, input_data);
    compute(graph, s.backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data output_image = migan_process_output(output_data.as_f32(), image.extent, m.migan_p);
    image_data mask_resized = image_scale(mask, image.extent);
    image_data composited = image_alpha_composite(output_image, image, mask_resized);
    return image_to_json(composited);
}

json_value esrgan_predict(server& s, model_entry& m, image_data const& image, int tile_size) {
    if (tile_size <= 0) {
        throw http_error(400, "Invalid tile size: {}", tile_size);
    }

    tile_layout tiles = tile_layout(image.extent, tile_size, 16);
    tile_layout tiles_out = tile_scale(tiles, m.esrgan_p.scale);
    image_data input_tile = image_alloc(tiles.tile_size, image_format::rgb_f32);
    image_data output_tile = image_alloc(tiles_out.tile_size, image_format::rgb_f32);
    image_data output_image = image_alloc(image.extent * m.esrgan_p.scale, image_format::rgb_f32);
    image_clear(output_image);

    compute_graph graph = compute_graph_init(esrgan_estimate_graph_size(m.esrgan_p));
    model_ref model(m.weights, graph);
    i64x4 input_shape = {3, tiles.tile_size[0], tiles.tile_size[1], 1};
    tensor input = compute_graph_input(model, GGML_TYPE_F32, input_shape);
    tensor output = esrgan_generate(model, input, m.esrgan_p);
    check_device_budget(graph, s);
    compute_graph_allocate(graph, s.backend);

    for (int t = 0; t < tiles.total(); ++t) {
        i32x2 tile_coord = tiles.coord(t);
        i32x2 tile_offset = tiles.start(tile_coord);
        image_u8_to_f32(image, input_tile, f32x4(0), f32x4(1), tile_offset);
        transfer_to_backend(input, input_tile);
        compute(graph, s.backend);
        transfer_from_backend(output, output_tile);
        tile_merge(output_tile, output_image, tile_coord, tiles_out);
    }

    return image_to_json(image_f32_to_u8(output_image, image_format::rgba_u8));
}

json_value server_health(server& s) {
    json_value v;
    v.set("status", std::string("ok"));
    v.set("backend", std::string(to_string(s.backend.type())));

    json_value models;
    models.t = json_value::type::array;
    for (auto const& m : s.models) {
        if (!m.loaded) {
            continue;
        }
        json_value e;
        e.set("family", to_string(m.family));
        e.set("path", m.path);
        models.array.push_back(std::move(e));
    }
    v.set("models", std::move(models));
    return v;
}

// Routes one parsed request. Throws http_error for client errors and
// except for everything unexpected.
json_value server_handle(server& s, http_request const& req) {
    std::string method = req.method;
    std::string path = req.path;

    if (path == "/health") {
        if (method != "GET" && method != "POST") {
            throw http_error(405, "Method not allowed: {} {}", method, path);
        }
        return server_health(s);
    }

    bool known = path == "/sam" || path == "/birefnet" || path == "/depth" ||
        path == "/depth_anything" || path == "/migan" || path == "/esrgan";
    if (method != "POST") {
        // Known routes only accept POST; unknown paths 404 regardless of method.
        if (known) {
            throw http_error(405, "Method not allowed: {} {}", method, path);
        }
        throw http_error(404, "Unknown route: {} {}", method, path);
    }
    if (!known) {
        throw http_error(404, "Unknown route: {} {}", method, path);
    }

    json_value req_json;
    try {
        req_json = json_parse(req.body);
    } catch (std::exception const& e) {
        throw http_error(400, "Invalid JSON: {}", e.what());
    }
    if (!req_json.is_object()) {
        throw http_error(400, "Request body must be a JSON object");
    }

    auto get_image = [&](char const* field) {
        try {
            std::string const& b64 = req_json.get_string(field);
            return image_decode_rgb(base64_decode(base64_of(b64)));
        } catch (std::exception const& e) {
            throw http_error(400, "Invalid '{}' image: {}", field, e.what());
        }
    };

    if (path == "/sam") {
        model_entry& m = server_find_model(s, model_family::sam);
        image_data image = get_image("image");
        return sam_predict(s, m, image, req_json);
    }

    if (path == "/birefnet") {
        model_entry& m = server_find_model(s, model_family::birefnet);
        return birefnet_predict(s, m, get_image("image"));
    }

    if (path == "/depth" || path == "/depth_anything") {
        model_entry& m = server_find_model(s, model_family::depth_anything);
        return depthany_predict(s, m, get_image("image"));
    }

    if (path == "/migan") {
        model_entry& m = server_find_model(s, model_family::migan);
        image_data image = get_image("image");
        image_data mask;
        try {
            mask = image_decode_alpha(base64_decode(base64_of(req_json.get_string("mask"))));
        } catch (std::exception const& e) {
            throw http_error(400, "Invalid 'mask' image: {}", e.what());
        }
        return migan_predict(s, m, image, mask);
    }

    if (path == "/esrgan") {
        model_entry& m = server_find_model(s, model_family::esrgan);
        int tile_size = 224;
        if (json_value const* t = req_json.find("tile")) {
            if (t->is_number()) {
                tile_size = int(t->number);
            } else {
                throw http_error(400, "'tile' must be a number");
            }
        }
        return esrgan_predict(s, m, get_image("image"), tile_size);
    }

    throw http_error(404, "Unknown route: {} {}", method, path);
}

//
// Server lifecycle

void handle_connection(server& s, int fd) {
    timer t;
    int status = 500;
    std::string method = "?";
    std::string path = "?";
    try {
        http_request req = http_read(fd);
        method = req.method;
        path = req.path;
        std::lock_guard lock(s.infer_mutex);
        http_response resp;
        resp.json(server_handle(s, req));
        http_write(fd, resp);
        status = resp.status;
    } catch (std::exception const& e) {
        http_response resp;
        resp.error(500, e.what());
        if (auto const* he = dynamic_cast<http_error const*>(&e)) {
            resp.status = he->status;
        }
        http_write(fd, resp);
        status = resp.status;
    } catch (...) {
        http_response resp;
        resp.error(500, "Internal error");
        http_write(fd, resp);
    }
    ::close(fd);
    fixed_string<16> ts;
    t.elapsed_str(ts);
    fprintf(
        stderr, "[vision-server] %s %s -> %d (%s)\n", method.c_str(), path.c_str(), status,
        ts.c_str());
}

void server_start(server& s) {
    timer t;
    printf("Initializing backend... ");
    if (s.cfg.backend) {
        s.backend = backend_init(*s.cfg.backend);
    } else {
        s.backend = backend_init();
    }
    fixed_string<16> ts;
    t.elapsed_str(ts);
    printf("done (%s)\n", ts.c_str());

    ggml_backend_dev_t dev = ggml_backend_get_device(s.backend);
    printf("- device: %s - %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    reserve_arena(s.backend, s.cfg.max_arena_mb * 1024 * 1024);

    s.models.resize(s.cfg.models.size());
    for (size_t i = 0; i < s.cfg.models.size(); ++i) {
        s.models[i].path = s.cfg.models[i];
    }
    server_load_models(s);

    s.listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s.listen_fd < 0) {
        throw except("socket() failed");
    }
    int on = 1;
    setsockopt(s.listen_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(s.cfg.port));
    if (inet_pton(AF_INET, s.cfg.listen.c_str(), &addr.sin_addr) != 1) {
        throw except("Invalid listen address: {}", s.cfg.listen);
    }
    if (::bind(s.listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        throw except("bind() failed for {}:{}", s.cfg.listen, s.cfg.port);
    }
    if (::listen(s.listen_fd, 16) < 0) {
        throw except("listen() failed");
    }
    printf("Serving on http://%s:%d (Ctrl+C to stop)\n", s.cfg.listen.c_str(), s.cfg.port);
}

//
// CLI

char const* const short_usage = R"(
Usage: vision-server [options]
See 'vision-server --help' for more details.
)";

void print_usage() {
    char const* const usage = R"(
Usage: vision-server [options]

Options:
    -l, --listen <addr>          Address to listen on (default: 127.0.0.1)
    -p, --port <n>               Port to listen on (default: 8080)
    -m, --model <file>           Model file (.gguf), repeatable
    -b, --backend <auto|cpu|gpu> Backend type (default: auto)
    --max-arena <MB>             Device-memory budget per request (default: 4096) - probed at boot
                                 and enforced before each graph allocation, so the server fails
                                 loudly instead of starving other GPU tenants
    -h, --help                   Print usage and exit

Routes (images are PNG or JPEG, base64-encoded; data: URLs accepted):
    GET  /health                 Server and model status
    POST /sam                    {"image": "<b64>", "point": {"x","y"} | "box": {"x1","y1","x2","y2"}}
                                 -> {"width","height","iou":[4],"masks":[4 b64 PNGs]}
    POST /birefnet               {"image": "<b64>"}
                                 -> {"width","height","image": "<b64 PNG, alpha matte>"}
    POST /depth                  {"image": "<b64>"}
                                 -> {"width","height","image": "<b64 PNG, depth>"}
    POST /migan                  {"image": "<b64>", "mask": "<b64>"}
                                 -> {"width","height","image": "<b64 PNG, inpainted>"}
    POST /esrgan                 {"image": "<b64>", "tile"?: <n>}
                                 -> {"width","height","image": "<b64 PNG, upscaled>"}
)";
    printf("%s", usage);
}

char const* next_arg(int argc, char** argv, int& i) {
    if (++i < argc) {
        return argv[i];
    } else {
        throw except("Missing argument after {}", argv[i - 1]);
    }
}

int parse_int(std::string_view arg) {
    int value = 0;
    auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (ec != std::errc()) {
        throw except("Invalid integer argument: {}", arg);
    }
    return value;
}

void parse_args(int argc, char** argv, server_config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-l" || arg == "--listen") {
            cfg.listen = next_arg(argc, argv, i);
        } else if (arg == "-p" || arg == "--port") {
            cfg.port = parse_int(next_arg(argc, argv, i));
        } else if (arg == "-m" || arg == "--model") {
            cfg.models.push_back(next_arg(argc, argv, i));
        } else if (arg == "-b" || arg == "--backend") {
            std::string_view b = next_arg(argc, argv, i);
            if (b == "auto") {
                cfg.backend = std::nullopt;
            } else if (b == "cpu") {
                cfg.backend = backend_type::cpu;
            } else if (b == "gpu") {
                cfg.backend = backend_type::gpu;
            } else {
                throw except("Unknown backend type '{}', must be one of: auto, cpu, gpu", b);
            }
        } else if (arg == "--max-arena") {
            cfg.max_arena_mb = size_t(parse_int(next_arg(argc, argv, i)));
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            std::exit(0);
        } else {
            throw except("Unknown argument: {}\n{}", arg, short_usage);
        }
    }
}

std::atomic<bool> g_stop{false};

} // namespace visp

//
// main

int main(int argc, char** argv) {
    using namespace visp;
    try {
        ggml_time_init();

        server_config cfg;
        parse_args(argc, argv, cfg);

        server s;
        s.cfg = cfg;

        auto on_signal = [](int) {
            g_stop = true;
        };
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);

        server_start(s);
        s.running = true;
        while (s.running && !g_stop) {
            timeval tv{.tv_sec = 0, .tv_usec = 200 * 1000};
            setsockopt(s.listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            sockaddr_in client{};
            socklen_t len = sizeof(client);
            int fd = ::accept(s.listen_fd, (sockaddr*)&client, &len);
            if (fd >= 0) {
                std::thread(handle_connection, std::ref(s), fd).detach();
            }
        }
        s.running = false;
        ::shutdown(s.listen_fd, SHUT_RDWR);
        ::close(s.listen_fd);
    } catch (std::exception const& e) {
        printf("Error: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("Error: unknown\n");
        return 1;
    }
    return 0;
}
