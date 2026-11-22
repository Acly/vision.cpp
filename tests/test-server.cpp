//
// Functional tests for the vision-server binary: spawns a local server with
// the MobileSAM model, then exercises the HTTP API (including the
// Expect: 100-continue path that stalls naive HTTP/1.0-style servers).
//
// POSIX only - see the UNIX guard in CMake.

#include "testing.h"
#include "util/json.h"
#include "util/string.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <filesystem>

namespace visp {
using std::filesystem::path;

static int connect_to(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket() failed");
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) != 1) {
        throw std::runtime_error("inet_pton failed");
    }
    if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        ::close(fd);
        throw std::runtime_error("connect() failed");
    }
    return fd;
}

static void send_all(int fd, std::string_view data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
        if (n <= 0) {
            throw std::runtime_error("send() failed");
        }
        sent += size_t(n);
    }
}

static std::string recv_until(int fd, char const* pattern) {
    std::string out;
    timeval tv{10, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    char buf[4096];
    for (;;) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            throw std::runtime_error("timed out waiting for: " + std::string(pattern));
        }
        out.append(buf, size_t(n));
        if (out.find(pattern) != std::string::npos) {
            return out;
        }
    }
}

// The server closes the connection after each response, so the remainder of
// the stream is the complete response.
static std::string read_rest(int fd) {
    std::string out;
    timeval tv{120, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    char buf[65536];
    for (;;) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n == 0) {
            return out;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue; // keep waiting; the server closes after the response
            }
            throw std::runtime_error("recv() failed");
        }
        out.append(buf, size_t(n));
    }
}

struct test_server {
    int pid = -1;
    int port = 0;

    test_server() {
        // Pick a free port
        int probe = ::socket(AF_INET, SOCK_STREAM, 0);
        if (probe < 0) {
            throw std::runtime_error("socket() failed");
        }
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = 0;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t len = sizeof(a);
        if (::bind(probe, (sockaddr*)&a, sizeof(a)) < 0 ||
            ::getsockname(probe, (sockaddr*)&a, &len) < 0) {
            throw std::runtime_error("port discovery failed");
        }
        port = ntohs(a.sin_port);
        ::close(probe);

        path server_path = test_dir().root / "build" / "bin" / "vision-server";
        if (!std::filesystem::exists(server_path)) {
            throw std::runtime_error("vision-server binary not found at " + server_path.string());
        }
        path model = test_dir().models / "MobileSAM-F16.gguf";

        pid = fork();
        if (pid == 0) {
            // Child
            std::vector<std::string> storage;
            storage.push_back(server_path.string());
            storage.push_back("-l");
            storage.push_back("127.0.0.1");
            storage.push_back("-p");
            storage.push_back(std::to_string(port));
            storage.push_back("-b");
            storage.push_back("cpu");
            storage.push_back("-m");
            storage.push_back(model.string());
            std::vector<char*> argv;
            for (auto& s : storage) {
                argv.push_back(const_cast<char*>(s.c_str()));
            }
            argv.push_back(nullptr);
            execv(server_path.c_str(), argv.data());
            _exit(127);
        }

        // Wait for /health
        for (int i = 0; i < 120; ++i) {
            try {
                int fd = connect_to(port);
                send_all(fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
                read_rest(fd);
                ::close(fd);
                return;
            } catch (...) {
                if (i % 5 == 4) {
                    // Check the child is still alive - a crash should fail the test,
                    // not wait out the whole timeout.
                    int st = 0;
                    if (::waitpid(pid, &st, WNOHANG) == pid) {
                        throw std::runtime_error("vision-server exited during startup");
                    }
                }
                usleep(500 * 1000);
            }
        }
        throw std::runtime_error("vision-server did not come up");
    }

    ~test_server() {
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            int st = 0;
            ::waitpid(pid, &st, 0);
        }
    }
};

test_server& server() {
    static test_server s;
    return s;
}

static std::string image_b64() {
    path p = test_dir().input / "cat-and-hat.jpg";
    std::ifstream f(p, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (raw.empty()) {
        throw std::runtime_error("test image missing: " + p.string());
    }
    std::vector<std::byte> data;
    data.resize(raw.size());
    std::memcpy(data.data(), raw.data(), raw.size());
    return base64_encode(data);
}

static int http_status(std::string const& response) {
    if (response.size() < 12 || response.substr(0, 5) != "HTTP/") {
        throw std::runtime_error("bad response: " + response.substr(0, 64));
    }
    return std::atoi(response.substr(9, 3).c_str());
}

static json_value post_json(char const* route, std::string const& body) {
    int fd = connect_to(server().port);
    std::string req = std::string("POST ") + route +
        " HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    send_all(fd, req);
    std::string resp = read_rest(fd);
    ::close(fd);
    size_t sep = resp.find("\r\n\r\n");
    if (sep == std::string::npos) {
        throw std::runtime_error("no response body");
    }
    CHECK(http_status(resp) == 200);
    return json_parse(resp.substr(sep + 4));
}

static json_value post_json_expect_status(
    char const* route, std::string const& body, int expected) {
    int fd = connect_to(server().port);
    std::string req = std::string("POST ") + route +
        " HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    send_all(fd, req);
    std::string resp = read_rest(fd);
    ::close(fd);
    CHECK(http_status(resp) == expected);
    size_t sep = resp.find("\r\n\r\n");
    if (sep == std::string::npos) {
        return json_value();
    }
    try {
        return json_parse(resp.substr(sep + 4));
    } catch (...) {
        return json_value(); // error bodies may be non-JSON on the worst path
    }
}

// The first eight bytes must be the PNG signature; width/height come from the
// IHDR chunk (big-endian u32s at offsets 16 and 20).
static void check_png(std::string const& b64, int expect_w, int expect_h) {
    auto bytes = base64_decode(b64);
    CHECK_EQUAL(bytes.size() > 33, true);
    char const* sig = "\x89PNG\r\n\x1a\n";
    CHECK(std::memcmp(bytes.data(), sig, 8) == 0);
    auto be32 = [&](size_t off) {
        return (int)((uint32_t(uint8_t(bytes[off])) << 24) |
                     (uint32_t(uint8_t(bytes[off + 1])) << 16) |
                     (uint32_t(uint8_t(bytes[off + 2])) << 8) | uint32_t(uint8_t(bytes[off + 3])));
    };
    CHECK_EQUAL(be32(16), expect_w);
    CHECK_EQUAL(be32(20), expect_h);
}

VISP_TEST(server_health) {
    int fd = connect_to(server().port);
    send_all(fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
    std::string resp = read_rest(fd);
    ::close(fd);
    CHECK(http_status(resp) == 200);
    size_t sep = resp.find("\r\n\r\n");
    json_value v = json_parse(resp.substr(sep + 4));
    CHECK_EQUAL(v.get_string("status"), std::string("ok"));
    CHECK_EQUAL(v.find("models")->array.size(), size_t(1));
    CHECK_EQUAL(v.find("models")->array[0].get_string("family"), std::string("sam"));
}

VISP_TEST(server_sam_point) {
    std::string body = "{\"image\": \"" + image_b64() + "\", \"point\": {\"x\": 100, \"y\": 100}}";
    json_value v = post_json("/sam", body);
    int w = v.get_int("width");
    int h = v.get_int("height");
    CHECK(w > 0 && h > 0);
    CHECK_EQUAL(v.find("iou")->array.size(), size_t(4));
    CHECK_EQUAL(v.find("masks")->array.size(), size_t(4));
    for (auto const& m : v.find("masks")->array) {
        check_png(m.string, w, h);
    }
}

VISP_TEST(server_sam_box) {
    std::string body = "{\"image\": \"" + image_b64() +
        "\", \"box\": {\"x1\": 10, \"y1\": 10, \"x2\": 200, \"y2\": 200}}";
    json_value v = post_json("/sam", body);
    CHECK_EQUAL(v.find("masks")->array.size(), size_t(4));
}

VISP_TEST(server_sam_box_rejects_inverted) {
    std::string body = "{\"image\": \"" + image_b64() +
        "\", \"box\": {\"x1\": 300, \"y1\": 300, \"x2\": 100, \"y2\": 100}}";
    post_json_expect_status("/sam", body, 400);
}

VISP_TEST(server_expect_100_continue) {
    std::string body = "{\"image\": \"" + image_b64() + "\", \"point\": {\"x\": 100, \"y\": 100}}";
    int fd = connect_to(server().port);
    std::string head = "POST /sam HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: "
                       "application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\nExpect: 100-continue\r\n\r\n";
    send_all(fd, head);
    std::string interim = recv_until(fd, "100 Continue");
    CHECK(interim.find("HTTP/1.1 100 Continue") != std::string::npos);
    send_all(fd, body);
    std::string resp = read_rest(fd);
    ::close(fd);
    CHECK(http_status(resp) == 200);
    size_t sep = resp.find("\r\n\r\n");
    json_value v = json_parse(resp.substr(sep + 4));
    CHECK_EQUAL(v.find("masks")->array.size(), size_t(4));
}

VISP_TEST(server_bad_json) {
    post_json_expect_status("/sam", "{not json at all", 400);
}

VISP_TEST(server_bad_image) {
    std::string body = "{\"image\": \"!!!!\", \"point\": {\"x\": 1, \"y\": 1}}";
    post_json_expect_status("/sam", body, 400);
}

VISP_TEST(server_unknown_route) {
    int fd = connect_to(server().port);
    send_all(fd, "GET /nope HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
    std::string resp = read_rest(fd);
    ::close(fd);
    CHECK(http_status(resp) == 404);
}

VISP_TEST(server_unloaded_model_404) {
    std::string body = "{\"image\": \"" + image_b64() + "\"}";
    post_json_expect_status("/birefnet", body, 404);
}

} // namespace visp
