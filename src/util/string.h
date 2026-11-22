#pragma once

#include "visp/util.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef VISP_FMT_LIB
#    include <fmt/format.h>
#else
#    include <format>
namespace fmt {
using std::format;
using std::make_format_args;
using std::vformat_to;
} // namespace fmt
#endif

#define UNUSED(x) (void)(x)

#ifdef VISP_ASSERT_DISABLE
#    define ASSERT(cond, ...) UNUSED(cond)
#else
#    define ASSERT(cond, ...)                                                                      \
        if (!(cond)) {                                                                             \
            visp::assertion_failure(__FILE__, __LINE__, #cond #__VA_ARGS__);                       \
        }
#endif

namespace visp {
using fmt::format;

struct truncating_iterator {
    using iterator_category = std::output_iterator_tag;
    using value_type = char;
    using difference_type = std::ptrdiff_t;
    using pointer = char*;
    using reference = char&;

    char* cur;
    char* end;

    truncating_iterator(char* data, size_t size) : cur(data), end(data + size) {}

    truncating_iterator& operator=(char c) {
        *cur = c;
        return *this;
    }
    truncating_iterator& operator++() {
        if (cur < end - 1) {
            ++cur;
        }
        return *this;
    }
    truncating_iterator operator++(int) {
        truncating_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    truncating_iterator& operator*() { return *this; }
    difference_type operator-(truncating_iterator const& other) const { return cur - other.cur; }
    auto operator<=>(truncating_iterator const&) const = default;
};

template <size_t N, typename... Args>
char const* format(fixed_string<N>& dst, char const* fmt, Args&&... args) {
    auto it = truncating_iterator(dst.data, N);
    auto out = fmt::vformat_to(it, fmt, fmt::make_format_args(args...));
    dst.length = std::min(size_t(out - it), N - 1);
    dst.data[dst.length] = 0;
    return dst.c_str();
}

template <typename String, typename... Args>
String format(char const* fmt, Args&&... args) {
    String result;
    format(result, fmt, std::forward<Args>(args)...);
    return result;
}

template <typename... Args>
exception except(char const* fmt, Args&&... args) {
    return exception(format<fixed_string<128>>(fmt, std::forward<Args>(args)...));
}

//
// Base64

inline std::string base64_encode(std::span<std::byte const> data) {
    static char const alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    size_t i = 0;
    while (i + 3 <= data.size()) {
        uint32_t v = (uint32_t(uint8_t(data[i])) << 16) | (uint32_t(uint8_t(data[i + 1])) << 8) |
            uint32_t(uint8_t(data[i + 2]));
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += alphabet[(v >> 6) & 63];
        out += alphabet[v & 63];
        i += 3;
    }
    if (i < data.size()) {
        uint32_t v = uint32_t(uint8_t(data[i])) << 16;
        bool two = i + 1 < data.size();
        if (two) {
            v |= uint32_t(uint8_t(data[i + 1])) << 8;
        }
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += two ? alphabet[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

inline std::vector<std::byte> base64_decode(std::string_view in) {
    auto value = [](char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        throw except("Invalid base64 character: {}", c);
    };
    std::string_view s = in;
    size_t pad = 0;
    while (!s.empty() && (s.back() == '=' || s.back() == '\n' || s.back() == '\r')) {
        if (s.back() == '=') {
            ++pad;
        }
        s.remove_suffix(1);
    }
    // A valid stream is a multiple of four including padding; the stripped
    // tail (2 or 3 chars) is the final partial group.
    if (in.size() % 4 != 0) {
        throw except("Invalid base64 length: {}", in.size());
    }
    if (pad > 2) {
        throw except("Too much base64 padding");
    }
    std::vector<std::byte> out;
    out.reserve(s.size() / 4 * 3);
    size_t i = 0;
    while (i < s.size()) {
        size_t chunk = std::min(size_t(4), s.size() - i);
        uint32_t v = 0;
        for (size_t j = 0; j < chunk; ++j) {
            v = (v << 6) | uint32_t(value(s[i + j]));
        }
        v <<= (4 - chunk) * 6; // align a short final group to the top
        size_t nbytes = (chunk * 3) / 4;
        for (size_t b = 0; b < nbytes; ++b) {
            out.push_back(std::byte((v >> (16 - 8 * b)) & 0xFF));
        }
        i += chunk;
    }
    return out;
}

inline void assertion_failure(char const* file, int line, char const* expr) {
    auto msg = format<fixed_string<256>>("Assertion failed at {}:{}: {}\n", file, line, expr);
    fwrite(msg.data, 1, msg.length, stderr);

#if defined(VISP_ASSERT_BREAK)
#    ifdef _MSC_VER
    __debugbreak();
#    else
    __builtin_trap();
#    endif
#elif defined(VISP_ASSERT_THROW)
    throw exception(msg.c_str());
#else
    std::abort();
#endif
}

} // namespace visp
