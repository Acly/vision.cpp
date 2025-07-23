#pragma once

#include "visp/util.hpp"

#include <cstdio>
#include <utility>

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
