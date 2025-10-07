#pragma once

#include "visp/util.h"

#include <algorithm>
#include <cmath>

namespace visp {
using std::clamp;
// clang-format off

constexpr int32_t div_ceil(int32_t a, int32_t b) { return (a + b - 1) / b; }
constexpr int64_t div_ceil(int64_t a, int64_t b) { return (a + b - 1) / b; }

constexpr int32_t next_multiple(int32_t x, int32_t mult) { return div_ceil(x, mult) * mult; }

constexpr float sqr(float x) { return x * x; }
constexpr int sqr(int x) { return x * x; }

constexpr int log2(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;
        ++log;
    }
    return log;
}

// f32x4 operations

constexpr f32x4 operator-(f32x4 const& a) { return f32x4{-a[0], -a[1], -a[2], -a[3]}; }
constexpr f32x4 operator+(f32x4 const& a, f32x4 const& b) { return f32x4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}; }
constexpr f32x4 operator-(f32x4 const& a, f32x4 const& b) { return f32x4{a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]}; }
constexpr f32x4 operator*(f32x4 const& a, f32x4 const& b) { return f32x4{a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]}; }
constexpr f32x4 operator/(f32x4 const& a, f32x4 const& b) { return f32x4{a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]}; }
constexpr f32x4 operator*(f32x4 const& a, float b) { return f32x4{a[0] * b, a[1] * b, a[2] * b, a[3] * b}; }
constexpr f32x4 operator*(float b, f32x4 const& a) { return f32x4{b * a[0], b * a[1], b * a[2], b * a[3]}; }
constexpr f32x4 operator/(f32x4 const& a, float b) { return f32x4{a[0] / b, a[1] / b, a[2] / b, a[3] / b}; }
constexpr f32x4 operator/(float b, f32x4 const& a) { return f32x4{b / a[0], b / a[1], b / a[2], b / a[3]}; }

constexpr f32x4 clamp(f32x4 const& a, float min, float max) {
    return f32x4{clamp(a[0], min, max), clamp(a[1], min, max), clamp(a[2], min, max), clamp(a[3], min, max)};
}

constexpr float dot(f32x4 const& a, f32x4 const& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

// i32x2 operations

constexpr i32x2 operator+(i32x2 a, i32x2 b) { return {a[0] + b[0], a[1] + b[1]}; }
constexpr i32x2 operator-(i32x2 a, i32x2 b) { return {a[0] - b[0], a[1] - b[1]}; }
constexpr i32x2 operator*(i32x2 a, i32x2 b) { return {a[0] * b[0], a[1] * b[1]}; }
constexpr i32x2 operator/(i32x2 a, i32x2 b) { return {a[0] / b[0], a[1] / b[1]}; }
constexpr i32x2 operator*(i32x2 a, int32_t b) { return {a[0] * b, a[1] * b}; }
constexpr i32x2 operator/(i32x2 a, int32_t b) { return {a[0] / b, a[1] / b}; }

constexpr i32x2 div_ceil(i32x2 a, i32x2 b) { return {div_ceil(a[0], b[0]), div_ceil(a[1], b[1])}; }
constexpr i32x2 div_ceil(i32x2 a, int32_t b) { return div_ceil(a, i32x2{b, b}); }
constexpr i32x2 next_multiple(i32x2 x, int32_t mult) { return div_ceil(x, mult) * mult; }
constexpr i32x2 min(i32x2 a, i32x2 b) { return {std::min(a[0], b[0]), std::min(a[1], b[1])}; }

// i64x2 operations
constexpr i64x2 operator*(i64x2 a, int64_t b) { return {a[0] * b, a[1] * b}; }
constexpr i64x2 operator/(i64x2 a, int64_t b) { return {a[0] / b, a[1] / b}; }

// clang-format on
} // namespace visp