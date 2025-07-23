#pragma once

#include "util/string.hpp"

#include <filesystem>
#include <vector>

namespace visp {
enum class backend_type;

struct test_failure {
    char const* file;
    int line;
    char const* condition;
    fixed_string<64> eval;

    test_failure(char const* file, int line, char const* condition)
        : file(file), line(line), condition(condition) {}
};

using test_function = void (*)();
using test_backend_function = void (*)(backend_type);

struct test_case {
    char const* name;
    char const* file;
    int line;
    bool is_backend_test;
    union {
        test_function func;
        test_backend_function backend_func;
    };
};

struct test_registry {
    std::vector<test_case> tests;
};

test_registry& test_registry_instance();

struct test_registration {
    test_registration(char const* name, test_function f, char const* file, int line);
    test_registration(char const* name, test_backend_function f, char const* file, int line);
};

using std::filesystem::path;

struct test_directories {
    path root;
    path models;
    path test;
    path input;
    path results;
    path reference;
};

test_directories const& test_dir();

float& test_tolerance_value();

struct test_with_tolerance {
    float old_value;

    test_with_tolerance(float epsilon) : old_value(test_tolerance_value()) {
        test_tolerance_value() = epsilon;
    }
    ~test_with_tolerance() { test_tolerance_value() = old_value; }
};

void test_set_info(std::string_view);

template <typename T>
bool test_is_equal(T const& a, T const& b) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) <= test_tolerance_value();
    } else {
        return a == b;
    }
}

template <typename LHS, typename RHS>
test_failure test_failure_not_equal(
    char const* file, int line, char const* condition, LHS const& lhs, RHS const& rhs) {

    test_failure result(file, line, condition);
    if constexpr (std::is_floating_point_v<LHS> && std::is_floating_point_v<RHS>) {
        format(result.eval, "-> {} != {} \033[90m(eps={})", lhs, rhs, test_tolerance_value());
    } else {
        format(result.eval, "-> {} != {}", lhs, rhs);
    }
    return result;
}

test_failure test_failure_image_mismatch(char const* file, int line, char const*, float rms);

} // namespace visp

#define VISP_TEST(name)                                                                            \
    void test_func_##name();                                                                       \
    const visp::test_registration test_reg_##name(#name, test_func_##name, __FILE__, __LINE__);    \
    void test_func_##name()

#define VISP_BACKEND_TEST(name)                                                                    \
    void test_func_##name(visp::backend_type);                                                     \
    const visp::test_registration test_reg_##name(#name, test_func_##name, __FILE__, __LINE__);    \
    void test_func_##name

#define CHECK(...)                                                                                 \
    if (!(__VA_ARGS__)) {                                                                          \
        throw visp::test_failure(__FILE__, __LINE__, #__VA_ARGS__);                                \
    }

#define CHECK_EQUAL(lhs, rhs)                                                                      \
    if (!test_is_equal(lhs, rhs)) {                                                                \
        throw visp::test_failure_not_equal(__FILE__, __LINE__, #lhs " == " #rhs, lhs, rhs);        \
    }

#define CHECK_IMAGES_EQUAL(lhs, rhs)                                                               \
    if (float rms = visp::image_difference_rms(lhs, rhs); rms > visp::test_tolerance_value()) {    \
        throw visp::test_failure_image_mismatch(__FILE__, __LINE__, #lhs " == " #rhs, rms);        \
    }
