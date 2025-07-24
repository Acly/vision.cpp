#include "testing.hpp"
#include "visp/ml.hpp"

#include <chrono>
#include <filesystem>
#include <string_view>

using std::chrono::steady_clock;

namespace visp {
// Globals
float tolerance = 1e-5f;
std::string extra_info;
} // namespace visp

int main(int argc, char** argv) {
    using namespace visp;

    auto& registry = test_registry_instance();

    int passed = 0;
    int failed = 0;
    int errors = 0;

    std::string_view filter;
    bool exclude_gpu = false;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--no-gpu") {
            exclude_gpu = true;
        } else {
            filter = arg;
        }
    }

    auto run = [&](test_case const& test, char const* name, backend_type backend) {
        try {
            if (!filter.empty() && name != filter && test.name != filter) {
                return; // test not selected
            }
            if (verbose) {
                printf("%s", name);
                fflush(stdout);
            }

            if (test.is_backend_test) {
                test.backend_func(backend);
            } else {
                test.func();
            }

            ++passed;
            if (verbose) {
                printf(" %s\n", "\033[32mPASSED\033[0m");
            }
        } catch (const visp::test_failure& e) {
            ++failed;
            printf(" %s\n", "\033[31mFAILED\033[0m");
            printf("  \033[90m%s:%d:\033[0m Assertion failed\n", e.file, e.line);
            printf("  \033[93m%s\033[0m\n", e.condition);
            if (e.eval) {
                printf("  \033[93m%s\033[0m\n", e.eval.c_str());
            }
            if (!visp::extra_info.empty()) {
                printf("  %s\n", visp::extra_info.c_str());
            }
        } catch (const std::exception& e) {
            ++errors;
            printf(" %s\n", "\033[31mERROR\033[0m");
            printf("  \033[90m%s:%d:\033[0m Unhandled exception\n", test.file, test.line);
            printf("  \033[93m%s\033[0m\n", e.what());
        }
        visp::extra_info.clear();
    };

    auto time_start = steady_clock::now();
    fixed_string<128> name;

    for (auto& test : registry.tests) {
        if (test.is_backend_test) {
            run(test, format(name, "{}[cpu]", test.name), backend_type::cpu);
            if (!exclude_gpu) {
                run(test, format(name, "{}[gpu]", test.name), backend_type::gpu);
            }
        } else {
            run(test, test.name, backend_type::cpu);
        }
    }

    auto time_end = steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    char const* color = (failed > 0 || errors > 0) ? "\033[31m" : "\033[32m";
    if (verbose || failed > 0 || errors > 0) {
        printf("%s----------------------------------------------------------------------\n", color);
    }
    if (failed > 0) {
        printf("\033[31m%d failed, ", failed);
    }
    if (errors > 0) {
        printf("\033[31m%d errors, ", errors);
    }
    printf("\033[92m%d passed %sin %lldms\033[0m\n", passed, color, (long long)duration);

    return (failed > 0 || errors > 0) ? 1 : 0;
}

namespace visp {

test_registry& test_registry_instance() {
    static test_registry registry;
    return registry;
}

test_registration::test_registration(
    char const* name, test_function f, char const* file, int line) {
    test_case t;
    t.name = name;
    t.file = file;
    t.line = line;
    t.func = f;
    t.is_backend_test = false;
    test_registry_instance().tests.push_back(t);
}

test_registration::test_registration(
    char const* name, test_backend_function f, char const* file, int line) {
    test_case t;
    t.name = name;
    t.file = file;
    t.line = line;
    t.backend_func = f;
    t.is_backend_test = true;
    test_registry_instance().tests.push_back(t);
}

test_directories const& test_dir() {
    static test_directories const result = []() {
        path cur = std::filesystem::current_path();
        while (!exists(cur / "README.md")) {
            cur = cur.parent_path();
            if (cur.empty()) {
                throw std::runtime_error("root directory not found");
            }
        }
        test_directories dirs{
            .root = cur,
            .models = cur / "models",
            .test = cur / "tests",
            .input = cur / "tests" / "input",
            .results = cur / "tests" / "results",
            .reference = cur / "tests" / "reference"};
        if (!exists(dirs.results)) {
            create_directories(dirs.results);
        }
        return dirs;
    }();
    return result;
}

void test_set_info(std::string_view info) {
    extra_info = info;
}

float& test_tolerance_value() {
    return tolerance;
}

test_failure test_failure_image_mismatch(
    char const* file, int line, char const* condition, float rms) {
    test_failure result(file, line, condition);
    format(result.eval, "-> rmse {:.5f} > {:.5f} tolerance", rms, test_tolerance_value());
    return result;
}

} // namespace visp
