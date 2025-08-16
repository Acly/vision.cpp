# Main library C++ tests

add_executable(test-vision)
target_sources(test-vision PRIVATE
  testing.cpp
  test-image.cpp
  test-ml.cpp
)
target_include_directories(test-vision PRIVATE . ../src)
target_compile_definitions(test-vision PRIVATE ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(test-vision PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_link_options(test-vision PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(test-vision PRIVATE visioncpp ${VISP_FMT_LINK})
add_test(NAME vision COMMAND test-vision -v)

#
# Model tests (image comparisons)

add_executable(test-models)
target_sources(test-models PRIVATE
  testing.cpp
  test-models.cpp
)
target_include_directories(test-models PRIVATE . ../src)
target_compile_definitions(test-models PRIVATE ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(test-models PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_link_options(test-models PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(test-models PRIVATE visioncpp ${VISP_FMT_LINK})
if(VISP_VULKAN AND NOT VISP_CI)
  add_test(NAME models COMMAND test-models -v)
else()
  # GPU tests currently don't pass on GitHub runners with Vulkan/llvmpipe
  add_test(NAME models COMMAND test-models -v --no-gpu)
endif()

include(reference-images.cmake)

#
# Workbench library for Python tests

add_library(vision-workbench workbench.cpp)
target_include_directories(vision-workbench PRIVATE ../src)
target_compile_definitions(vision-workbench PRIVATE ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(vision-workbench PRIVATE ${VISP_COMP_OPTIONS})
target_link_options(vision-workbench PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(vision-workbench PRIVATE visioncpp ggml ${VISP_FMT_LINK})

#
# Benchmarks

add_executable(vision-bench)
target_sources(vision-bench PRIVATE benchmark.cpp testing.cpp)
target_include_directories(vision-bench PRIVATE . ../src)
target_compile_definitions(vision-bench PRIVATE VISP_TEST_NO_MAIN ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(vision-bench PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_link_options(vision-bench PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(vision-bench PRIVATE visioncpp ${VISP_FMT_LINK})
