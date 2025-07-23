# Main library C++ tests

add_executable(test-vision)
target_sources(test-vision PRIVATE
  testing.cpp
  test-image.cpp
)
target_include_directories(test-vision PRIVATE . ../src)
target_compile_options(test-vision PRIVATE ${VISP_WARNINGS})
target_compile_definitions(test-vision PRIVATE ${VISP_ASSERT} ${VISP_FMT_DEFS})
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
target_compile_options(test-models PRIVATE ${VISP_WARNINGS})
target_compile_definitions(test-models PRIVATE ${VISP_ASSERT} ${VISP_FMT_DEFS})
target_link_libraries(test-models PRIVATE visioncpp ${VISP_FMT_LINK})
if(VISP_VULKAN)
  add_test(NAME models COMMAND test-models -v)
else()
  add_test(NAME models COMMAND test-models -v --no-gpu)
endif()

include(reference-images.cmake)

#
# Workbench library for Python tests

add_library(vision-workbench workbench.cpp)
target_include_directories(vision-workbench PRIVATE ../src)
target_compile_definitions(vision-workbench PRIVATE ${VISP_ASSERT} ${VISP_FMT_DEFS})
target_link_libraries(vision-workbench PRIVATE visioncpp ggml ${VISP_FMT_LINK})
