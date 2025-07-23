cmake_minimum_required(VERSION 3.22)

project(vision.cpp VERSION 0.1.0 LANGUAGES CXX)

option(VISP_VULKAN "Enable Vulkan support" OFF)
option(VISP_DEV "Enable development mode" OFF)
option(VISP_CI "Enable for continuous integration environment" OFF)
option(VISP_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(VISP_FMT_LIB "Use external fmt library instead of standard C++ <format>" OFF)
option(VISP_ASAN "Enable AddressSanitizer" OFF)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

if(VISP_DEV)
  set(VISP_ASSERT "VISP_ASSERT_BREAK")
elseif(VISP_CI)
  set(VISP_ASSERT "VISP_ASSERT_THROW")
elseif(CMAKE_BUILD_TYPE)
  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(VISP_ASSERT "VISP_ASSERT_BREAK")
  elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(VISP_ASSERT "VISP_ASSERT_DISABLE")
  endif()
endif()

if(VISP_ASAN)
  if(MSVC)
    add_compile_options(/fsanitize=address)
    add_link_options(clang_rt.asan_dynamic-x86_64.lib clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
  else()
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  endif()
endif()

if(MSVC)  
  add_compile_options(/Zi /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  add_link_options(/DEBUG) # Enable debug symbols also in release builds
endif()

if(VISP_DEV OR VISP_CI)
  if(MSVC)
    set(VISP_WARNINGS /W4 /WX /wd4251)
  else()
    set(VISP_WARNINGS -Wall -Wextra -Wpedantic -Werror)
  endif()
endif()

# Dependencies

add_subdirectory(depend/stb)

if(VISP_FMT_LIB)
  add_subdirectory(depend/fmt)
  set(VISP_FMT_LINK fmt::fmt)
  set(VISP_FMT_DEFS VISP_FMT_LIB)
endif()

set(GGML_VULKAN ${VISP_VULKAN})
set(GGML_LLAMAFILE ON)
if(VISP_CI)
  set(GGML_BACKEND_DL ON)
  if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(aarch64|arm.*|ARM64)$")
    # set default for ARM
  else()
    set(GGML_NATIVE OFF)
    foreach (feat SSE42 AVX AVX2 F16C BMI2 FMA) # ~haswell and newer
      set(GGML_${feat} ON)
    endforeach()
  endif()
endif()
add_subdirectory(depend/ggml)

# Vision.cpp libraries and executables

add_subdirectory(src/visp)
add_subdirectory(src/cli)

# Tests

if(VISP_TESTS)
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(models)
endif()

# Installation and packaging

install(TARGETS visioncpp
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
if(PROJECT_IS_TOP_LEVEL)
  install(FILES README.md LICENSE DESTINATION .)
endif()

include(CMakePackageConfigHelpers)

set(VISP_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header files")
set(VISP_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Location of library files")
set(VISP_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Location of binary files")

configure_package_config_file(
  scripts/cmake/visioncpp-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/visioncpp-config.cmake
  INSTALL_DESTINATION lib/cmake/visioncpp
  PATH_VARS VISP_INCLUDE_INSTALL_DIR VISP_LIB_INSTALL_DIR VISP_BIN_INSTALL_DIR
)
write_basic_package_version_file(
  visioncpp-version.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/visioncpp-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/visioncpp-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/visioncpp
)

# if(WIN32)
#   set(CPACK_GENERATOR "ZIP")
#   set(CPACK_PACKAGE_FILE_NAME visioncpp-windows-x64-${PROJECT_VERSION})
# else()
#   set(CPACK_GENERATOR "TGZ")
#   set(CPACK_PACKAGE_FILE_NAME visioncpp-linux-x64-${PROJECT_VERSION})
# endif()
# set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
# include(CPack)
