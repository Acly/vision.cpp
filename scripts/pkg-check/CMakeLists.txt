cmake_minimum_required(VERSION 3.28)

project(visioncpp_pkg_check LANGUAGES C CXX)

find_package(visioncpp CONFIG REQUIRED)

add_executable(pkg-check main.cpp)
target_link_libraries(pkg-check PRIVATE visioncpp)
set_target_properties(pkg-check PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:visioncpp>")
