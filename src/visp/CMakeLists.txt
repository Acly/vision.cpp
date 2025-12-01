add_library(visioncpp)

target_sources(visioncpp PRIVATE
  arch/birefnet.cpp
  arch/depth-anything.cpp
  arch/dino.cpp
  arch/esrgan.cpp
  arch/migan.cpp
  arch/mobile-sam.cpp
  arch/swin.cpp
  image.cpp
  ml.cpp
  nn.cpp
  platform.cpp
  vision.cpp
)
target_compile_features(visioncpp PUBLIC cxx_std_20)
target_compile_definitions(visioncpp PRIVATE VISP_API_EXPORT ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(visioncpp PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_include_directories(visioncpp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE ..
)
target_link_options(visioncpp PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(visioncpp
  PUBLIC ggml
  PRIVATE stb ${VISP_FMT_LINK}
)
# set_target_properties(visioncpp PROPERTIES
#   VERSION ${PROJECT_VERSION}
#   SOVERSION ${PROJECT_VERSION_MAJOR}
# )

if (MSVC AND VISP_TESTS)
  set_target_properties(visioncpp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
