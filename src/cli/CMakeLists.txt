add_executable(vision-cli)
target_sources(vision-cli PRIVATE cli.cpp)
target_include_directories(vision-cli PRIVATE ..)
target_compile_definitions(vision-cli PRIVATE
  ${VISP_ASSERT}
  ${VISP_DEFINITIONS}
  VISP_MODEL_INSTALL_DIR="${VISP_MODEL_INSTALL_DIR}")
target_compile_options(vision-cli PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_link_options(vision-cli PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(vision-cli PRIVATE ggml visioncpp ${VISP_FMT_LINK})