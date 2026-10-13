if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT GGML_DMABUF OR NOT GGML_VULKAN)
    message(FATAL_ERROR "Rocket requires Linux, GGML_DMABUF=ON and GGML_VULKAN=ON")
endif()

find_path(ROCKET_DRM_INCLUDE_DIR drm.h PATH_SUFFIXES libdrm drm REQUIRED)
ggml_add_backend_library(ggml-rocket ggml-rocket.cpp rocket-regs.cpp)
target_include_directories(ggml-rocket PRIVATE ${ROCKET_DRM_INCLUDE_DIR})
target_link_libraries(ggml-rocket PRIVATE ggml-vulkan)
