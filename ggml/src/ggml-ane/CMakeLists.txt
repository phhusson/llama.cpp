if (NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" OR NOT GGML_METAL_TARGET_OS STREQUAL "macos")
    message(FATAL_ERROR "GGML_ANE requires native Apple Silicon macOS")
endif()

find_library(ANE_FOUNDATION Foundation REQUIRED)
find_library(ANE_METAL Metal REQUIRED)
find_library(ANE_IOSURFACE IOSurface REQUIRED)

ggml_add_backend_library(ggml-ane ggml-ane.m)
target_link_libraries(ggml-ane PRIVATE ${ANE_FOUNDATION} ${ANE_METAL} ${ANE_IOSURFACE})
