if (NOT GGML_DMABUF OR NOT GGML_DMABUF_HAVE_DMA_BUF OR NOT GGML_DMABUF_HAVE_DMA_HEAP)
    message(FATAL_ERROR "Fake NPU requires GGML_DMABUF=ON")
endif()

ggml_add_backend_library(ggml-fakenpu
                         ggml-fakenpu.cpp
                        )
