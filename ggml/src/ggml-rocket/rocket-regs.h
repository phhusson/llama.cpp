#pragma once

#include <cstdint>
#include <vector>

std::vector<uint64_t> rocket_matmul_regs(uint32_t h, uint32_t k, uint32_t n, uint32_t stride,
                                       uint32_t input, uint32_t weights, uint32_t output, uint32_t bias);
