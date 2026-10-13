// SPDX-License-Identifier: MIT
// Register sequence derived from Mesa Rocket (Copyright 2024 Tomeu Vizoso).
#include "rocket-regs.h"
#include <algorithm>
#include <cassert>

std::vector<uint64_t> rocket_matmul_regs(uint32_t h, uint32_t k, uint32_t n, uint32_t stride,
                                       uint32_t input, uint32_t weights, uint32_t output, uint32_t bias) {
    assert(h >= 4 && h % 4 == 0 && k % 32 == 0 && n % 16 == 0);
    const uint32_t channels = n;
    const uint32_t data_banks = (h*k*2 + 32767)/32768;
    assert(data_banks > 0 && data_banks < 12);
    const uint32_t banks = ((12 - data_banks) << 4) | data_banks;
    std::vector<uint64_t> regs;
    regs.reserve(130);
    auto emit = [&](uint32_t target, uint32_t reg, uint32_t value) {
        regs.push_back((uint64_t(target) << 48) | (uint64_t(value) << 16) | reg);
    };
    emit(0x0201, 0x1040, banks); // CNA_CBUF_CON0
    emit(0x0201, 0x1104, 0x00000000); // CNA_DCOMP_REGNUM
    emit(0x0201, 0x1100, 0x00000000); // CNA_DCOMP_CTRL
    emit(0x0201, 0x100c, 0x000001b0); // CNA_CONV_CON1
    emit(0x1001, 0x4004, 0x0000000e); // DPU_S_POINTER
    emit(0x2001, 0x5004, 0x0000000e); // DPU_RDMA_RDMA_S_POINTER
    emit(0x0201, 0x100c, 0x000001b0); // CNA_CONV_CON1
    emit(0x0201, 0x1010, h << 4); // CNA_CONV_CON2: keep the whole feature tile resident
    emit(0x0201, 0x1014, 0x00000009); // CNA_CONV_CON3
    emit(0x0201, 0x1020, 0x10000 | h); // CNA_DATA_SIZE0
    emit(0x0201, 0x1024, ((k - 1) << 16) | k); // CNA_DATA_SIZE1
    emit(0x0201, 0x1028, 0x00000001); // CNA_DATA_SIZE2
    emit(0x0201, 0x102c, h); // CNA_DATA_SIZE3
    emit(0x0201, 0x1030, k*n*2); // CNA_WEIGHT_SIZE0
    emit(0x0201, 0x1034, k*2); // CNA_WEIGHT_SIZE1
    emit(0x0201, 0x1038, 0x1010000 | n); // CNA_WEIGHT_SIZE2
    emit(0x0201, 0x1040, banks); // CNA_CBUF_CON0
    emit(0x0201, 0x1044, k/32); // CNA_CBUF_CON1
    emit(0x0201, 0x104c, 0x00000001); // CNA_CVT_CON0
    emit(0x0201, 0x1050, 0x00000001); // CNA_CVT_CON1
    emit(0x0201, 0x1054, 0x00000001); // CNA_CVT_CON2
    emit(0x0201, 0x1058, 0x00000001); // CNA_CVT_CON3
    emit(0x0201, 0x105c, 0x00000001); // CNA_CVT_CON4
    emit(0x0201, 0x1060, 0x00000000); // CNA_FC_CON0
    emit(0x0201, 0x1064, 0x00000000); // CNA_FC_CON1
    emit(0x0201, 0x1068, 0x00000000); // CNA_PAD_CON0
    emit(0x0201, 0x1070, input); // CNA_FEATURE_DATA_ADDR
    emit(0x0201, 0x1074, 0x00000000); // CNA_FC_CON2
    emit(0x0201, 0x1078, 0x000f000f); // CNA_DMA_CON0
    emit(0x0201, 0x107c, 0x00000004); // CNA_DMA_CON1
    emit(0x0201, 0x1080, h - 4); // CNA_DMA_CON2
    emit(0x0201, 0x1084, 0x10000 | h); // CNA_FC_DATA_SIZE0
    emit(0x0201, 0x1088, k); // CNA_FC_DATA_SIZE1
    emit(0x0201, 0x1100, 0x00000000); // CNA_DCOMP_CTRL
    emit(0x0201, 0x1104, 0x00000000); // CNA_DCOMP_REGNUM
    emit(0x0201, 0x1110, weights); // CNA_DCOMP_ADDR0
    emit(0x0201, 0x1140, 0x00000000); // CNA_DCOMP_AMOUNT0
    emit(0x0201, 0x1144, 0x00000000); // CNA_DCOMP_AMOUNT1
    emit(0x0201, 0x1148, 0x00000000); // CNA_DCOMP_AMOUNT2
    emit(0x0201, 0x114c, 0x00000000); // CNA_DCOMP_AMOUNT3
    emit(0x0201, 0x1150, 0x00000000); // CNA_DCOMP_AMOUNT4
    emit(0x0201, 0x1154, 0x00000000); // CNA_DCOMP_AMOUNT5
    emit(0x0201, 0x1158, 0x00000000); // CNA_DCOMP_AMOUNT6
    emit(0x0201, 0x115c, 0x00000000); // CNA_DCOMP_AMOUNT7
    emit(0x0201, 0x1160, 0x00000000); // CNA_DCOMP_AMOUNT8
    emit(0x0201, 0x1164, 0x00000000); // CNA_DCOMP_AMOUNT9
    emit(0x0201, 0x1168, 0x00000000); // CNA_DCOMP_AMOUNT10
    emit(0x0201, 0x116c, 0x00000000); // CNA_DCOMP_AMOUNT11
    emit(0x0201, 0x1170, 0x00000000); // CNA_DCOMP_AMOUNT12
    emit(0x0201, 0x1174, 0x00000000); // CNA_DCOMP_AMOUNT13
    emit(0x0201, 0x1178, 0x00000000); // CNA_DCOMP_AMOUNT14
    emit(0x0201, 0x117c, 0x00000000); // CNA_DCOMP_AMOUNT15
    emit(0x0201, 0x1180, 0x00000000); // CNA_CVT_CON5
    emit(0x0201, 0x1184, 0x00000000); // CNA_PAD_CON1
    emit(0x0801, 0x3010, 0x00000301); // CORE_MISC_CFG
    emit(0x0801, 0x3014, (h - 1) << 16); // CORE_DATAOUT_SIZE_0
    emit(0x0801, 0x3018, channels - 1); // CORE_DATAOUT_SIZE_1
    emit(0x0801, 0x301c, 0x00000000); // CORE_CLIP_TRUNCATE
    emit(0x0801, 0x3030, 0x00000000); // reserved
    emit(0x1001, 0x400c, 0x000001e4); // DPU_FEATURE_MODE_CFG
    emit(0x1001, 0x4010, 0x6c000003); // DPU_DATA_FORMAT
    emit(0x1001, 0x4014, 0x00000000); // DPU_OFFSET_PEND
    emit(0x1001, 0x4020, output); // DPU_DST_BASE_ADDR
    emit(0x1001, 0x4024, stride*16); // DPU_DST_SURF_STRIDE
    emit(0x1001, 0x4030, 0x00000000); // DPU_DATA_CUBE_WIDTH
    emit(0x1001, 0x4034, h - 1); // DPU_DATA_CUBE_HEIGHT
    emit(0x1001, 0x4038, 0x00000000); // DPU_DATA_CUBE_NOTCH_ADDR
    emit(0x1001, 0x403c, ((n - 1) << 16) | (channels - 1)); // DPU_DATA_CUBE_CHANNEL
    emit(0x1001, 0x4040, 0x00020150); // DPU_BS_CFG
    emit(0x1001, 0x4044, 0x00000000); // DPU_BS_ALU_CFG
    emit(0x1001, 0x4048, 0x00000000); // DPU_BS_MUL_CFG
    emit(0x1001, 0x404c, 0x00000000); // DPU_BS_RELUX_CMP_VALUE
    emit(0x1001, 0x4050, 0x00000124); // DPU_BS_OW_CFG
    emit(0x1001, 0x4054, 0x00000000); // DPU_BS_OW_OP
    emit(0x1001, 0x4058, 0x08000000 | (channels - 1)); // DPU_WDMA_SIZE_0
    emit(0x1001, 0x405c, (h - 1) << 16); // DPU_WDMA_SIZE_1
    emit(0x1001, 0x4060, 0x00000053); // DPU_BN_CFG
    emit(0x1001, 0x4064, 0x00000000); // DPU_BN_ALU_CFG
    emit(0x1001, 0x4068, 0x00000000); // DPU_BN_MUL_CFG
    emit(0x1001, 0x406c, 0x00000000); // DPU_BN_RELUX_CMP_VALUE
    emit(0x1001, 0x4070, 0x00000383); // DPU_EW_CFG
    emit(0x1001, 0x4074, 0x00000000); // DPU_EW_CVT_OFFSET_VALUE
    emit(0x1001, 0x4078, 0x00000001); // DPU_EW_CVT_SCALE_VALUE
    emit(0x1001, 0x407c, 0x00000000); // DPU_EW_RELUX_CMP_VALUE
    emit(0x1001, 0x4080, 0x00000000); // DPU_OUT_CVT_OFFSET
    emit(0x1001, 0x4084, 0x00010001); // DPU_OUT_CVT_SCALE
    emit(0x1001, 0x4088, 0x00000000); // DPU_OUT_CVT_SHIFT
    emit(0x1001, 0x4090, 0x00000000); // DPU_EW_OP_VALUE_0
    emit(0x1001, 0x4094, 0x00000000); // DPU_EW_OP_VALUE_1
    emit(0x1001, 0x4098, 0x00000000); // DPU_EW_OP_VALUE_2
    emit(0x1001, 0x409c, 0x00000000); // DPU_EW_OP_VALUE_3
    emit(0x1001, 0x40a0, 0x00000000); // DPU_EW_OP_VALUE_4
    emit(0x1001, 0x40a4, 0x00000000); // DPU_EW_OP_VALUE_5
    emit(0x1001, 0x40a8, 0x00000000); // DPU_EW_OP_VALUE_6
    emit(0x1001, 0x40ac, 0x00000000); // DPU_EW_OP_VALUE_7
    emit(0x1001, 0x40c0, stride*32); // DPU_SURFACE_ADD
    emit(0x1001, 0x40c4, 0x00000000); // reserved
    emit(0x1001, 0x4100, 0x00000000); // DPU_LUT_ACCESS_CFG
    emit(0x1001, 0x4104, 0x00000000); // DPU_LUT_ACCESS_DATA
    emit(0x1001, 0x4108, 0x00000000); // DPU_LUT_CFG
    emit(0x1001, 0x410c, 0x00000000); // DPU_LUT_INFO
    emit(0x1001, 0x4110, 0x00000000); // DPU_LUT_LE_START
    emit(0x1001, 0x4114, 0x00000000); // DPU_LUT_LE_END
    emit(0x1001, 0x4118, 0x00000000); // DPU_LUT_LO_START
    emit(0x1001, 0x411c, 0x00000000); // DPU_LUT_LO_END
    emit(0x1001, 0x4120, 0x00000000); // DPU_LUT_LE_SLOPE_SCALE
    emit(0x1001, 0x4124, 0x00000000); // DPU_LUT_LE_SLOPE_SHIFT
    emit(0x1001, 0x4128, 0x00000000); // DPU_LUT_LO_SLOPE_SCALE
    emit(0x1001, 0x412c, 0x00000000); // DPU_LUT_LO_SLOPE_SHIFT
    emit(0x2001, 0x500c, 0x00000000); // DPU_RDMA_RDMA_DATA_CUBE_WIDTH
    emit(0x2001, 0x5010, h - 1); // DPU_RDMA_RDMA_DATA_CUBE_HEIGHT
    emit(0x2001, 0x5014, channels - 1); // DPU_RDMA_RDMA_DATA_CUBE_CHANNEL
    emit(0x2001, 0x5018, 0x00000000); // DPU_RDMA_RDMA_SRC_BASE_ADDR
    emit(0x2001, 0x501c, 0x00000002); // DPU_RDMA_RDMA_BRDMA_CFG
    emit(0x2001, 0x5020, bias); // DPU_RDMA_RDMA_BS_BASE_ADDR
    emit(0x2001, 0x5028, 0x00000000); // DPU_RDMA_RDMA_NRDMA_CFG
    emit(0x2001, 0x502c, 0x00000000); // DPU_RDMA_RDMA_BN_BASE_ADDR
    emit(0x2001, 0x5034, 0x00000001); // DPU_RDMA_RDMA_ERDMA_CFG
    emit(0x2001, 0x5038, 0x00000000); // DPU_RDMA_RDMA_EW_BASE_ADDR
    emit(0x2001, 0x5040, 0x00000000); // DPU_RDMA_RDMA_EW_SURF_STRIDE
    emit(0x2001, 0x5044, 0x0001f878); // DPU_RDMA_RDMA_FEATURE_MODE_CFG
    emit(0x2001, 0x5048, 0x00000000); // DPU_RDMA_RDMA_SRC_DMA_CFG
    emit(0x2001, 0x504c, 0x00000000); // DPU_RDMA_RDMA_SURF_NOTCH
    emit(0x2001, 0x5064, 0x00000000); // DPU_RDMA_RDMA_PAD_CFG
    emit(0x2001, 0x5068, 0x01010101); // DPU_RDMA_RDMA_WEIGHT
    emit(0x2001, 0x506c, 0x00000000); // DPU_RDMA_RDMA_EW_SURF_NOTCH
    regs.push_back(0x0000000000000000ULL);
    regs.push_back(0x0101000000000014ULL);
    regs.push_back(0x0041000000000000ULL);
    regs.push_back(0x00810000001d0008ULL);
    return regs;
}
