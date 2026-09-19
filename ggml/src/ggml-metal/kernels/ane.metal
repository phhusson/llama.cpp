#include "common.h"

kernel void kernel_ane_cast(
        device const float4 * src [[buffer(0)]],
        device       half4  * dst [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
    dst[i] = half4(src[i]);
}

kernel void kernel_ane_join(
        device const half4  * src   [[buffer(0)]],
        device       float4 * dst   [[buffer(1)]],
        constant     uint2  & shape [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    uint row = i % (shape.y/4);
    uint token = i / (shape.y/4);
    dst[token*(shape.x/4) + row] = float4(src[i])*(1.0f/32.0f);
}

kernel void kernel_pq2_planar_expand(
        device const uchar * q     [[buffer(0)]],
        device       half4 * w     [[buffer(1)]],
        device const half  * d     [[buffer(2)]],
        constant     uint2 & shape [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
    uint row = i / (shape.x/4);
    uint col = i % (shape.x/4);
    uint block = col/32;
    uint j = (col%32)*4;
    uint at = row*(shape.x/4) + block*32 + j%32;
    uint shift = 2*(j/32);
    w[i] = (half4((q[at]>>shift)&3, (q[at+1]>>shift)&3, (q[at+2]>>shift)&3, (q[at+3]>>shift)&3) - half4(1))*d[row*shape.y + block];
}
