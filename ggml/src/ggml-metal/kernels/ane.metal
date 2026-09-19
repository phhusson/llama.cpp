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
