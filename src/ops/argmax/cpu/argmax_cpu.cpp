#include "argmax_cpu.hpp"
#include "../../../utils.hpp" // 确保能找到 src/utils.hpp

template <typename T>
void argmax_impl(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) return;
    
    T curr_max = vals[0];
    int64_t curr_idx = 0;

    for (size_t i = 1; i < numel; ++i) {
        float v = llaisys::utils::cast<float>(vals[i]);
        float m = llaisys::utils::cast<float>(curr_max);
        if (v > m) {
            curr_max = vals[i];
            curr_idx = i;
        }
    }
    *max_idx = curr_idx;
    *max_val = curr_max;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), 
                           reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_impl(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), 
                           reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_impl(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), 
                           reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}