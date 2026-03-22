#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_impl(T *out, const T *in, const T *weight, float eps, size_t rows, size_t dim) {
    for (size_t i = 0; i < rows; ++i) {
        float sum_sq = 0.0f;
        const T* row_in = in + i * dim;
        T* row_out = out + i * dim;

        for (size_t j = 0; j < dim; ++j) {
            float x = llaisys::utils::cast<float>(row_in[j]);
            sum_sq += x * x;
        }

        float rms = std::sqrt(sum_sq / dim + eps);
        float inv_rms = 1.0f / rms;

        for (size_t j = 0; j < dim; ++j) {
            float x = llaisys::utils::cast<float>(row_in[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            row_out[j] = llaisys::utils::cast<T>(x * inv_rms * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              float eps, llaisysDataType_t dtype, size_t rows, size_t dim) {
    switch(dtype) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_impl(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in),
                                 reinterpret_cast<const float*>(weight), eps, rows, dim);
        case LLAISYS_DTYPE_F16:
            return rms_norm_impl(reinterpret_cast<llaisys::fp16_t*>(out), reinterpret_cast<const llaisys::fp16_t*>(in),
                                 reinterpret_cast<const llaisys::fp16_t*>(weight), eps, rows, dim);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_impl(reinterpret_cast<llaisys::bf16_t*>(out), reinterpret_cast<const llaisys::bf16_t*>(in),
                                 reinterpret_cast<const llaisys::bf16_t*>(weight), eps, rows, dim);
        default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}