#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_impl(T *out, const T *in, const int64_t *pos_ids, 
               float theta, size_t seqlen, size_t nheads, size_t head_dim) {
    // in/out shape: [seqlen, nheads, head_dim]
    size_t half_dim = head_dim / 2;
    
    for (size_t s = 0; s < seqlen; ++s) {
        int64_t p = pos_ids[s];
        for (size_t h = 0; h < nheads; ++h) {
            const T* src_head = in + (s * nheads + h) * head_dim;
            T* dst_head = out + (s * nheads + h) * head_dim;

            for (size_t j = 0; j < half_dim; ++j) {
                float freq = static_cast<float>(p) / std::pow(theta, 2.0f * j / head_dim);
                float sin_val = std::sin(freq);
                float cos_val = std::cos(freq);

                float a = llaisys::utils::cast<float>(src_head[j]);
                float b = llaisys::utils::cast<float>(src_head[j + half_dim]);

                float a_out = a * cos_val - b * sin_val;
                float b_out = b * cos_val + a * sin_val;

                dst_head[j] = llaisys::utils::cast<T>(a_out);
                dst_head[j + half_dim] = llaisys::utils::cast<T>(b_out);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          float theta, llaisysDataType_t dtype, size_t seqlen, size_t nheads, size_t head_dim) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return rope_impl(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), 
                             reinterpret_cast<const int64_t*>(pos_ids), theta, seqlen, nheads, head_dim);
        case LLAISYS_DTYPE_F16:
            return rope_impl(reinterpret_cast<llaisys::fp16_t*>(out), reinterpret_cast<const llaisys::fp16_t*>(in), 
                             reinterpret_cast<const int64_t*>(pos_ids), theta, seqlen, nheads, head_dim);
        case LLAISYS_DTYPE_BF16:
            return rope_impl(reinterpret_cast<llaisys::bf16_t*>(out), reinterpret_cast<const llaisys::bf16_t*>(in), 
                             reinterpret_cast<const int64_t*>(pos_ids), theta, seqlen, nheads, head_dim);
        default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}