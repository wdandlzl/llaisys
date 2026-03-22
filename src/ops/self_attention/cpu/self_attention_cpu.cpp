#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

template <typename T>
void self_attention_impl(T *attn_val, const T *q, const T *k, const T *v, 
                         float scale, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t head_dim) {
    
    // q: [qlen, nhead, dim]
    // k, v: [kvlen, nkvhead, dim]
    // out: [qlen, nhead, dim]
    size_t group_size = nhead / nkvhead;

    // 预分配缓冲区，避免在循环内部频繁分配
    std::vector<float> scores(kvlen);
    std::vector<float> out_accumulator(head_dim); // 新增：用于 float 精度累加

    for (size_t i = 0; i < qlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_h = h / group_size;

            // 1. Q * K^T
            const T* q_vec = q + (i * nhead + h) * head_dim;
            
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < kvlen; ++j) {
                // Causal Mask logic: j > i + (kvlen - qlen)
                if (j > i + kvlen - qlen) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                } else {
                    const T* k_vec = k + (j * nkvhead + kv_h) * head_dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += llaisys::utils::cast<float>(q_vec[d]) * llaisys::utils::cast<float>(k_vec[d]);
                    }
                    scores[j] = dot * scale;
                }
                if (scores[j] > max_score) max_score = scores[j];
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            float inv_sum = 1.0f / sum_exp;

            // 3. Scores * V (使用 float 累加器)
            
            // 重置累加器为 0
            std::fill(out_accumulator.begin(), out_accumulator.end(), 0.0f);

            for (size_t j = 0; j < kvlen; ++j) {
                float weight = scores[j] * inv_sum;
                const T* v_vec = v + (j * nkvhead + kv_h) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    // 始终在 float 精度下累加
                    out_accumulator[d] += weight * llaisys::utils::cast<float>(v_vec[d]);
                }
            }

            // 4. 将最终的高精度结果转回 T 并写入输出
            T* out_vec = attn_val + (i * nhead + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                out_vec[d] = llaisys::utils::cast<T>(out_accumulator[d]);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
                    float scale, llaisysDataType_t dtype, 
                    size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t head_dim) {
    switch(dtype) {
        case LLAISYS_DTYPE_F32:
            return self_attention_impl(reinterpret_cast<float*>(attn_val), reinterpret_cast<const float*>(q),
                                       reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v),
                                       scale, qlen, kvlen, nhead, nkvhead, head_dim);
        case LLAISYS_DTYPE_F16:
            return self_attention_impl(reinterpret_cast<llaisys::fp16_t*>(attn_val), reinterpret_cast<const llaisys::fp16_t*>(q),
                                       reinterpret_cast<const llaisys::fp16_t*>(k), reinterpret_cast<const llaisys::fp16_t*>(v),
                                       scale, qlen, kvlen, nhead, nkvhead, head_dim);
        case LLAISYS_DTYPE_BF16:
            return self_attention_impl(reinterpret_cast<llaisys::bf16_t*>(attn_val), reinterpret_cast<const llaisys::bf16_t*>(q),
                                       reinterpret_cast<const llaisys::bf16_t*>(k), reinterpret_cast<const llaisys::bf16_t*>(v),
                                       scale, qlen, kvlen, nhead, nkvhead, head_dim);
        default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}