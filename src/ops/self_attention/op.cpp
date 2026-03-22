#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    
    // q: [seqlen, nhead, d]
    // k, v: [total_len, nkvhead, d]
    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];

    size_t kvlen = k->shape()[0];
    size_t nkvhead = k->shape()[1];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), 
                                   scale, attn_val->dtype(), qlen, kvlen, nhead, nkvhead, d);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}