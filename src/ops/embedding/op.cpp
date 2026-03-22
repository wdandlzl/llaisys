#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    if (index->dtype() != LLAISYS_DTYPE_I64) EXCEPTION_DATATYPE_MISMATCH;
    
    // index is 1D, weight is 2D [vocab, dim], out is [index_len, dim]
    size_t num_indices = index->numel();
    size_t embedding_dim = weight->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                              out->dtype(), num_indices, embedding_dim);
    }

    // GPU dispatch ...
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}