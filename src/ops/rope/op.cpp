#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) EXCEPTION_DATATYPE_MISMATCH;

    // assume contiguous [seqlen, nheads, head_dim]
    size_t seqlen = in->shape()[0];
    size_t nheads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seqlen, nheads, head_dim);
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}