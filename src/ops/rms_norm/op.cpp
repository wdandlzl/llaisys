#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    // in: [rows, dim], weight: [dim]
    size_t rows = in->shape()[0];
    size_t dim = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), rows, dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}