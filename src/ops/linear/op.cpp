#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // in: [M, K], weight: [N, K], out: [M, N]
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    std::byte* bias_data = bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), M, N, K);
    }
    
    // GPU dispatch ...
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}