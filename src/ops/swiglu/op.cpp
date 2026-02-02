#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}