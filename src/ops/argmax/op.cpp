#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp" // 必须包含这个头文件

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 简单检查设备一致性
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // max_idx 必须是 int64 类型
    if (max_idx->dtype() != LLAISYS_DTYPE_I64) EXCEPTION_DATATYPE_MISMATCH;

    // 如果是 CPU 设备，调用 CPU 实现
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                           vals->dtype(), vals->numel());
    }

    // GPU 实现预留...
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops