#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::distributed {
void init();
bool enabled();
int rank();
int worldSize();
void barrier();
void finalize();
void allreduceSumInplace(std::byte *data, llaisysDataType_t dtype, size_t count);
void broadcastInt64Inplace(int64_t *value, int root = 0);
} // namespace llaisys::distributed
