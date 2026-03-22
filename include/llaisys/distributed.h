#ifndef LLAISYS_DISTRIBUTED_H
#define LLAISYS_DISTRIBUTED_H

#include "../llaisys.h"

__C {
    __export int llaisysDistributedInit(void);
    __export int llaisysDistributedIsEnabled(void);
    __export int llaisysDistributedRank(void);
    __export int llaisysDistributedWorldSize(void);
    __export void llaisysDistributedBarrier(void);
    __export void llaisysDistributedFinalize(void);
}

#endif // LLAISYS_DISTRIBUTED_H
