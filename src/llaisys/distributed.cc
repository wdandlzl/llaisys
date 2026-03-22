#include "llaisys/distributed.h"
#include "distributed_internal.hpp"

#include "../utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef ENABLE_MPI
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include <mpi.h>
#endif

namespace llaisys::distributed {
namespace {
std::mutex g_mutex;
bool g_initialized = false;
bool g_owned_init = false;
bool g_finalize_registered = false;
bool g_finalized = false;
#ifdef ENABLE_MPI
MPI_Comm g_comm = MPI_COMM_NULL;
#endif

void finalizeImpl() {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_initialized || g_finalized) {
        return;
    }
#ifdef ENABLE_MPI
    if (g_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&g_comm);
        g_comm = MPI_COMM_NULL;
    }
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (g_owned_init && !finalized) {
        MPI_Finalize();
    }
#endif
    g_finalized = true;
}

template <typename T>
void toFloatBuffer(float *dst, const T *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = llaisys::utils::cast<float>(src[i]);
    }
}

template <typename T>
void fromFloatBuffer(T *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = llaisys::utils::cast<T>(src[i]);
    }
}
} // namespace

void init() {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (g_initialized) {
        return;
    }
#ifdef ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char **argv = nullptr;
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        g_owned_init = true;
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &g_comm);
#endif
    g_initialized = true;
    if (!g_finalize_registered) {
        std::atexit(finalizeImpl);
        g_finalize_registered = true;
    }
}

bool enabled() {
#ifdef ENABLE_MPI
    return true;
#else
    return false;
#endif
}

int rank() {
    init();
#ifdef ENABLE_MPI
    int r = 0;
    MPI_Comm_rank(g_comm, &r);
    return r;
#else
    return 0;
#endif
}

int worldSize() {
    init();
#ifdef ENABLE_MPI
    int s = 1;
    MPI_Comm_size(g_comm, &s);
    return s;
#else
    return 1;
#endif
}

void barrier() {
    init();
#ifdef ENABLE_MPI
    if (worldSize() > 1) {
        MPI_Barrier(g_comm);
    }
#endif
}

void finalize() {
    finalizeImpl();
}

void allreduceSumInplace(std::byte *data, llaisysDataType_t dtype, size_t count) {
    init();
#ifdef ENABLE_MPI
    if (worldSize() == 1 || count == 0) {
        return;
    }
    if (dtype == LLAISYS_DTYPE_F32) {
        MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<float *>(data), static_cast<int>(count), MPI_FLOAT, MPI_SUM, g_comm);
        return;
    }

    std::vector<float> send(count);
    std::vector<float> recv(count);

    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        toFloatBuffer(send.data(), reinterpret_cast<const llaisys::fp16_t *>(data), count);
        break;
    case LLAISYS_DTYPE_BF16:
        toFloatBuffer(send.data(), reinterpret_cast<const llaisys::bf16_t *>(data), count);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    MPI_Allreduce(send.data(), recv.data(), static_cast<int>(count), MPI_FLOAT, MPI_SUM, g_comm);

    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        fromFloatBuffer(reinterpret_cast<llaisys::fp16_t *>(data), recv.data(), count);
        break;
    case LLAISYS_DTYPE_BF16:
        fromFloatBuffer(reinterpret_cast<llaisys::bf16_t *>(data), recv.data(), count);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
#else
    (void)data;
    (void)dtype;
    (void)count;
#endif
}

void broadcastInt64Inplace(int64_t *value, int root) {
    init();
#ifdef ENABLE_MPI
    if (worldSize() > 1) {
        MPI_Bcast(value, 1, MPI_LONG_LONG, root, g_comm);
    }
#else
    (void)value;
    (void)root;
#endif
}

} // namespace llaisys::distributed

__C {

int llaisysDistributedInit(void) {
    llaisys::distributed::init();
    return 0;
}

int llaisysDistributedIsEnabled(void) {
    return llaisys::distributed::enabled() ? 1 : 0;
}

int llaisysDistributedRank(void) {
    return llaisys::distributed::rank();
}

int llaisysDistributedWorldSize(void) {
    return llaisys::distributed::worldSize();
}

void llaisysDistributedBarrier(void) {
    llaisys::distributed::barrier();
}

void llaisysDistributedFinalize(void) {
    llaisys::distributed::finalize();
}

} // extern "C"
