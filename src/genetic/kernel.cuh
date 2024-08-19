#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <common.cuh>

#ifndef LOCAL_KB
__global__ void reorder(pop_t, keyboard*, char*, const char*);
#endif

#ifdef LOCAL_KB
__global__ void copyRange(pop_t, const keyboard*, keyboard*);
__global__ void createNext(pop_t, pop_t, keyboard*, const keyboard*);
#else
__global__ void copyRange(pop_t, const keyboard*, char*);
__global__ void createNext(pop_t, pop_t, keyboard*, const char*);
#endif


__global__ void prepareSnapshot(pop_t, const keyboard*, char*, stats*);
#endif //KERNEL_CUH
