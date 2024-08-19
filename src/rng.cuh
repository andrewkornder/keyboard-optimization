#ifndef RNG_CUH
#define RNG_CUH

#include <def.cuh>

#ifdef RNG64
typedef uint64_t state;
#else
typedef uint32_t state;
#endif

__device__ state next(state &x);
__device__ float nextf(state &x);

__device__ state hash(const char* arr, state i);
__device__ state hash(state x, state y, state z);
__device__ state hash(state x, state y);

__managed__ __device__ extern state SEED;

void updateState(const char*, state);

#endif //RNG_CUH
