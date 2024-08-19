#include "kernel.cuh"
#include <def.cuh>
#include <rng.cuh>


#ifdef LOCAL_KB
__global__ void copyRange(const pop_t size, const keyboard* population, keyboard* copyTo) {
    const pop_t i = IDX(pop_t);
    if (i < size) {
        copyTo[i] = population[i];
    }
}

#else
#if COPYMODE == COPY_MEMCPY
#define _COPY(size) memcpy(to, from, size)
#elif COPYMODE == COPYMODE_BYTE
#define _COPY(size) \
_Pragma("unroll") \
for (int i = 0; i < size; i++) to[i] = from[i]
#elif COPYMODE == COPYMODE_CAST
#define _COPY(size) \
const auto* A = (uint64_t*) from; \
auto* B = (uint64_t*) to; \
constexpr int max = size / sizeof(uint64_t); \
_Pragma("unroll") \
for (int i = 0; i < max; i++) B[i] = A[i];
#endif

#define CFIXED(size) __device__ __forceinline__ void _copyFixed##size(const char* from, char* to) {_COPY(size);}
#define copyFixed(from, to, size)  _copyFixed##size(from, to)
CFIXED(copyStride);
CFIXED(ALIGNED);
__global__ void copyRange(const pop_t size, const keyboard* population, char* copyTo) {
    const pop_t i = IDX(pop_t);

    const pop_t idx = i / copyGroups;
    const pop_t group = i % copyGroups;

    if (idx >= size) return;

    const char* from = population[idx].arr + group * copyStride;
    char* to = &copyTo[idx * ALIGNED] + group * copyStride;
    copyFixed(from, to, copyStride);
}

__global__ void reorder(const pop_t size, keyboard* population, char* base, const char* copy) {
    const pop_t i = IDX(pop_t);

    const pop_t idx = i / copyGroups;
    const pop_t group = i % copyGroups;

    if (idx >= size) return;

    const char* from = &copy[idx * ALIGNED] + group * copyStride;
    char* to = from - copy + base;
    copyFixed(from, to, copyStride);
    population[idx].arr = to;
}
#endif

#ifdef LOCAL_KB
__global__ void createNext(const pop_t n, const pop_t k, keyboard* population, const keyboard* top) {
#else
__global__ void createNext(const pop_t n, const pop_t k, keyboard* population, const char* top) {
#endif
    const pop_t i = IDX(pop_t);
    if (i >= n) return;

    char* out = population[i].arr;
    population[i].stats = {};
#ifdef REMOVE_DUPLICATES
    population[i].rescore = true;
#endif

#ifdef SINGLEMUTATE
#ifdef ORDERED_CREATION
    const int x_i = k * i / n;
    state rng = hash(out, hash3(i, x_i, out[0]));
#else
    state rng = hash(out, i);
    const int x_i = next(rng) % k;
#endif

    const char* x = &top[ALIGNED * x_i];

    copyFixed(x, out, ALIGNED);
#else
#ifdef ORDERED_CREATION
    const int offset = i * k * k / n;
    const int x_i = offset % k;
    const int y_i = offset / k % k;
    state rng = hash(out, hash(i, x_i, y_i));
#else
    state rng = hash(out, i);
    const int x_i = next(rng) % k;
    const int y_i = next(rng) % k;
#endif

    const char* x = &top[ALIGNED * x_i];
    const char* y = &top[ALIGNED * y_i];

    int totalEmpty = 0;
    char empty[KEYS] = {};

    bool used[KEYS] = {};
    memset(used, false, KEYS);

    const state choices = next(rng);
    for (int j = 0; j < KEYS; j++) {
        if (const char let = (choices >> j & 1 ? x : y)[j]; used[let]) {
            empty[totalEmpty++] = j;
        } else {
            out[j] = let;
            used[let] = true;
        }
    }

    for (int j = totalEmpty - 1; j > 0; j--) {
        const int v = next(rng) % (j + 1);
        SWAP(empty, j, v);
    }

    for (int j = 0, l = 0; j < totalEmpty; j++) {
        while (used[l]) ++l;

        out[empty[j]] = l++;
    }
#endif
    const uint32_t totalSwaps = next(rng) % maxSwaps;
    for (int s = 0; s < totalSwaps; ++s) {
        const uint32_t ij = next(rng);

        const int a = ij % KEYS;
        if (movable[a] == LCK) continue;

        const int b = (ij >> 8) % KEYS;
        if (movable[b] == LCK) continue;

        SWAP(out, a, b);
    }

}


__global__ void prepareSnapshot(const pop_t n, const keyboard* population, char* arr, stats* scores) {
    const int i = threadIdx.x;
    const pop_t j = (n - 1) / 2 * i;

    const keyboard &kb = population[j];
    memcpy(arr + i * KEYS, kb.arr, KEYS);
    scores[i] = kb.stats;
}
