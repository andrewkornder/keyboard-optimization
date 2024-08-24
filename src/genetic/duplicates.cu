#include "duplicates.cuh"



__device__ bool compareArrays(const char* a, const char* b) {
    for (int j = 0; j < KEYS; ++j) {
        if (a[j] != b[j]) { return false; }
    }
    return true;
}

#include <rng.cuh>


__global__ void removeDuplicatesAIO(const pop_t n, keyboard* kbs) {
    const int tid = threadIdx.x;

    for (int i = tid + 1; i < n; i += 1024) {
        kbs[i].rescore = compareArrays(kbs[i - 1].arr, kbs[i].arr);
    }

    __syncthreads();

    for (int i = tid + 1; i < n; i += 1024) {
        if (kbs[i].rescore) {
            char* out = kbs[i].arr;
            state rng = hash(out, i);

#pragma unroll
            for (int j = 0; j < KEYS - 1; ++j) {
                if (movable[j] == LCK) continue;

                const char k = j + next(rng) % (KEYS - j);
                if (movable[k] == LCK) continue;

                SWAP(out, j, k);
            }
        }
    }
}

__global__ void identifyDuplicates(const pop_t n, keyboard* kbs) {
    const pop_t i = IDX(pop_t);
    if (0 < i && i < n)
        kbs[i].rescore = compareArrays(kbs[i - 1].arr, kbs[i].arr);
}
__global__ void removeDuplicates(const pop_t n, const keyboard* kbs) {
    const pop_t i = IDX(pop_t);

    if (0 < i && i < n && kbs[i].rescore) {
        char* out = kbs[i].arr;
        state rng = hash(out, i);

#pragma unroll
        for (int j = 0; j < KEYS - 1; ++j) {
            if (movable[j] == LCK) continue;

            const char k = j + next(rng) % (KEYS - j);
            if (movable[k] == LCK) continue;

            SWAP(out, j, k);
        }
    }
}


__global__ void countDuplicates(const pop_t n, const keyboard* kbs, pop_t* count) {
    constexpr int k = 1024;
    __shared__ pop_t block[k];

    const int tid = threadIdx.x;

    block[tid] = 0;
    for (int i = tid; i < n - 1; i += k) {
        if (i < n - 1 && compareArrays(kbs[i + 1].arr, kbs[i].arr)) {
            ++block[tid];
        }
    }

    __syncthreads();


#pragma unroll
    for (int offset = k / 2; offset > 0; offset = offset / 2) {
        if (tid < offset) {
            block[tid] += block[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) *count = *block;
}

pop_t countDuplicates(const pop_t n, const keyboard* kbs) {
    pop_t* out;
    cudaMallocManaged(&out, sizeof(pop_t));

    countDuplicates<<<1, 1024>>>(n, kbs, out);
    cudaDeviceSynchronize();

    const pop_t x = *out;
    cudaFree(out);
    return x;
}
