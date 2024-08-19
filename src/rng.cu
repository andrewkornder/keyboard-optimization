#include "rng.cuh"

__managed__ __device__ state SEED;

#ifdef RNG64
constexpr __constant__ state RS[] {
    0x6f17b935f6e98b7fULL, 0x6a488b3741fbb86bULL, 0x4b0731b10ff131ebULL, 0x527316c55f3b941fULL, 0x570e5c965a13afebULL,
    0x4685c99fc65418b3ULL, 0x5539e1c873b0d283ULL, 0x787e4a132e5ce993ULL, 0x5711a9f8d80e9c87ULL, 0x4a18f5fc28a25091ULL,
};
constexpr __constant__ state RK[] {
    0x5b737c9ec727d301ULL, 0x7fe12909b1f3d897ULL, 0x579ee157f16ee655ULL, 0x476882d452065bcbULL, 0x60831701bde193c9ULL,
    0x6a8f1ed0f79bd6e9ULL, 0x676e537a3bb81939ULL, 0x4fa85f2022ec6a13ULL, 0x524a0812d4f5b4cbULL, 0x655fe226ea0e503fULL,
    0x724b21b6c4b76d07ULL, 0x5e7ef0e675a3a8d3ULL, 0x5b0d5536e8e823e1ULL, 0x4db34e8c878ac7fdULL, 0x4869980fd0eccaa7ULL,
    0x70975fc99ed257b9ULL, 0x5719c56fd5db0403ULL, 0x48f2221fdee011cfULL, 0x48fe5fcbd32a5ee3ULL, 0x448318ba956bb0cdULL,
    0x620104f07717ca9bULL, 0x5cb995e193184985ULL, 0x50989e4c8a7197c7ULL, 0x4d89e496826e3237ULL, 0x656a77a176a220b1ULL,
    0x789b1d131ccbd0fbULL, 0x689e5076328c5543ULL, 0x66e553d500b2682fULL, 0x5028e21c00f54bd9ULL, 0x4d6626bb2c0b81b9ULL,
};

__device__ state next(state &x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}
#else
constexpr __constant__ state RS[] {
    0x4d06ca3d, 0x63648235, 0x706a1dbd, 0x5f67fca9, 0x4b6ef69b,
    0x5bfbc19f, 0x55dd80ff, 0x461c76bf, 0x7b30f553, 0x6a4d6485,
};
constexpr __constant__ state RK[] {
    0x51f16c93, 0x4c6757b1, 0x4bfd1e15, 0x5494f35b, 0x61b5f267,
    0x4bc3373f, 0x4a928961, 0x42e03105, 0x5da28279, 0x7cb2e205,
    0x76367053, 0x422b5863, 0x5189927b, 0x716a5189, 0x4875ca91,
    0x6e517733, 0x515dd9a5, 0x4c30a4c7, 0x74309497, 0x7b09be39,
    0x5f9e4dad, 0x7e922e6d, 0x7acfb43b, 0x58af1107, 0x48ea7277,
    0x67d1bf41, 0x6875b0ed, 0x4540ff91, 0x539c0825, 0x7e3a641b,
};

__device__ state next(state &x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}
#endif
__device__ float nextf(state &x) {
    return next(x) / (float) (state) -1;
}

__device__ state hash(const char* arr, const state i) {
    state s = RS[0] * (SEED * i + RS[1]);
    #pragma unroll
    for (int j = 0; j < KEYS; ++j) {
        s ^= (2 + arr[j]) * RK[j];
    }
    return s * (SEED + RS[9]);
}

__device__ state hash(const state x, const state y) {
    return x * RS[6] ^ y * RS[7];
}

__device__ state hash(const state x, const state y, const state z) {
    return x * RS[6] ^ y * RS[7] ^ z * RS[8];
}

__global__ void updateStateLocal(const char* kb, const state offset) {
    state s = hash(kb, offset) * RS[2];
    state out = RS[3] * SEED;
    for (int i = 0; i < 3; i++) out = out * RS[4] ^ next(s) * RS[5];
#ifdef RNG64
    SEED = out;
#else
    SEED = out >> 32 ^ out;
#endif
}

__host__ void updateState(const char* kb, const state offset) {
    updateStateLocal<<<1, 1>>>(kb, offset);
    cudaDeviceSynchronize();
}
