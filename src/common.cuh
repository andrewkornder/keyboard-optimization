#ifndef SCORE_CUH
#define SCORE_CUH
#include <def.cuh>
#include <text.cuh>
#include <fstream>

typedef double score_t;
struct stats {
    score_t score{};

#ifdef METRIC_OKP
    constexpr static int attrs = 9;
    const constexpr static char* names[attrs] = {
        "sameHand", "sameFinger", "inRoll", "outRoll", "rowChange", "ringJump", "toCenter", "homeJump", "doubleJump"
    };
    text_t sameHand{};
    text_t sameFinger{};
    text_t inRoll{};
    text_t outRoll{};
    text_t rowChange{};
    text_t ringJump{};
    text_t toCenter{};
    text_t homeJump{};
    text_t doubleJump{};
#else
    constexpr static int attrs = 0;
    const constexpr static char** names = nullptr;
#endif
};



inline std::ostream& operator<<(std::ostream& file, const stats &s) {
    file << "{score=" << s.score;
    const text_t* arr = (text_t*) &s;
    for (int i = 0; i < stats::attrs; ++i) {
        file << "," << stats::names[i] << "=" << arr[i + 1];
    }
    file << "}";
    return file;
}

struct keyboard {
#ifdef LOCAL_KB
    __host__ __device__ keyboard(const keyboard &other) {
        score = other.score;

        const auto* A = (uint32_t*) other.arr;
        constexpr int max = ALIGNED / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < max; i++)
            arr[i] = A[i];
    }
    char arr[ALIGNED] = {};
#else
    char* arr = nullptr;
#endif
    stats stats{};

#ifdef REMOVE_DUPLICATES
    bool rescore = true;
#endif
};

struct keyboardc {
    char arr[ALIGNED] {};
    stats stats{};
};

typedef uint64_t pop_t;
struct Snapshot {
    Snapshot(pop_t, keyboardc*, keyboardc*, keyboardc*, double);
    explicit Snapshot(const Snapshot &other);
    void add(const std::unique_ptr<Snapshot>& other);
    pop_t size;
    keyboardc *best, *median, *worst;

    double average;

    ~Snapshot();
};


#define IDX(T) ((T) blockIdx.x * (T) blockDim.x + (T) threadIdx.x);


__global__ void initKeyboards(pop_t, keyboard*, char*);
__device__ void resetLockedKeys(char* pos);
__global__ void randomize(pop_t, keyboard*);

#endif //SCORE_CUH
