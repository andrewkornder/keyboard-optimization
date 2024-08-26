#ifndef TEXT_CUH
#define TEXT_CUH

#include <def.cuh>
#include <memory>
#include <fstream>
#include <filesystem>

typedef uint32_t text_t;
typedef uint64_t count_t;

inline std::filesystem::path cacheDirectory;

__host__ __device__ static constexpr uint64_t ipow(const uint64_t x, const uint64_t p, const uint64_t m = 1) {
    return p > 0 ? ipow(x, p - 1, m * x) : m;
}

constexpr int ngramStride = KEYS + 1;
constexpr int ngramCount = ipow(ngramStride, textWindow);

__forceinline__ __host__ __device__ constexpr int sampleIdx(const int a, const int b, const int c) {
    constexpr int k0 = ipow(ngramStride, 0);
    constexpr int k1 = ipow(ngramStride, 1);
    constexpr int k2 = ipow(ngramStride, 2);
    constexpr int bias = k0 + k1 + k2;
    return a * k2 + b * k1 + c * k0 + bias;
}
__forceinline__ __host__ __device__ constexpr int sampleIdx(const int a, const int b) {
    return sampleIdx(-1, a, b);
}
__forceinline__ __host__ __device__ constexpr int sampleIdx(const int a) {
    return sampleIdx(-1, -1, a);
}


template<typename T>
__forceinline__ __host__ __device__ T sample(const T* arr, const int a, const int b) {
    return arr[sampleIdx(a, b)];
}
template<typename T>
__forceinline__ __host__ __device__ T sample(const T* arr, const int a, const int b, const int c) {
    return arr[sampleIdx(a, b, c)];
}


template<typename T>
__forceinline__ __host__ __device__ T* sampleAddr(T* arr, const int a, const int b) {
    return &arr[sampleIdx(a, b)];
}
template<typename T>
__forceinline__ __host__ __device__ T* sampleAddr(T* arr, const int a, const int b, const int c) {
    return &arr[sampleIdx(a, b, c)];
}


struct LettersClass {
private:
    char keyboardPosition[256] = {};

public:
    constexpr LettersClass() noexcept {
        for (int i = 0; i < 256; ++i) {
            keyboardPosition[i] = -1;
        }
        for (int i = 0; i < KEYS; ++i) {
            keyboardPosition[KEYS_LOWER[i]] = i;
            keyboardPosition[KEYS_UPPER[i]] = i;
        }
    }

    constexpr char positionOf(const int c) const {
        if (c > 255) return -1;
        return keyboardPosition[c];
    }
    constexpr char positionOf(const char c) const {
        return keyboardPosition[c];
    }

    template<int size, int width, typename T>
    constexpr static void getCharsAtIndex(uint64_t i, T out[size]) {
        for (int j = 0; j < size; ++j) {
            out[size - j - 1] = i % (width + 1) - 1;
            i /= width + 1;
        }
    }

    template<int size, int width, typename T>
    constexpr static void getCharsAtIndexUnsigned(uint64_t i, T out[size]) {
        for (int j = 0; j < size; ++j) {
            out[size - j - 1] = i % (width + 1);
            i /= width + 1;
        }
    }

    template<int size, int width, typename T>
    constexpr static uint64_t getIndexAtChars(const T out[size]) {
        uint64_t x = 0;
        for (int j = 0; j < size; ++j) {
            x = (width + 1) * x + (1 + out[j]);
        }
        return x;
    }

    template <int size, int width, typename T>
    static void applyOnIndices(const T function) {
        constexpr int it = ipow(width + 1, size);
        for (int i = 0; i < it; ++i) {
            char code[size];
            getCharsAtIndex<size, width>(i, code);
            function(i, code);
        }
    }
};
constexpr LettersClass letterUtils;


struct FinishedText {
    constexpr static count_t N = ipow(ngramStride, textWindow);
    explicit FinishedText(const count_t *counts) : count(0), scaled(0) {
        cpuArray = new text_t[N];

        count_t maxValue = 0;
        constexpr count_t cap = (text_t) -1;
        for (int i = 0; i < N; ++i) {
            maxValue = maxValue > counts[i] ? maxValue : counts[i];
        }

        count_t div = maxValue == 0 ? 1 : maxValue / cap + (maxValue % cap != 0);
        div = div == 0 ? 1 : div;

        for (int i = 0; i < N; ++i) {
            const count_t v = counts[i];

            cpuArray[i] = v / div;
            scaled += v / div;
            count += v;
        }

        cudaMalloc(&gpuArray, sizeof(text_t) * N);
        cudaMemcpy(gpuArray, cpuArray, sizeof(text_t) * N, cudaMemcpyHostToDevice);
    }

    FinishedText(FinishedText &&other) = delete;
    FinishedText(const FinishedText &other) = delete;

    count_t count, scaled;
    text_t *cpuArray, *gpuArray;
};

std::unique_ptr<FinishedText> initText(const std::filesystem::path &exportTo, std::vector<std::filesystem::path> &corpora);

__host__ __device__ void printArr(const char* arr);
__host__ void printArrQ(const char* arr);
#endif //TEXT_CUH
