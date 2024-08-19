#ifndef TEXT_H
#define TEXT_H

#include <def.cuh>
#include <memory>
#include <fstream>
#include <vector>

typedef uint32_t text_t;
typedef uint64_t count_t;

inline char cachedIndexFile[1024] = R"(K:\data\code\clion\untitled\cached\index)";
inline char cacheDirectory[1024] = R"(K:\data\code\clion\untitled\cached\)";
inline bool cachedText = false;


__host__ __device__ static constexpr int ipow(const int x, const int p, const int m = 1) {
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
    // not including space
    static constexpr int totalPrintable = 94;
    char keyboardPosition[256] = {};
    char printableToCode[256] = {};
    char codeToPrintable[256] = {};

    constexpr LettersClass() noexcept {
        for (int i = 0; i < 256; ++i) {
            keyboardPosition[i] = -1;
            printableToCode[i] = -1;
            codeToPrintable[i] = ' ';
        }
        for (char c = '!'; c <= '~'; ++c) {
            printableToCode[c] = c - '!';
            codeToPrintable[c - '!'] = c;
        }
        for (int i = 0; i < KEYS; ++i) {
            keyboardPosition[QC0[i]] = i;
            keyboardPosition[QC1[i]] = i;
        }
    }

    template<int size, int width>
    static int indexOf(const char code[size]) {
        int i = 0, m = ipow(width + 1, size - 1);
        for (int j = 0; j < size; ++j) {
            i += (1 + code[j]) * m;
            m /= width + 1;
        }
        return i;
    }

    template<int size, int width>
    static void getCharsAtIndex(int i, char out[size]) {
        for (int j = 0; j < size; ++j) {
            out[size - j - 1] = i % width - 1;
            i /= width;
        }
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


struct UnfinishedText {
    static constexpr count_t N = ipow(letterUtils.totalPrintable + 1, maxTextWindow);

    UnfinishedText() {
        ngrams = new count_t[N]();
    }
    ~UnfinishedText() { delete[] ngrams; }
    UnfinishedText(const UnfinishedText &other) = delete;

    count_t* ngrams;
};

template <int window>
struct FinishedText {
    static constexpr int N = ipow(ngramStride, window);

private:
    template <typename T, typename Tranform>
    void moveCounts(const T* arr, Tranform f) {
        letterUtils.applyOnIndices<maxTextWindow, letterUtils.totalPrintable>([&f, self = cpuArray, other = arr](const int index, const char keys[window]) {
            char code[window] = {};
            for (int i = 0; i < window; ++i) {
                const char remapped = letterUtils.codeToPrintable[keys[i]];
                code[i] = letterUtils.keyboardPosition[remapped];
            }
            self[letterUtils.indexOf<window, KEYS>(code)] = f(other[index]);
        });
    }

public:
    explicit FinishedText(const UnfinishedText &text) : count(0) {
        cpuArray = new text_t[N]();
        for (int i = 0; i < text.N; ++i) {
            count += text.ngrams[i];
        }

        if constexpr (std::is_integral_v<text_t>) {
            constexpr text_t maxValue = (text_t) -1;

            count_t max = 0;
            for (int i = 0; i < N; ++i) {
                max = max < text.ngrams[i] ? text.ngrams[i] : max;
            }
            count_t div = max == 0 ? 1 : (max + maxValue - 1) / maxValue;
            moveCounts(text.ngrams, [div](const count_t c) { return (text_t) (c / div); });
        } else {
            moveCounts(text.ngrams, [count = count](const count_t c) { return (text_t) c / count; });
        }

        cudaMalloc(&gpuArray, N * sizeof(text_t));
        cudaMemcpy(gpuArray, cpuArray, N * sizeof(text_t), cudaMemcpyHostToDevice);
    }
    ~FinishedText() {
        cudaFree(gpuArray);
        delete[] cpuArray;
    }
    FinishedText(const FinishedText &other) = delete;

    count_t count;
    text_t* cpuArray;
    text_t* gpuArray;
};

std::unique_ptr<FinishedText<textWindow>> initText(const char*, const std::vector<std::string>&);
std::unique_ptr<FinishedText<textWindow>> initText(const char*);

void mapKeys(const char* arr, char* out);

__host__ __device__ void printArr(const char* arr);
__host__ void printArrQ(const char* arr);
#endif //TEXT_H
