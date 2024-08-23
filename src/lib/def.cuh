#ifndef DEF_CUH
#define DEF_CUH


#include <choices.cuh>
#include <string>


constexpr uint64_t ceilLog2(uint64_t x) {
    int z = (x & x - 1) != 0;
    while (x > 1) {
        x >>= 1;
        ++z;
    }
    return z;
}

inline constexpr char KEYS_LOWER[] = {
    // '1','2','3','4','5','6','7','8','9','0',
    'q','w','e','r','t','y','u','i','o','p',
    'a','s','d','f','g','h','j','k','l',';',
    'z','x','c','v','b','n','m',',','.','/',
};
inline constexpr char KEYS_UPPER[] = {
    // '!','@','#','$','%','^','&','*','(',')',
    'Q','W','E','R','T','Y','U','I','O','P',
    'A','S','D','F','G','H','J','K','L',':',
    'Z','X','C','V','B','N','M','<','>','?',
};
constexpr uint32_t KEYS = std::size(KEYS_LOWER);
constexpr uint32_t KEYS2 = KEYS * KEYS;
constexpr uint32_t ALIGNED = 1 << ceilLog2(KEYS);

#ifdef METRIC_CARPALX
constexpr uint32_t textWindow = 3;
#else
#ifdef METRIC_OKP
constexpr uint32_t textWindow = 2;
#elif defined(METRIC_DIST)
constexpr uint32_t textWindow = 2;
#endif
#endif


inline __device__ __constant__ bool movable[KEYS] = {};


#if defined(MINIMIZE) ^ defined(MAXIMIZE)
    #if defined(MINIMIZE)
        #define CMP(boolean) boolean
    #else
        #define CMP(boolean) (!(boolean))
    #endif
#else
    #error Either 'MINIMIZE' or 'MAXIMIZE' can be defined, not both
#endif

#ifdef LOCAL_KB
#undef REARRANGE
#endif


#ifndef COPYMODE
#error Must define COPYMODE to one of the three values.
#endif
constexpr int copyStride = ALIGNED / copyGroups;



#include <cstdio>
inline void cudaThrowErr(const cudaError_t c, const char* file, const int line) {
    if (c != cudaSuccess) {
        printf("%s+%d [CUDA Error: 0x%02x] %s\n", file, line, c, cudaGetErrorString(c));
        exit(c);
    }
}
#define err(c) cudaThrowErr(c, __FILE__, __LINE__)

template<int precision = 4, typename T>
std::string formatNumber(T x) {
    std::string s;

    {
        uint64_t t = (uint64_t) x;

        int i = 0;
        do {
            s.insert(0, 1, (char) ('0' + (t % 10)));
            t /= 10;
            if (i++ % 3 == 2 && t) s.insert(0, ",");
        } while (t);
    }
    if constexpr (std::is_integral_v<T>) {
        return s;
    }

    x -= (uint64_t) x;

    s.append(1, '.');
    for (int i = 0; i < precision; ++i) {
        x *= 10;
        s.append(1, (char) ('0' + (int) x % 10));
        x -= (uint64_t) x;
    }
    return s;
}
#define F3(number) formatNumber(number).c_str()
#define F3p(number, p) formatNumber<p>(number).c_str()

#endif //DEF_CUH
