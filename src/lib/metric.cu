// ReSharper disable CppTooWideScope
#include <fstream>
#include <metric.cuh>
#include <text.cuh>



#define SCALESCORE(s, count) { \
    if constexpr (std::is_integral_v<text_t> && std::is_floating_point_v<score_t>) { \
        s.score /= count; \
    } \
}

struct Metric {
    Metric() {
        cpuArray = new mtype[ngramCount]();
    }
    void send() {
        cudaMalloc(&gpuArray, sizeof(mtype) * ngramCount);
        cudaMemcpy(gpuArray, cpuArray, sizeof(mtype) * ngramCount, cudaMemcpyHostToDevice);
    }

    ~Metric() { cudaFree(gpuArray); delete[] cpuArray; }
    Metric(const Metric &other) = delete;

    void add(int index, const char keys[textWindow]) const;

    mtype* cpuArray = nullptr;
    mtype* gpuArray = nullptr;
};

std::shared_ptr<FinishedText> sharedText;
Metric metric;


__global__ LAUNCH_BOUNDS_DEFAULT void scoreGPU(pop_t n, keyboard* population, count_t count,
    const text_t* ngrams, const mtype* metric);

void score(const pop_t n, keyboard* population) {
    const pop_t blocks = n > 1024 ? n / 1024 : 1;
    scoreGPU<<<blocks, 1024>>>(n, population, sharedText->scaled, sharedText->gpuArray, metric.gpuArray);
}

const mtype* precomputeMetric(const std::shared_ptr<FinishedText>& text) {
    sharedText = text;

    letterUtils.applyOnIndices<textWindow, KEYS>([](const int index, const char keys[textWindow]) {
        metric.add(index, keys);
    });

    metric.send();
    return metric.cpuArray;
}

#ifdef METRIC_DIST

double weightFinger(const int i) {
    const int col = i % 10;
    const double dist = col / 9. - 0.5;
    return 1 + dist * dist;
}

void Metric::add(const int index, const char keys[2]) const {
    const char i = keys[0], j = keys[1];
    if (j == -1) {
        cpuArray[index] = {};
        return;
    }

    static constexpr char fingers[] {
        10, 11, 12, 13, 13, 16, 16, 17, 18, 19,
        10, 11, 12, 13, 13, 16, 16, 17, 18, 19,
        10, 11, 12, 13, 13, 16, 16, 17, 18, 19,
    };

    const char fi = fingers[i];
    const char fj = fingers[j];

    if (fi != fj || i == -1) {
        const char k2[2] = {fj, j};
        add(index, k2);
        return;
    }

    // ReSharper disable once CppTooWideScope
    constexpr bool log = true;
    constexpr double horizontalWeight = 1, verticalWeight = 1;
    constexpr double xOffset[] = {0, 0.25, 0.75};

    const int colA = i % 10, rowA = i / 10;
    const int colB = j % 10, rowB = j / 10;

    double dst = hypot(
        horizontalWeight * (colB - colA + xOffset[rowB] - xOffset[rowA]),
        verticalWeight * (rowB - rowA)
    );

    if (log) {
        dst = log1p(dst);
    }
    dst *= weightFinger(fi) * weightFinger(fj);
    cpuArray[index] = {dst};
}


__host__ __device__ stats score(const char* positions, const count_t count, const text_t* text, const mtype* metric) {
    stats total = {};
    // printArr(positions);
    for (char x = 0; x < KEYS; x++) {
        const char a = positions[x];

        for (char y = 0; y < KEYS; y++) {
            const char b = positions[y];
            // printf("[%d -> %d] %u x %f\n", x, y, sample(text, x, y), sample(metric, a, b).cost);
            total.score += sample(text, x, y) * sample(metric, a, b).cost;
        }

        total.score += sample(text, -1, x) * sample(metric, -1, a).cost;
    }
    SCALESCORE(total, count);
    return total;
}

__global__ LAUNCH_BOUNDS_DEFAULT void scoreGPU(const pop_t n, keyboard* population, const count_t count,
                                            const text_t* ngrams, const mtype* metric) {
    const int tid = threadIdx.x;

    __shared__ mtype s_metric[ngramCount];
    __shared__ text_t s_text[ngramCount];

    if (tid < ngramCount) {
        s_metric[tid] = metric[tid];
        s_text[tid] = ngrams[tid];
    }
    __syncthreads();

    const pop_t idx = IDX(pop_t);
    if (idx >= n) return;
    if (!population[idx].rescore) return;
    population[idx].rescore = false;

    keyboard &kb = population[idx];
    const char* positions = kb.arr;

    kb.stats = score(positions, count, s_text, s_metric);
}

stats score(const char* positions) {
    return score(positions, sharedText->scaled, sharedText->cpuArray, metric.cpuArray);
}

#elif defined(METRIC_OKP)

constexpr char INDEX = 3;
constexpr char MIDDLE = 2;
constexpr char RING = 1;
constexpr char PINKY = 0;
constexpr char fingers[KEYS] {
    PINKY,RING,MIDDLE,INDEX,INDEX,INDEX,INDEX,MIDDLE,RING,PINKY,
    PINKY,RING,MIDDLE,INDEX,INDEX,INDEX,INDEX,MIDDLE,RING,PINKY,
    PINKY,RING,MIDDLE,INDEX,INDEX,INDEX,INDEX,MIDDLE,RING,PINKY,
};

// DUPLICATED(constexpr double, FingerPercents[4],
//     {0.075, 0.1, 0.2, 0.2}
// );
// DUPLICATED(constexpr double, FingerWorkCosts[4],
//     {40, 30, 20, 20}
// );

struct HomeJumpReturn {
    int cost = 0;
    bool homeJump = false;
    bool doubleJump = false;
};

inline HomeJumpReturn getHomeJump(const int i, const int j) {
    const int row0 = i / 10;
    const int row1 = j / 10;

    HomeJumpReturn ret;
    if (row0 < row1) {
        if (1 <= row0 || 1 >= row1) return ret;
    } else if (row0 > row1) {
        if (1 <= row1 || 1 >= row0) return ret;
    } else return ret;

    const int dr = abs(row0 - row1);
    if (dr < 2) return ret;

    const char fi = fingers[i];
    const char fj = fingers[j];

    constexpr int homeJump = 100;
    constexpr int homeJumpIndex = -90;
    constexpr int doubleJump = 220;

    if (dr == 2) { ret.cost += homeJump; ret.homeJump = true; }
    else { ret.cost += doubleJump; ret.doubleJump = true; }

    if ((row0 > row1 && fi == INDEX && (fj == MIDDLE || fj == RING)) ||
        (row1 > row0 && fj == INDEX && (fi == MIDDLE || fi == RING))) {
        ret.cost += homeJumpIndex;
    };
    return ret;
}


void Metric::add(const int index, const char keys[2]) const {
    const char i = keys[0], j = keys[1];
    mtype &s = cpuArray[index];

    if (j == -1) return;
    if (i == -1) {
        constexpr short distanceCosts[] {
            40, 40, 30, 40, 70, 80, 40, 30, 40, 40,
             0,  0,  0,  0, 30, 30,  0,  0,  0,  0,
            70, 70, 70, 50, 95, 60, 40, 60, 70, 70,
        };
        s.cost = distanceCosts[j];
        return;
    }

    constexpr short inRoll =       -40;
    constexpr short outRoll =        5;
    constexpr short sameHand =       5;
    constexpr short rowChangeDown = 10;
    constexpr short rowChangeUp =   15;
    constexpr short handWarp =      25;
    constexpr short handSmooth =    -5;
    constexpr short ringJump =      40;
    constexpr short toCenter =      30;

#define T true
#define F false
    constexpr bool hand[KEYS] = {
        F,F,F,F,F,  T,T,T,T,T,
        F,F,F,F,F,  T,T,T,T,T,
        F,F,F,F,F,  T,T,T,T,T,
    };
    constexpr bool outside[10] = {
        T,F,F,F,F,  F,F,F,F,T
    };
    constexpr bool center[10] = {
        F,F,F,F,T,  T,F,F,F,F
    };
    constexpr int sameFingerCosts[4] = {
        150, 140, 110, 90
    };

    if (hand[i] != hand[j]) return;
    const char fi = fingers[i], ri = i / 10;
    const bool oi = outside[i % 10], ci = center[i % 10], ei = ci || oi;

    const char fj = fingers[j], rj = j / 10;
    const bool oj = outside[j % 10], cj = center[j % 10], ej = cj || oj;


    s.cost += sameHand;
    s.sameHand = true;

    if (fi == fj && i != j) {
        s.cost += sameFingerCosts[fi];
        s.sameFinger = true;
    }
    if (ri == rj && !ei && !ej) {
        if (fj == fi + 1) {s.inRoll = true; s.cost += inRoll;}
        if (fj == fi - 1) {s.outRoll = true; s.cost += outRoll;}
    }
    if (ri != rj) {
        s.rowChange = true;
        if (ri < rj) {
            s.cost += rowChangeDown;
            if (fi == 2 && fj == 3) s.cost += handSmooth;
            else if (abs(fi - fj) == 1) s.cost += handWarp;
        } else {
            s.cost += rowChangeUp;
            if (fi == 3 && fj == 2) s.cost += handSmooth;
            else if (abs(fi - fj) == 1) s.cost += handWarp;
        }
    }

    if ((fi == 0 && fj == 2) || (fi == 2 && fj == 0)) {
        s.ringJump = true; s.cost += ringJump;
    }
    if (ci ^ cj) {s.toCenter = true; s.cost += toCenter;}

    const HomeJumpReturn hj = getHomeJump(i, j);
    s.cost += hj.cost;
    s.homeJump = hj.homeJump;
    s.doubleJump = hj.doubleJump;
}



__host__ __device__ __forceinline void addCost(stats &out, const text_t freq, const mtype cost) {
    out.sameHand   += freq * cost.sameHand;
    out.inRoll     += freq * cost.inRoll;
    out.outRoll    += freq * cost.outRoll;
    out.sameFinger += freq * cost.sameFinger;
    out.rowChange  += freq * cost.rowChange;
    out.homeJump   += freq * cost.homeJump;
    out.doubleJump += freq * cost.doubleJump;
    out.ringJump   += freq * cost.ringJump;
    out.toCenter   += freq * cost.toCenter;

    // explicit conversion to avoid overflow
    out.score      += freq * (score_t) cost.cost;
}

__host__ __device__ stats score(const char* positions, const count_t count, const text_t* ngrams, const mtype* metric) {
    stats total = {};
    for (int i = 0; i < KEYS; ++i) {
        const char a = positions[i];
        addCost(total, sample(ngrams, -1, i), sample(metric, -1, a));
        for (int j = 0; j < KEYS; ++j) {
            const char b = positions[j];
            addCost(total, sample(ngrams, i, j), sample(metric, a, b));
        }
    }
    SCALESCORE(total, count);
    return total;
}

__global__ LAUNCH_BOUNDS_DEFAULT void scoreGPU(const pop_t n, keyboard* population, const count_t count,
                                             const text_t* ngrams, const mtype* metric) {
    const int tid = threadIdx.x;

    __shared__ mtype s_metric[ngramCount];
    __shared__ text_t s_text[ngramCount];

    if (tid < ngramCount) {
        s_metric[tid] = metric[tid];
        s_text[tid] = ngrams[tid];
    }
    __syncthreads();

    const pop_t idx = IDX(pop_t);
    if (idx >= n) return;
    if (!population[idx].rescore) return;
    population[idx].rescore = false;

    keyboard &kb = population[idx];
    const char* positions = kb.arr;

    kb.stats = score(positions, count, s_text, s_metric);
}

stats score(const char* positions) {
    return score(positions, sharedText->scaled, sharedText->cpuArray, metric.cpuArray);
}
#elif defined(METRIC_CARPALX)

#include "carpalx.cu"

void Metric::add(const int index, const char keys[3]) const {
    if (keys[0] != -1 && keys[1] != -1 && keys[2] != -1) {
        cpuArray[index] = mtype{effort(keys[0], keys[1], keys[2])};
    }
}

__global__ LAUNCH_BOUNDS_DEFAULT void scoreGPU(const pop_t n, keyboard* population, const count_t count,
                                             const text_t* ngrams, const mtype* metric) {
    const int tid = threadIdx.x;
    const pop_t idx = IDX(pop_t);
    if (idx >= n) return;
    if (!population[idx].rescore) return;
    population[idx].rescore = false;

    const char* positions = population[idx].arr;

    __shared__ text_t s_text[ipow(ngramStride, 2)];

    stats total = {};
    for (int x = 0; x < KEYS; x++) {
        if (tid < KEYS2) {
            const int i = tid / KEYS, j = tid % KEYS;
            *sampleAddr(s_text, i, j) = sample(ngrams, x, i, j);
        }
        __syncthreads();

        const char a = positions[x];
        for (int y = 0; y < KEYS; y++) {
            const char b = positions[y];
            for (int z = 0; z < KEYS; z++) {
                const char c = positions[z];
                total.score += sample(s_text, y, z) * sample(metric, a, b, c).cost;
            }
        }
    }
    SCALESCORE(total, count);
    population[idx].stats = total;
}

stats score(const char* positions) {
    stats total = {};
    for (int x = 0; x < KEYS; x++) {
        const char a = positions[x];
        for (int y = 0; y < KEYS; y++) {
            const char b = positions[y];
            for (int z = 0; z < KEYS; z++) {
                const char c = positions[z];
                total.score += sample(sharedText->cpuArray, x, y, z) * sample(metric.cpuArray, a, b, c).cost;
            }
        }
    }
    SCALESCORE(total, sharedText->scaled);
    return total;
}
#endif
