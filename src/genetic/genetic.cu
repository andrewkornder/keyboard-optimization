#include "genetic.cuh"

#include <kernel.cuh>
#include <metric.cuh>
#include "sort.cuh"
#include <rng.cuh>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#if true

population::population(const pop_t n, const pop_t k) {
    this->n = n;
    this->k = k;

#define SPLIT(name, v) name##block = (v + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; \
    name##thread = min(v, THREADS_PER_BLOCK); \
    // printf("Split for %s=%llu: %d x %d\n", #name, v, name##thread, name##block)
    SPLIT(n, n);
    SPLIT(k, k);

    const size_t size = (size_t) ALIGNED * (2 * n + k);

    cudaMalloc(&base, size);
    top = base + n * ALIGNED;
    aux = top + k * ALIGNED;
    cudaMalloc(&kbs, n * sizeof(keyboard));

    if (n > 0) initKeyboards<<<nblock, nthread>>>(n, kbs, base);
    cudaDeviceSynchronize();
}

population::~population() {
    cudaDeviceSynchronize();
    cudaFree(kbs);
    cudaFree(base);
}

void population::rearrange() {
    copyRange<<<copyGroups * nblock, nthread>>>(n, kbs, aux);
    reorder<<<copyGroups * nblock, nthread>>>(n, kbs, base, aux);
}


#include "duplicates.cuh"
pop_t population::duplicates() const {
    return countDuplicates(n, kbs);
}


void population::scoreAndSort() {
    score(n, kbs);
    sort(n, kbs);
    rearrange();

#ifdef REMOVE_DUPLICATES
    {
        identifyDuplicates<<<nblock, nthread>>>(n, kbs);
        removeDuplicates<<<nblock, nthread>>>(n, kbs);
    }

    score(n, kbs);
    sort(n, kbs);
    rearrange();
#endif

    copyRange<<<copyGroups * kblock, kthread>>>(k, kbs, top);
    cudaDeviceSynchronize();
}

void population::init() {
    randomize<<<nblock, nthread>>>(n, kbs);
    scoreAndSort();
}

void population::advance() {
    createNext<<<nblock, nthread>>>(n, k, kbs, top);
    scoreAndSort();
    updateState(base, 3);
}

FUNCTOR(GetScore, double, (const keyboard &in), { return in.stats.score; });

double population::averageScore() const {
    constexpr GetScore getter;
    return thrust::transform_reduce(thrust::device, kbs, kbs + n, getter, 0., thrust::plus<double>()) / n;
}

std::unique_ptr<Snapshot> population::snapshot() const {
    char* arrays;
    stats* scores;
    cudaMalloc(&arrays, 3 * KEYS);
    cudaMallocManaged(&scores, 3 * sizeof(stats));

    prepareSnapshot<<<1, 3>>>(n, kbs, arrays, scores);
    cudaDeviceSynchronize();

#define CREATE(name, i) auto* name = new keyboardc{}; \
    cudaMemcpy(name->arr, arrays + i * KEYS, KEYS, cudaMemcpyDeviceToHost); \
    name->stats = scores[i];

    CREATE(best, 0);
    CREATE(median, 1);
    CREATE(worst, 2);


    cudaFree(scores);
    cudaFree(arrays);
    return std::make_unique<Snapshot>(n, best, median, worst, averageScore());
}


__global__ void sampleDistribution(const pop_t n, const keyboard* kbs, const int samples, uint32_t* dist,
                                const score_t min, const score_t max) {
    const pop_t i = IDX(pop_t);
    if (i >= n) return;

    const double m = 1. / (max - min);
    const double t = (kbs[i].stats.score - min) * m;
    atomicInc(&dist[(int) (samples * t)], 1);
}


std::vector<uint32_t> population::getDistribution(const int samples, const score_t min, const score_t max) {
    std::vector<uint32_t> dist;
    dist.reserve(samples);

    uint32_t* dist_;
    cudaMallocManaged(&dist_, sizeof(uint32_t) * samples);

    sampleDistribution<<<nblock, nthread>>>(n, kbs, samples, dist_, min, max);
    cudaDeviceSynchronize();

    for (int i = 0; i < samples; ++i) {
        dist.push_back(dist_[i]);
    }
    cudaFree(dist_);

    return dist;
}
#endif // population
