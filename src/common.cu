#include "common.cuh"

#include <metric.cuh>
#include <rng.cuh>

Snapshot::~Snapshot() {
    delete best;
    delete median;
    delete worst;
}

Snapshot::Snapshot(const pop_t size, keyboardc* best, keyboardc* median, keyboardc* worst,
                   const double average) {
    this->size = size;
    this->best = best;
    this->median = median;
    this->worst = worst;

    this->average = average;
}

Snapshot::Snapshot(const Snapshot &other) {
        this->size = other.size;
        this->average = other.average;

#ifdef LOCAL_KB
        best = new keyboard(*other->best);
        median = new keyboard(*other->median);
        worst = new keyboard(*other->worst);
#else
        best = new keyboardc(*other.best);
        if (other.median == nullptr) {
            median = nullptr;
        } else {
            median = new keyboardc(*other.median);
        }
        worst = new keyboardc(*other.worst);
        memcpy(worst->arr, other.worst->arr, KEYS);
#endif
    }

void Snapshot::add(const std::unique_ptr<Snapshot> &other) {
#define COPYKB(from, to) { \
        to->stats = from->stats; \
        memcpy(to, from, KEYS); \
    }

    average = other->size * other->average + size * average;
    size += other->size;
    average /= size;

    if (CMP(other->best->stats.score < best->stats.score)) {
        COPYKB(other->best, best);
    }
    if (CMP(other->worst->stats.score > worst->stats.score)) {
        COPYKB(other->worst, worst);
    }
    median = nullptr;
}

__global__ void initKeyboards(const pop_t n, keyboard* pointers, char* base) {
    const pop_t i = IDX(pop_t);
    if (i < n) {
        char* arr = base + i * ALIGNED;
        pointers[i].arr = arr;
        memset(arr, 0, ALIGNED);

        pointers[i].stats = {};

        pointers[i].rescore = true;
    }
}

__global__ void randomize(const pop_t size, keyboard* group) {
    const pop_t i = IDX(pop_t);
    if (i >= size) return;

    char* pos = group[i].arr;

    score_t score = group[i].stats.score;
    state rng = hash(pos, hash(i, * (state*) &score, (state) pos + (state) group + i));

    #pragma unroll
    for (int j = 0; j < KEYS; j++) {
        pos[j] = j;
    }

    #pragma unroll
    for (int j = 0; j < KEYS - 1; ++j) {
        if (!movable[j]) continue;
        const char k = j + next(rng) % (KEYS - j);
        if (!movable[k]) continue;
        SWAP(pos, j, k);
    }

    group[i].rescore = true;
}
