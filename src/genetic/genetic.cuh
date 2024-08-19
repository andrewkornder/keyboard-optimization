#ifndef GENETIC_CUH
#define GENETIC_CUH


#include <common.cuh>
#include <vector>

struct population {
    population(pop_t, pop_t);
    ~population();

    void init();
    void scoreAndSort();
    void advance();

    double averageScore() const;
    pop_t duplicates() const;
    std::unique_ptr<Snapshot> snapshot() const;
    std::vector<uint32_t> getDistribution(int samples, score_t min, score_t max);

    pop_t n, k;

    int nblock, nthread;
    int kblock, kthread;

    void rearrange();

    char* aux;
    char* base;
    char* top;

    keyboard* kbs;
};

#endif //GENETIC_CUH
