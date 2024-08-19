#include "sort.cuh"


#include <thrust/sort.h>
#include <thrust/execution_policy.h>

FUNCTOR(Comparator, bool, (const keyboard &a, const keyboard &b), {return CMP(a.stats.score < b.stats.score);});

__host__ void sort(const pop_t n, keyboard* arr) {
    static constexpr Comparator cmp;
    thrust::sort(thrust::device, arr, arr + n, cmp);
}
