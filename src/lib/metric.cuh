#ifndef METRIC_CUH
#define METRIC_CUH
#include <common.cuh>
#include <memory>
#include <text.cuh>


void precomputeMetric(const std::shared_ptr<FinishedText<textWindow>>& text);
__host__ void score(pop_t n, keyboard* population);
__host__ stats score(const char* positions);

#endif //METRIC_CUH
