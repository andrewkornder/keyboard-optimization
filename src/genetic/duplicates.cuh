#ifndef DUPLICATES_CUH
#define DUPLICATES_CUH
#include <common.cuh>


#ifdef REMOVE_DUPLICATES
__global__ void identifyDuplicates(pop_t n, keyboard* kbs);
__global__ void removeDuplicates(pop_t n, const keyboard* kbs);
__global__ void removeDuplicatesAIO(pop_t n, keyboard* kbs);
#endif
pop_t countDuplicates(pop_t n, const keyboard* kbs);

#endif //DUPLICATES_CUH
