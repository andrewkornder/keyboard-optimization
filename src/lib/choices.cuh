#ifndef CHOICES_H
#define CHOICES_H
#include "constants.cuh"


// #define METRIC_CARPALX
// #define METRIC_OKP
// #define METRIC_DIST

// #define TEST_LOCKS

// #define SIMILARITY

constexpr int maxSwaps = 10;
constexpr int saveDist = 0;

#define MOV true
#define LCK false

#define MINIMIZE
// #define MAXIMIZE

constexpr int copyGroups = 4;
#define COPYMODE COPYMODE_CAST

#define SHOWOUTPUT

// #define SINGLEMUTATE
#define ORDERED_CREATION

#endif //CHOICES_H
