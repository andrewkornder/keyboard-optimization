#ifndef CHOICES_H
#define CHOICES_H
#include "constants.cuh"


constexpr int generations = 25, rounds = 20000;
// #define METRIC_CARPALX
// #define METRIC_OKP
// #define METRIC_DIST

// #define TEST_LOCKS

// #define SIMILARITY

constexpr int maxSwaps = 10;
constexpr int saveDist = 0;
constexpr int maxRepeat = 2;

#define MOV true
#define LCK false

#define RNG64


// #define REMOVE_DUPLICATES


#define MINIMIZE
// #define MAXIMIZE

constexpr int copyGroups = 4;
#define COPYMODE COPYMODE_CAST

#define SHOWOUTPUT

// #define SINGLEMUTATE
#define ORDERED_CREATION

#endif //CHOICES_H
