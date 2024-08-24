#include "perf.cuh"

#include "kernel.cuh"
#include <def.cuh>
#include <duplicates.cuh>
#include <metric.cuh>
#include <rng.cuh>

#include "sort.cuh"
#include <string>


#define BODY(...) {__VA_ARGS__}

#define MAXIT 1000
#define PERFSTART(seconds) { \
        constexpr double budget = CLOCKS_PER_SEC * seconds; \
        int Pn = 0; \
        PerfResult _perf[20];
#define PERFEND printResults(Pn, _perf); \
        printf("\n"); \
    }
#define PROFg(name) PROF_(group.name(), name)
#define PROFc(name) PROF_(name(), name)
#define PROF_(body, title) \
    double title##Time; \
    { \
        body; \
        printf("\r[%s] %-6d %.1f / %.1f seconds      ", #title, 0, 0., budget / CLOCKS_PER_SEC); \
        int cnt = 0; \
        double start = (double) clock(); \
        while (cnt < MAXIT) { \
            int elapsed = clock() - start; \
            if (elapsed > budget) break; \
            body; \
            cudaDeviceSynchronize(); \
            printf("\r[%s] %-6d %.1f / %.1f seconds      ", #title, ++cnt, \
                    (double) elapsed / CLOCKS_PER_SEC, budget / CLOCKS_PER_SEC); \
        } \
        title##Time = (clock() - start) / ((double) cnt); \
        _perf[Pn++] = {#title, title##Time, cnt}; \
    }

struct PerfResult {
    std::string name;
    double time;
    int it;
};

__global__ void doNothingGPU_() {}
void doNothingGPU() {doNothingGPU_<<<1, 1>>>(); cudaDeviceSynchronize();}

void printResults(const int n, PerfResult* res) {
    printf("\r                                       \r");
    for (int i = 0; i < n; i++) {
        int b = 0;
        for (int j = 0; j < n; j++) {
            if (res[j].time > res[b].time) {
                b = j;
            }
        }
        printf("%-17s %8.3f ms x %-5d (%.2f s)\n",
            res[b].name.c_str(), res[b].time, res[b].it, res[b].time * res[b].it / CLOCKS_PER_SEC);
        res[b].time = -1;
    }
}

void runPerfMethods(population &group) {
    {
        PERFSTART(0.5)
            PROF_(cudaDeviceSynchronize(), doNothing);
            PROFc(doNothingGPU);
        PERFEND
    }

    PERFSTART(2)
        PROF_(BODY(
            initKeyboards<<<group.nblock, group.nthread>>>(group.n, group.kbs, group.base);
        ), setup);
        PROF_(BODY(
            initKeyboards<<<group.nblock, group.nthread>>>(group.n, group.kbs, group.base);
            score(group.n, group.kbs);
        ), score);
    PERFEND

    {
        PERFSTART(1)
            PROF_(BODY(
                identifyDuplicates<<<group.nblock, group.nthread>>>(group.n, group.kbs);
                removeDuplicates<<<group.nblock, group.nthread>>>(group.n, group.kbs);
            ), removeDuplicates);
            PROF_(BODY(
                score(group.n, group.kbs);
                sort(group.n, group.kbs);
                group.rearrange();
            ), SSBody)
        PERFEND
    }

    {
        PERFSTART(1.5)
            PROF_(BODY(
                sort(group.n, group.kbs);
            ), sort);
            PROF_(BODY(
                randomize<<<group.nblock, group.nthread>>>(group.n, group.kbs);
            ), randomize);
            PROF_(BODY(
                createNext<<<group.nblock, group.nthread>>>(group.n, group.k, group.kbs, group.top);
            ), createNext);
        PERFEND

        PERFSTART(1.5)
            PROFg(averageScore);
#ifdef REARRANGE
            PROFg(rearrange);
#endif
            PROFg(scoreAndSort);
        PERFEND
    }

    {
        PERFSTART(1)
            PROF_(BODY(
                copyRange<<<copyGroups * group.nblock, group.nthread>>>(group.n, group.kbs, group.aux);
            ), copyAll);
            PROF_(BODY(
                copyRange<<<copyGroups * group.kblock, group.kthread>>>(group.k, group.kbs, group.top);
            ), copyTop);
        PERFEND
    }

    #ifndef LOCAL_KB
    {
        PERFSTART(1)
            PROF_(BODY(
                initKeyboards<<<group.nblock, group.nthread>>>(group.n, group.kbs, group.base);
            ), alloc);
        PERFEND
    }
    #endif
    printf("\n");
}

void runPerfRound(population &group, const int generations, const int rounds) {
    SEED = 0x69048868cdaf6565ULL;
    initKeyboards<<<group.nblock, group.nthread>>>(group.n, group.kbs, group.base);
    cudaDeviceSynchronize();

    for (int i = 0; i < 5; i++) {
        if (i) group.advance();
        else group.init();

        const std::unique_ptr<Snapshot> snap = group.snapshot();
        printf("[%d] %s, %s, %s, %s\n", i + 1, F3(snap->best->stats.score), F3(snap->median->stats.score),
                F3(snap->average), F3(snap->worst->stats.score));
        // printArrQ(snap->best->arr);
    }
    printf("\n");

    {
        double ticksPerRound = 0;

        PERFSTART(2)
            PROFg(init);
            PROFg(advance);
            PROFg(snapshot);

            ticksPerRound += initTime;
            ticksPerRound += advanceTime * (generations - 1);
            ticksPerRound += snapshotTime * generations;
        PERFEND

        printf("%.2f ms per round (%.2f ms per generation)\n", ticksPerRound, ticksPerRound / generations);

        const double seconds = ticksPerRound * rounds / CLOCKS_PER_SEC;
        const int minutes = (int) (seconds / 60);
        const int hours = minutes / 60;
        printf("%d hours, %d minutes and %.1f seconds\n\n", hours, minutes % 60, seconds - 60 * minutes);

        const double kbsPerSecond = group.n * generations * CLOCKS_PER_SEC / ticksPerRound;
        printf("%s keyboards/second\n", F3((pop_t) kbsPerSecond));
    }
}
