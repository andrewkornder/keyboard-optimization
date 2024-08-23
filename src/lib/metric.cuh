#ifndef METRIC_CUH
#define METRIC_CUH
#include <common.cuh>
#include <memory>
#include <text.cuh>

#ifdef METRIC_OKP
struct mtype {
    short cost = 0;
    bool sameHand = false;
    bool sameFinger = false;
    bool inRoll = false;
    bool outRoll = false;
    bool rowChange = false;
    bool ringJump = false;
    bool toCenter = false;
    bool homeJump = false;
    bool doubleJump = false;
};

inline std::ostream& operator<<(std::ostream& file, const mtype &m) {
    file << '{';
    file << (int) m.cost << ',';
    file << (int) m.sameHand << ',';
    file << (int) m.sameFinger << ',';
    file << (int) m.inRoll << ',';
    file << (int) m.outRoll << ',';
    file << (int) m.rowChange << ',';
    file << (int) m.ringJump << ',';
    file << (int) m.toCenter << ',';
    file << (int) m.homeJump << ',';
    file << (int) m.doubleJump;
    file << '}';
    return file;
}
#else
struct mtype {
    score_t cost;
};

inline std::ostream& operator<<(std::ostream& file, const mtype &m) {
    file << '{' << m.cost << '}';
    return file;
}
#endif

const mtype* precomputeMetric(const std::shared_ptr<FinishedText>& text);
__host__ void score(pop_t n, keyboard* population);
__host__ stats score(const char* positions);

#endif //METRIC_CUH
