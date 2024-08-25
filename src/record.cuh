#ifndef RECORD_CUH
#define RECORD_CUH

#include <rng.cuh>
#include <general.cuh>

template <typename T, bool hasMedian>
struct History {
    explicit History(const int size) {
        best.reserve(size);
        average.reserve(size);
        if (hasMedian) median.reserve(size);
        worst.reserve(size);
    }
    std::vector<T> best{};
    std::vector<double> average{};
    std::vector<T> median{};
    std::vector<T> worst{};
};

class Record {
public:
    static int plateauLength;
    static std::filesystem::path output;

    explicit Record(population* g, int generations, int rounds);

    void get();

    int generations, rounds;

    int unique = 0;
    std::vector<int> uniqueHistory;
    std::vector<int> generationsRan;
    std::vector<uint64_t> set;

    population* g;

    History<stats, true> history;
    History<stats, false> reduced;
    History<stats, false> totalHistory, totalReduced;

    std::unique_ptr<Snapshot> snap = nullptr;

private:
    void addToSet(const char* positions);

    static void writeKB(std::ofstream &file, const char* positions);

    void saveAllToFile() const;

    void saveToFile(int i, state seed, const std::unique_ptr<Snapshot> &result) const;

    template<bool median>
    static void add(const std::unique_ptr<Snapshot> &snap, History<stats, median> &arr);

    std::unique_ptr<Snapshot> next(int i);
};

#endif //RECORD_CUH
