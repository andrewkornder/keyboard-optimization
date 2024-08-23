#ifndef RECORD_CUH
#define RECORD_CUH

#include <rng.cuh>
#include <general.cuh>

template <typename T, int Size, bool hasMedian>
struct History {
    History() {
        best.reserve(Size);
        average.reserve(Size);
        if (hasMedian) median.reserve(Size);
        worst.reserve(Size);
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

    explicit Record(population* g);

    void get();

    int unique = 0;
    std::vector<int> uniqueHistory = std::vector<int>(rounds);
    std::vector<int> generationsRan = std::vector<int>(rounds);
    std::vector<uint64_t> set;

    population* g;

    History<stats, generations, true> history;
    History<stats, generations, false> reduced;
    History<stats, rounds, false> totalHistory, totalReduced;

    std::unique_ptr<Snapshot> snap = nullptr;

private:
    void addToSet(const char* positions);

    static void writeKB(std::ofstream &file, const char* positions);

    void saveAllToFile() const;

    void saveToFile(int i, state seed, const std::unique_ptr<Snapshot> &result) const;

    template<int Size, bool median>
    static void add(const std::unique_ptr<Snapshot> &snap, History<stats, Size, median> &arr);

    std::unique_ptr<Snapshot> next(int i);
};

#endif //RECORD_CUH
