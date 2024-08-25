#include "record.cuh"
#include <def.cuh>
#include <filesystem>

#include <fstream>
#include <sstream>
#include <text.cuh>

constexpr int setSize = ALIGNED / sizeof(uint64_t);

Record::Record(population* g, const int generations, const int rounds) :
    generations(generations), rounds(rounds), history(generations),
    reduced(generations), totalHistory(rounds), totalReduced(rounds)
{
    if (!is_directory(output)) {
        printf("Creating output directory: '%ls'\n", output.c_str());
        create_directory(output);
    }
    this->g = g;
}

void Record::addToSet(const char* positions) {
    const auto* cast = (uint64_t*) positions;

    int offset = 0;
    for (int i = 0; i < unique; ++i) {
        bool equal = true;
        for (int j = 0; j < setSize; ++j) {
            if (cast[j] != set[j + offset]) {
                equal = false;
                break;
            }
        }
        if (equal) return;

        offset += setSize;
    }

    ++unique;
    for (int i = 0; i < setSize; ++i) {
        set.push_back(cast[i]);

    }
}

void Record::get() {
    double secondsPerIt = 0;
    int last = clock();
    for (int i = 0; i < rounds; ++i) {
        std::unique_ptr<Snapshot> o = next(i);

        addToSet(o->best->arr);
        uniqueHistory[i] = unique;

        const clock_t time = clock() - last;
        last = clock();

        const double seconds = (double) time / CLOCKS_PER_SEC;
        if (i) secondsPerIt = secondsPerIt * (1 - 0.2) + seconds * 0.2;
        else secondsPerIt = seconds;

        saveToFile(i, SEED, o);

        const double minutesLeft = (rounds - i - 1) * secondsPerIt / 60.;

        // const std::unique_ptr<Snapshot> &p = o;
        const std::unique_ptr<Snapshot> &p = snap;
        printf("\r[%d (%d) / %d] %s, %s, %s | %.2f s/it | %.2f minutes left", i + 1, unique, rounds,
                F3(p->best->stats.score), F3(p->average), F3(p->worst->stats.score), secondsPerIt, minutesLeft);
        for (int r = 0; r < 35; ++r) {
            printf(" ");
        }

#ifdef SHOWOUTPUT
        printf("\n");
        printArrQ(p->best->arr);
#endif
    }
    saveAllToFile();
}


void Record::writeKB(std::ofstream &file, const char* positions) {
    char letters[KEYS] = {};
    for (int i = 0; i < KEYS; i++) {
        letters[positions[i]] = i;
        const char p = positions[i];
        if (p < 10) file << "0";
        file << (int) p << ",";
        if (i % 10 == 9) file << '\n';
    }
    file << '\n';
    for (int i = 0; i < KEYS; i++) {
        file << KEYS_LOWER[letters[i]];
        if (i % 10 == 9) file << '\n';
    }
    file << '\n';
}

template <typename T>
void writePlot(std::ofstream &file, const std::vector<T> &y, const int size) {
    file << '[';
    for (int i = 0; i < size; i++) {
        if (i) file << ',';
        file << y[i];
    }
    file << "]\n";
}


void Record::saveAllToFile() const {
    std::ofstream file(output / "all.gen", std::ofstream::out);

    file << "Total organisms seen: " << snap->size << "\n\n";

    file << "\nBest organism (" << snap->best->stats << "):\n";
    writeKB(file, snap->best->arr);

    file << "\nWorst organism (" << snap->worst->stats << "):\n";
    writeKB(file, snap->worst->arr);

    file << "History:\n";
    writePlot(file, totalHistory.best, rounds);
    writePlot(file, totalHistory.average, rounds);
    writePlot(file, totalHistory.worst, rounds);
    file << '\n';
    writePlot(file, totalReduced.best, rounds);
    writePlot(file, totalReduced.average, rounds);
    writePlot(file, totalReduced.worst, rounds);
    file << '\n';
    writePlot(file, uniqueHistory, rounds);
    file << '\n';
    writePlot(file, generationsRan, rounds);

    file.flush();
    file.close();
}

void Record::saveToFile(const int i, const state seed, const std::unique_ptr<Snapshot> &result) const {
    std::ofstream file((output / std::to_string(i)).string() + ".gen", std::ofstream::out);

    file << "SEED: " << seed << "\n";
    file << "Run: " << i + 1 << " / " << rounds;
    file << "\nGenerations: " << generationsRan[i] << " x " << result->size << " organisms\n\n";

    file << "\nBest organism (" << result->best->stats << "):\n";
    writeKB(file, result->best->arr);

    file << "\nWorst organism (" << result->worst->stats << "):\n";
    writeKB(file, result->worst->arr);

    file << "History:\n";
    writePlot(file, history.best, generationsRan[i]);
    writePlot(file, history.average, generationsRan[i]);
    writePlot(file, history.median, generationsRan[i]);
    writePlot(file, history.worst, generationsRan[i]);
    file << '\n';
    writePlot(file, reduced.best, generationsRan[i]);
    writePlot(file, reduced.average, generationsRan[i]);
    writePlot(file, reduced.worst, generationsRan[i]);

    file.flush();
    file.close();
}

template<bool median>
void Record::add(const std::unique_ptr<Snapshot> &snap, History<stats, median> &arr) {
    arr.best.push_back(snap->best->stats);
    arr.average.push_back(snap->average);
    if (median) arr.median.push_back(snap->median->stats);
    arr.worst.push_back(snap->worst->stats);
}

class Distribution {
    std::ofstream file{};

public:
    Distribution() {}
    explicit Distribution(const int index) {
        file.open((Record::output / std::to_string(index)).string() + ".dist", std::ofstream::out);
    }

    void add(population &g, const std::unique_ptr<Snapshot> &snapshot) {
        if (!file.is_open()) return;

        const double min = snapshot->best->stats.score;
        const double max = snapshot->worst->stats.score;
        add(g.getDistribution(1000, min, max), min, max);
    }

    void add(const std::vector<uint32_t> &dist, const score_t min, const score_t max) {
        if (!file.is_open()) return;

        file << '[' << min << ',' << max << "] = [";

        bool first = true;
        for (const uint32_t samples : dist) {
            if (!first) file << ',';
            first = false;
            file << samples;
        }
        file << "]\n";
    }
};


std::unique_ptr<Snapshot> Record::next(const int i) {
    Distribution dist;
    if (i < saveDist) {
        dist = Distribution(i);
    }

    g->init();
    std::unique_ptr<Snapshot> total = g->snapshot();
    dist.add(*g, total);

    add(total, history);
    add(total, reduced);

    score_t lastScore = -1;
    int repeated = 0, gen;
    for (gen = 1; gen < generations; ++gen) {
        g->advance();
        std::unique_ptr<Snapshot> curr = g->snapshot();
        dist.add(*g, curr);

        if (total->best->stats.score == lastScore) {
            if (++repeated == plateauLength) break;
        } else repeated = 0;

        lastScore = total->best->stats.score;

        total->add(curr);

        add(curr, history);
        add(total, reduced);
    }
    generationsRan[i] = gen;

    if (snap == nullptr) snap = std::make_unique<Snapshot>(*total);
    else snap->add(total);

    add(total, totalHistory);
    add(snap, totalReduced);

    return total;
}

int Record::plateauLength = 2;
std::filesystem::path Record::output = "output";