#include "record.cuh"
#include <def.cuh>
#include <filesystem>

#include <fstream>
#include <text.cuh>

constexpr int setSize = ALIGNED / sizeof(uint64_t);

Record::Record(population* g, const Config &config) :
    config(config), generations(config.generations), rounds(config.rounds),
    history(config.generations), reduced(config.generations), totalHistory(config.rounds), totalReduced(config.rounds)
{
    if (!is_directory(config.output)) {
        printf("Creating output directory: '%ls'\n", config.output.c_str());
        create_directory(config.output);
    }

    generationsRan = std::vector(rounds, 0);
    uniqueHistory = std::vector(rounds, 0);
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
        const state seed = SEED;
        std::unique_ptr<Snapshot> o = next(i);

        addToSet(o->best->arr);
        uniqueHistory[i] = unique;

        const clock_t time = clock() - last;
        last = clock();

        const double seconds = (double) time / CLOCKS_PER_SEC;
        if (i) secondsPerIt = secondsPerIt * (1 - 0.2) + seconds * 0.2;
        else secondsPerIt = seconds;

        saveToFile(i, seed, o);

        const double minutesLeft = (rounds - i - 1) * secondsPerIt / 60.;

        // const std::unique_ptr<Snapshot> &p = o;
        const std::unique_ptr<Snapshot> &p = snap;
        printf("\r[%d (%d) / %d] %s, %s, %s | %.2f s/it | %.2f minutes left", i + 1, unique, rounds,
                F3(p->best->stats.score), F3(p->average), F3(p->worst->stats.score), secondsPerIt, minutesLeft);
        for (int r = 0; r < 15; ++r) {
            printf(" ");
        }

        if (config.showOutput) {
            printf("\n");
            printArrQ(p->best->arr);
        }
    }
    saveAllToFile();
}


void Record::writeKB(std::ofstream &file, const char* positions) {
    char letters[KEYS] = {};
    for (int i = 0; i < KEYS; i++) {
        letters[positions[i]] = i;
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
    std::ofstream file(config.output / "all.gen", std::ofstream::out);
    file << config.buffer;

    file << "\nTotal organisms seen: " << snap->size << "\n";

    file << '\n';
    file << "Best organism (score = " << snap->best->stats << "):\n";
    writeKB(file, snap->best->arr);

    file << '\n';
    file << "Worst organism (score = " << snap->worst->stats << "):\n";
    writeKB(file, snap->worst->arr);

    file << "History:\n";
    file << "Best scores: ";
    writePlot(file, totalHistory.best, rounds);
    file << "Average scores: ";
    writePlot(file, totalHistory.average, rounds);
    file << "Worst scores: ";
    writePlot(file, totalHistory.worst, rounds);

    file << '\n';
    file << "Reduced:\n";
    file << "Best scores: ";
    writePlot(file, totalReduced.best, rounds);
    file << "Average scores: ";
    writePlot(file, totalReduced.average, rounds);
    file << "Worst scores: ";
    writePlot(file, totalReduced.worst, rounds);

    file << "Unique layouts seen: ";
    writePlot(file, uniqueHistory, rounds);
    file << "Length of each round: ";
    writePlot(file, generationsRan, rounds);
}


void Record::saveToFile(const int i, const state seed, const std::unique_ptr<Snapshot> &result) const {
    std::ofstream file((config.output / std::to_string(i)).string() + ".gen", std::ofstream::out);
    const int length = generationsRan[i];

    file << "Config hash: " << config.hash << '\n';
    file << "Seed: " << seed << '\n';
    file << "Round: " << i + 1 << " / " << rounds << '\n';

    file << "Generations: " << length << " x " << g->n << " organisms\n";
    file << "Total organisms: " << result->size << '\n';

    file << '\n';
    file << "Best organism (score = " << result->best->stats << "):\n";
    writeKB(file, result->best->arr);

    file << '\n';
    file << "Worst organism (score = " << result->worst->stats << "):\n";
    writeKB(file, result->worst->arr);

    file << "History:\n";
    file << "Best scores: ";
    writePlot(file, history.best, length);
    file << "Average scores: ";
    writePlot(file, history.average, length);
    file << "Median scores: ";
    writePlot(file, history.median, length);
    file << "Worst scores: ";
    writePlot(file, history.worst, length);
    file << '\n';
    file << "Reduced:\n";
    file << "Best scores: ";
    writePlot(file, reduced.best, length);
    file << "Average scores: ";
    writePlot(file, reduced.average, length);
    file << "Worst scores: ";
    writePlot(file, reduced.worst, length);
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
    explicit Distribution(const Config &config, const int index) {
        file.open((config.output / std::to_string(index)).string() + ".dist", std::ofstream::out);
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
        dist = Distribution(config, i);
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
            if (repeated++ == config.plateauLength) break;
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