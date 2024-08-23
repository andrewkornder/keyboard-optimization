#include <filesystem>
#include <md5.cuh>
#include <text.cuh>
#include <metric.cuh>

#include "test.cuh"
#include "record.cuh"
#include "general.cuh"


struct Config {
private:
    static uint64_t parseNum(const std::string &value) {
        uint64_t x = 0;
        for (const char p : value) {
            if ('0' <= p && p <= '9') {
                x = 10 * x + (p - '0');
            }
        }
        return x;
    }
    static uint64_t parseHex(const std::string &value) {
        uint64_t x = 0;
        for (const char p : value) {
            if ('0' <= p && p <= '9') {
                x = 0x10 * x + (p - '0');
            } else if ('a' <= p && p <= 'f') {
                x = 0x10 * x + (p - 'a' + 10);
            }
        }
        return x;
    }
    static bool getKey(std::ifstream &cnf, char* line, const bool silent) {
        cnf.get(line, 256, '\n');
        cnf.get();
        if (line[0] == '#') return false;

        if (line[0] != '<') {
            if (!silent) printf("Tag did not start with '<': '%s'\n", line);
            return false;
        }

        char* ptr = line;
        while (ptr[1] != '\0') ++ptr;
        if (*ptr != '>') {
            if (!silent) printf("Key not enclosed with with '<' & '>': '%s'\n", line);
            return false;
        }
        *ptr = '\0';
        return true;
    }

    bool match(std::ifstream &cnf) {
        char line[256];
        if (!getKey(cnf, line, false)) return false;

        const std::string key = line + 1;

        std::string value;
        char buf[256] = {};
        while (!getKey(cnf, buf, true)) {
            value.append(buf);
            value.push_back('\n');
        }
        value.pop_back();

        if (buf[1] != '/') {
            printf("No ending tag for key '%s': '%s'\n", key.c_str(), buf);
            return false;
        }

        if (key != buf + 2) {
            printf("Keys do not match '%s' != '%s'\n", key.c_str(), buf + 2);
            return false;
        }

        if (key == "movable") {
            std::vector<int> lmov;
            lmov.reserve(KEYS);

            for (const char c : value) {
                if (c == '0' || c == '1') {
                    lmov.push_back(c != '0');
                }
            }
            if (lmov.size() == KEYS) {
                bool arr[KEYS];
                for (int i = 0; i < KEYS; ++i) {
                    arr[i] = lmov[i];
                }
                cudaMemcpyToSymbol(movable, arr, sizeof(arr), 0, cudaMemcpyHostToDevice);
                hasSetMovable = true;
            } else {
                printf("Movable mask is the wrong size.\nExpected: %d\nFound: %lld.\n", KEYS, lmov.size());
            }
        } else if (key == "size") {
            size.pop = roundToPower(parseNum(value));
        } else if (key == "survivors") {
            size.surviving = roundToPower(parseNum(value));
        } else if (key == "cache") {
            cacheDirectory = value;
        } else if (key == "generations") {
            generations = parseNum(value);
        } else if (key == "plateauLength") {
            Record::plateauLength = parseNum(value);
        } else if (key == "rounds") {
            rounds = parseNum(value);
        } else if (key == "seed") {
            if (value[0] == '0' && value[1] == 'x') {
                SEED = parseHex(value);
            } else {
                SEED = parseNum(value);
            }
            hasSetSeed = true;
        } else if (key == "output") {
            Record::output = value;
        } else if (key == "corpus") {
            corpus = value;
        } else if (key == "exportTables") {
            exportTables = value;
        } else {
            printf("Unknown key: '%s'\n", key.c_str());
            return false;
        }
        return true;
    }

    std::filesystem::path exportTables;
    std::filesystem::path corpus;
    bool hasSetMovable = false;
    bool hasSetSeed = false;
    std::vector<std::string> textExtensions{};

public:
    explicit Config(const char* path) {
        std::ifstream cnf(path);

        if (!cnf.is_open()) {
            printf("Config could not be opened: '%s'\n", path);
            return;
        }

        while (true) {
            match(cnf);
            while (cnf.get() != '<' && !cnf.eof()) {
                cnf.seekg(-1, std::ifstream::cur);
                cnf.ignore(64, '\n');
            }
            if (cnf.eof()) break;
            cnf.seekg(-1, std::ifstream::cur);
        }

        if (corpus.empty()) {
            printf("Required field 'corpus' not found.\n");
            exit(1);
        }

        if (!is_directory(cacheDirectory)) {
            printf("Creating cache directory.\n");
            create_directory(cacheDirectory);
        }

        if (!hasSetMovable) {
            bool lmov[KEYS];
            memset(lmov, true, KEYS);
            cudaMemcpyToSymbol(movable, lmov, sizeof(lmov), 0, cudaMemcpyHostToDevice);
        }

        int totalMovable = 0;
        {
            bool mov[KEYS];
            cudaMemcpyFromSymbol(mov, movable, sizeof(bool) * KEYS, 0, cudaMemcpyDeviceToHost);
            for (int i = 0; i < KEYS; ++i) {
                totalMovable += mov[i];
            }
        }

        const std::shared_ptr text = initText(corpus);
        const mtype* metric = precomputeMetric(text);

        if (!hasSetSeed) {
            MD5 seed;
            seed.update((uint64_t) time(nullptr));
            seed.update((uint64_t) text.get());
            seed.update((uint64_t) clock());
            seed.update(0x7f18ee808626fcb9ULL);
            MD5::Digest hash;
            seed.checksum(hash);

            SEED = 0;
            uint8_t* bytes = (uint8_t*) &SEED;

            for (int i = 0; i < MD5::hashSize; ++i) {
                bytes[i % sizeof(SEED)] ^= hash[i];
            }
        }

        if (!exportTables.empty()) {
            {
                std::ofstream file(exportTables / "text.chc");

                const text_t* array = text->cpuArray;

                std::vector<uint64_t> values;
                values.reserve(FinishedText::N);

                for (int i = 0; i < FinishedText::N; ++i) {
                    values.push_back(i);
                }

                std::sort(values.begin(), values.end(), [array](const uint64_t a, const uint64_t b) {
                    return array[a] > array[b];
                });

                for (const uint64_t key : values) {
                    int letters[textWindow];
                    letterUtils.getCharsAtIndex<textWindow, KEYS>(key, letters);
                    for (const int q : letters) {
                        file << (q == -1 ? ' ' : KEYS_LOWER[q]);
                    }
                    file << ": " << array[key] << '\n';
                }
            }
            {
                std::ofstream file(exportTables / "metric.chc");
                letterUtils.applyOnIndices<textWindow, KEYS>([&file, metric](const int index, const char keys[textWindow]) {
                    for (int i = 0; i < textWindow; ++i) {
                        file << (keys[i] == -1 ? ' ' : KEYS_LOWER[keys[i]]);
                    }
                    file << '|' << metric[index].cost << '\n';
                });
            }
        }

        printf("\rConfig: {                                       ");
        printf("\n    text:            %s chars", F3(text->count));
        printf("\n    movable keys:    %d / %d chars", totalMovable, KEYS);
        printf("\n    population:      %s / %s", F3(size.surviving), F3(size.pop));
        printf("\n    cache:           %ls", cacheDirectory.c_str());
        printf("\n    output:          %ls", Record::output.c_str());
        if (!exportTables.empty()) printf("\n    exported tables: %ls", exportTables.c_str());
        printf("\n    evolution:       %s rounds of %s (plateau length = %s)", F3(rounds), F3(generations), F3(Record::plateauLength));
        printf("\n    seed:            %llu", SEED);
        printf("\n}");
        printf("\n");
    }

    struct {
        pop_t pop = 1 << 22;
        pop_t surviving = 1 << 10;
    } size;
    int generations = 40;
    int rounds = 1000;
};

std::unique_ptr<Snapshot> getLockedScore(population &group, const bool mv[KEYS], const int tries = 5) {
    cudaMemcpyToSymbol(movable, mv, sizeof(bool) * KEYS, 0, cudaMemcpyHostToDevice);

    std::unique_ptr<Snapshot> all = nullptr;
    for (int try_ = 0; try_ < tries; ++try_) {
        printf("%d...\r", try_);
        group.init();
        for (int j = 0; j < 25; ++j) {
            group.advance();
        }
        if (all == nullptr) {
            all = group.snapshot();
        } else {
            std::unique_ptr<Snapshot> s = group.snapshot();
            all->add(s);
        }
    }
    return all;
}

// todo: dont rearrange but just move pointer for kbs which will be overwritten anyway
// todo: multiple scored in single kernel pass

enum class Mode {
    Perf = 1,
    Lock = 2,
    Evolve = 4,
    TestNew = 8, TestOther = 16, TestAll = 32,
    Random = 64, QWERTY = 128,
    Help = 256,

    None = 0,
};
#define DEFINE_BITWISE2(symbol) \
Mode operator##symbol(const Mode a, const Mode b) { \
return (Mode) ((int) a symbol (int) b); \
}
#define DEFINE_BITWISE1(symbol) \
Mode operator##symbol(const Mode a) { \
    return (Mode) (symbol(int) a); \
}
DEFINE_BITWISE2(&)
DEFINE_BITWISE2(|)
DEFINE_BITWISE2(^)
DEFINE_BITWISE1(~)

class ArgParser {
    static Mode getFlag(const std::string &word) {
        if (word == "perf"       ) return Mode::Perf;
        if (word == "lock"       ) return Mode::Lock;
        if (word == "evolve"     ) return Mode::Evolve;
        if (word == "test-new"   ) return Mode::TestNew;
        if (word == "test-common") return Mode::TestOther;
        if (word == "test-all"   ) return Mode::TestAll;
        if (word == "rand"       ) return Mode::Random;
        if (word == "qwerty"     ) return Mode::QWERTY;
        if (word == "help"       ) return Mode::Help;
        printf("invalid mode: '%s'\n", word.c_str());
        return Mode::Help;
    }

    Mode all = Mode::None;
public:
    std::vector<std::pair<Mode, int>> modes{};

    explicit ArgParser(const char* argv[], const int argc) {
        for (int i = 0; i < argc; ++i) {
            Mode mode = getFlag(argv[i]);
            int rep = 1;
            if (mode != Mode::None && i != argc - 1) {
                bool number = true;
                int x = 0;
                for (const char* ptr = argv[i + 1]; *ptr != '\0'; ptr++) {
                    if ('0' <= *ptr && *ptr <= '9') {
                        x = 10 * x + (*ptr - '0');
                    } else {
                        number = false;
                        break;
                    }
                }
                if (number) {
                    rep = x;
                    ++i;
                }
            }
            modes.push_back(std::pair{mode, rep});
            all = all | mode;
        }
    }
    bool hasArgs() const { return modes.size() > 0; }
    bool hasMode(const Mode m) const {
        return (all & m) != Mode::None;
    }
};

void run(const Config &cnf, const ArgParser &args) {
    pop_t P = 0, K = 0;
    if (args.hasMode(Mode::Evolve | Mode::Lock | Mode::Perf | Mode::Random)) {
        P = cnf.size.pop;
        K = cnf.size.surviving;
        {
            constexpr size_t gibibytes = 4;
            constexpr size_t heap = gibibytes << 30;
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap);
        }
    }

    population group(P, K);

    for (const auto [mode, repetitions]: args.modes) {
        for (int i = 0; i < repetitions; ++i) {
            switch (mode) {
                case Mode::TestNew: {
                    testNew();
                    break;
                }
                case Mode::TestOther: {
                    testOther();
                    break;
                }
                case Mode::TestAll: {
                    testAll();
                    break;
                }
                case Mode::QWERTY: {
                    testQWERTY();
                    break;
                }
                case Mode::Perf: {
                    runPerfMethods(group);
                    runPerfRound(group, generations, rounds);
                    break;
                }
                case Mode::Random: {
                    group.init();
                    group.scoreAndSort();
                    printf("Average score for %s randomized organisms: %s\n", F3(group.n), F3p(group.averageScore(), 4));
                    break;
                }
                case Mode::Lock: {
                    bool movableSend[KEYS] = {};
                    memset(movableSend, MOV, KEYS);

                    score_t best, worst;
                    {
                        const std::unique_ptr<Snapshot> base = getLockedScore(group, movableSend, 30);
                        best = base->best->stats.score, worst = base->worst->stats.score;
                    }

                    for (int i = 0; i < KEYS; ++i) {
                        memset(movableSend, MOV, KEYS);
                        movableSend[i] = LCK;

                        const std::unique_ptr<Snapshot> s = getLockedScore(group, movableSend, 5);
                        printf("Locked %d: [%f, %f]\n", i, s->best->stats.score / best, s->worst->stats.score / worst);
                    }
                    break;
                }
                case Mode::Evolve: {
                    Record rec(&group);
                    rec.get();
                    break;
                }
                case Mode::Help:
                case Mode::None:
                    break;
            }
        }
    }
}


void printHelpMessage() {
    printf("Valid modes are:\n");
    printf(" - \"help\": show this message\n");
    printf(" - \"perf\": profile functions in the code\n");
    printf(" - \"rand\": compute the average score of a random keyboard\n");
    printf(" - \"qwerty\": compute the score of QWERTY\n");
    printf(" - \"evolve\": evolve a new keyboard\n");
    printf(" - \"lock\": compute the cost of leaving each single key in its place\n");
    printf(" - \"test-new\": test each new keyboard\n");
    printf(" - \"test-common\": test a few alternative keyboards and QWERTY)\n");
    printf(" - \"test-all\": run both other testing suites\n");
    printf("Any mode can be repeated multiple times by following it with a number (e.g. \"rand 100\" runs \"rand\" 100 times)\n");
    printf("\n");
    printf("Configurable settings include:\n");
    printf(" - \"size\": how many keyboards per generation.\n");
    printf(" - \"survivors\": how many of those keyboards are used to create the next generation.\n");
    printf(" - \"cache\": a directory to place cached corpora in. if not included, turns off caching.\n");
    printf(" - \"generations\": how many generations to run until finishing the evolution.\n");
    printf(" - \"plateauLength\": after this many generations without improvement, stop evolving.\n");
    printf(" - \"rounds\": when using the \"evolve\" mode, decides how many keyboards to generate.\n");
    printf(" - \"output\": when using the \"evolve\" mode, decides where to put generated keyboard files.\n");
    printf(" - \"seed\": the seed which drives the pRNG of the program. if not included, it is random. (can be written in hex)\n");
    printf(" - \"exportTables\": if provided, creates two files with the corpus and metric used.\n");
    printf(" - \"corpus\": a folder to read text from. all files and subfolders will be read and\n");
    printf("               used to train the keyboards.\n");
    printf(" - \"movable\": which keys are allowed to be modified on the QWERTY keyboard.\n");
    printf("                this should be 30 ones or zeros (1 = movable, 0 = left in place).\n");
    printf("\n");
}


int main(const int argc, const char* argv[]) {
    if (argc < 2) {  // executable config ...
        printf("Invalid number of arguments. Expected >= 1, Got: %d\n", argc - 1);
        return 1;
    }

    {
        const ArgParser args(argv + 2, argc - 2);
        if (args.hasMode(Mode::Help)) {
            printHelpMessage();
            printf("Press any key to continue...");
            system("pause 1>NUL");
            system("cls");
        }

        // if (!args.hasArgs()) { printf("No valid modes supplied.\n"); return 0; }

        const Config cnf(argv[1]);

        run(cnf, args);
    }

    // PAUSE("Waiting...");
    printf("\nFinished successfully. Press any key to exit.");
    system("pause 1>NUL");
    return 0;
}
