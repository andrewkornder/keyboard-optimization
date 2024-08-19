#include <filesystem>
#include <text.cuh>
#include <metric.cuh>

#include "test.cuh"
#include "record.cuh"
#include "general.cuh"


struct Config {
private:
    static uint64_t parseNum(const char* value) {
        char num[20] = {};
        int j = 0;
        for (const char* ptr = value; *ptr != '\n' && *ptr != '\0'; ++ptr) {
            if ('0' <= *ptr && *ptr <= '9') {
                num[j++] = *ptr;
            }
        }
        num[j] = '\0';
        return atoll(num);
    }
    static uint64_t parseHex(const char* value) {
        uint64_t x = 0;
        for (const char* ptr = value; *ptr != '\n' && *ptr != '\0'; ++ptr) {
            if (const char p = *ptr; '0' <= p && p <= '9') {
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

        char value[1024] = {};
        char buf[256] = {};
        char* wh = value;
        while (!getKey(cnf, buf, true)) {
            strcpy(wh, buf);
            wh += strlen(buf) + 1;
            *(wh - 1) = '\n';
        }
        if (buf[1] != '/') {
            printf("No ending tag for key '%s': '%s'\n", key.c_str(), buf);
            return false;
        }

        if (key != buf + 2) {
            printf("Keys do not match '%s' != '%s'\n", key.c_str(), buf + 2);
            return false;
        }

        if (key == "movable") {
            std::vector<int> lmov{};

            for (int j = 0; value[j] != '\0'; j++) {
                if (value[j] == '0' || value[j] == '1') {
                    lmov.push_back(value[j] != '0');
                }
            }
            if (lmov.size() == KEYS) {
                bool arr[KEYS] = {};
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
            const char* ptr = value;
            int i = 0;

            for (; ptr[i] != '\n'; ++i) {
                cachedIndexFile[i] = ptr[i];
                cacheDirectory[i] = ptr[i];
            }
            if (cacheDirectory[i] != '\\') {
                cachedIndexFile[i] = '\\';
            }

            cacheDirectory[++i] = '\0';
            strcpy(cachedIndexFile + i, "index");
            cachedText = true;
        } else if (key == "generations") {
            generations = parseNum(value);
        } else if (key == "plateauLength") {
            Record::plateauLength = parseNum(value);
        } else if (key == "rounds") {
            rounds = parseNum(value);
        } else if (key == "seed") {
            if (value[0] == '0' && value[1] == 'x') {
                SEED = parseHex(value + 2);
            } else {
                SEED = parseNum(value);
            }
            hasSetSeed = true;
        } else if (key == "corpus") {
            const char* ptr = value;
            int i = 0;
            for (; ptr[i] != '\n'; ++i) {
                corpus[i] = ptr[i];
            }
            corpus[i++] = '\0';

            char ext[32] = {};
            char* tmp = ext;

            for (; ptr[i] != '\0'; ++i) {
                if (ptr[i] == '\n') {
                    *tmp = '\0';
                    textExtensions.push_back(std::string(ext));
                    tmp = ext;
                    if (ptr[i + 1] != '.') {
                        *tmp++ = '.';
                    }
                } else {
                    *tmp++ = ptr[i];
                }
            }
        } else {
            printf("Unknown key: '%s'\n", key.c_str());
            return false;
        }
        return true;
    }

    bool hasSetMovable = false;
    bool hasSetSeed = false;
    char corpus[1024] = {};
    std::vector<std::string> textExtensions{};

public:
    explicit Config(const char* path) {
        printf("Loading config from file '%s'\n", path);
        std::ifstream cnf(path);

        if (!cnf.is_open()) {
            printf("Config could not be opened.\n");
            return;
        }

        while (true) {
            match(cnf);
            cnf.ignore(64, '<');
            if (cnf.eof()) break;

            cnf.seekg(-1, std::ifstream::cur);
        }

        if (!std::filesystem::is_directory(cacheDirectory)) {
            printf("Creating cache directory.\n");
            std::filesystem::create_directory(cacheDirectory);
        }

        if (!std::filesystem::exists(cachedIndexFile)) {
            printf("Creating cache index file.\n");
            std::ofstream file(cachedIndexFile);
        }

        if (!hasSetSeed) {
            SEED = time(nullptr) * 0x7f18ee808626fcb9ULL;
        }

        if (corpus[0] == '\0') {
            printf("Required field 'corpus' not found.\n");
            exit(1);
        }

        if (!hasSetMovable) {
            bool lmov[KEYS] = {};
            for (int i = 0; i < KEYS; ++i) {
                lmov[i] = true;
            }
            cudaMemcpyToSymbol(movable, lmov, sizeof(bool) * KEYS, 0, cudaMemcpyHostToDevice);
        }

        int totalMovable = 0;
        {
            bool mov[KEYS];
            cudaMemcpyFromSymbol(mov, movable, sizeof(bool) * KEYS, 0, cudaMemcpyDeviceToHost);
            for (int i = 0; i < KEYS; ++i) {
                totalMovable += mov[i];
            }
        }

        const std::shared_ptr text = initText(corpus, textExtensions);
        precomputeMetric(text);

        printf("Config: {");
        printf("\n    text:         %llu chars", text->count);
        printf("\n    movable keys: %d / %d chars", totalMovable, KEYS);
        printf("\n    population:   %llu / %llu", size.surviving, size.pop);
        printf("\n    population:   %llu / %llu", size.surviving, size.pop);
        printf("\n    cache:        %s @ %s", cachedIndexFile, cacheDirectory);
        printf("\n    evolution:    %d rounds of %d (plateau length = %d)", rounds, generations, Record::plateauLength);
        printf("\n    seed = %llu", SEED);
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
    None = 0,
    Perf = 1,
    Lock = 2,
    Evolve = 4,
    TestNew = 8, TestOther = 16
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
    static Mode getFlag(const char* word) {
        if (strcmp(word, "perf")    == 0) return Mode::Perf;
        if (strcmp(word, "lock")    == 0) return Mode::Lock;
        if (strcmp(word, "evolve")  == 0) return Mode::Evolve;
        if (strcmp(word, "testn")   == 0) return Mode::TestNew;
        if (strcmp(word, "testc")   == 0) return Mode::TestOther;
        printf("invalid mode: '%s'\n", word);
        return Mode::None;
    }

public:
    std::vector<Mode> modes{};
    Mode all = Mode::None;

    explicit ArgParser(const std::vector<const char*> &args) {
        for (const char* part : args) {
            modes.push_back(getFlag(part));
            all = all | modes.back();
        }
    }
    bool hasArgs() const { return modes.size() > 0; }
    bool hasMode(const Mode m) const {
        return (all & m) != Mode::None;
    }
};

void run(const Config &cnf, const std::vector<const char*> &argv) {
    const ArgParser order(argv);
    if (!order.hasArgs()) { printf("No valid modes supplied.\n"); return; }

    pop_t P = 0, K = 0;
    if (order.hasMode(Mode::Evolve | Mode::Lock | Mode::Perf)) {
        P = cnf.size.pop;
        K = cnf.size.surviving;
        printf("Using P=%s, K=%s\n", F3(P), F3(K));
        {
            constexpr size_t gibibytes = 4;
            constexpr size_t heap = gibibytes << 30;
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap);
        }
    }

    // printf("Total keyboards: %s\n", F3(P * generations * rounds));
    population group(P, K);

    for (const Mode mode : order.modes) {
        switch (mode) {
            case Mode::TestNew: {
                testNew();
                break;
            }
            case Mode::TestOther: {
                testOther();
                break;
            }
            case Mode::Perf: {
                runPerfMethods(group);
                runPerfRound(group, generations, rounds);
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
            case Mode::None:
                break;
        }
    }
}


int main(const int argc, const char* argv[]) {
    if (argc < 2) {  // executable config ...
        printf("Invalid number of arguments. Expected >= 1, Got: %d\n[", argc - 1);
        return 1;
    }

    const std::vector args(argv + 2, argv + argc);
    const Config cnf(argv[1]);
    run(cnf, args);

    // PAUSE("Waiting...");
    printf("\nFinished successfully.\n");
    return 0;
}
