#include <set>
#include <filesystem>

#include <metric.cuh>
#include <text.cuh>
#include <glob.cuh>
#include <md5.cuh>
#include "test.cuh"
#include "record.cuh"
#include "general.cuh"


struct Config {
private:
    static std::string formatFileSize(const uint64_t size) {
        const static char* prefixes[] = {
            "B", "KiB", "MiB", "GiB", "TiB",
        };
        double frac = size;
        int order = 0;
        while (frac > 1024) {
            order++;
            frac /= 1024;
        }

        std::string o = formatNumber<2>(frac);
        o.append(" ").append(prefixes[order]);
        return o;
    }

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
        if (value.at(value.size() - 1) == '\n') {
            value.pop_back();
        }

        if (buf[1] != '/') {
            printf("No ending tag for key '%s': '%s'\n", key.c_str(), buf);
            return false;
        }

        if (key != buf + 2) {
            printf("Keys do not match '%s' != '%s'\n", key.c_str(), buf + 2);
            return false;
        }

        seen.insert(key);
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
            } else {
                seen.extract(key);
                printf("Movable mask is the wrong size.\nExpected: %d\nFound: %lld.\n", KEYS, lmov.size());
            }
        } else if (key == "size") {
            size.pop = 1 << ceilLog2(parseNum(value));
        } else if (key == "survivors") {
            size.surviving = 1 << ceilLog2(parseNum(value));
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
        } else if (key == "output") {
            Record::output = value;
        } else if (key == "corpus") {
            corpus = value;
        } else if (key == "exportTables") {
            exportTables = value;
        } else {
            printf("Unknown key: '%s'\n", key.c_str());
            seen.extract(key);
            return false;
        }
        return true;
    }

    std::set<std::string> seen{};

    std::filesystem::path exportTables;
    std::string corpus;
    std::vector<std::string> textExtensions{};

    bool configured(const char* key) {
        return seen.find(key) != seen.end();
    }
public:
    explicit Config(const char* configuration) {
        std::ifstream cnf(configuration);

        if (!cnf.is_open()) {
            printf("Config could not be opened: '%s'\n", configuration);
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

        bool invalid = false;
        for (const char* required : {"corpus", "output", "size", "rounds"}) {
            if (!configured(required)) {
                printf("Required field `%s` not found", required);
                invalid = true;
            }
        }
        if (invalid) {
            exit(1);
        }

        if (configured("cache") && !is_directory(cacheDirectory)) {
            printf("Creating cache directory.\n");
            create_directory(cacheDirectory);
        }

        if (configured("movable")) {
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

        if (configured("size") && !configured("surviving")) {
            size.surviving = 1 << (ceilLog2(size.pop) - 1) / 2;
        }

        if (!configured("seed")) {
            MD5 seed;
            seed.update((uint64_t) time(nullptr));
            seed.update((uint64_t) &seen);
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

        uint64_t textBytes = 0;
        std::vector<std::filesystem::path> corpora;
        {
            for (std::filesystem::path &path : glob::rglob(corpus)) {
                if (!is_directory(path)) {
                    corpora.push_back(path);
                    textBytes += file_size(path);
                }
            }
        }
        if (corpora.size() == 0) {
            printf("No matching files for pattern '%s'.\n", corpus.c_str());
            exit(1);
        }

        printf("\rConfig: {                    ");
        printf("\n    corpus:          %s file(s) (%s)", F3(corpora.size()), formatFileSize(textBytes).c_str());
        printf("\n    movable keys:    %d / %d chars", totalMovable, KEYS);
        printf("\n    population:      %s / %s", F3(size.surviving), F3(size.pop));
        if (!cacheDirectory.empty()) printf("\n    cache:           %ls", cacheDirectory.c_str());
        else                         printf("\n    cache:           off");
        printf("\n    output:          %ls", Record::output.c_str());
        if (!exportTables.empty()) printf("\n    tables:          %ls", exportTables.c_str());
        printf("\n    evolution:       %s rounds of %s (plateau length = %s)", F3(rounds), F3(generations), F3(Record::plateauLength));
        printf("\n    seed:            %llu", SEED);
        printf("\n}");
        printf("\n");

        const std::shared_ptr text = initText(corpora);
        const mtype* metric = precomputeMetric(text);

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
    }

    struct {
        pop_t pop = 1 << 22;
        pop_t surviving = 0;
    } size;
    int generations = 30;
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
    TestAll = 8,
    TestUser = 16,
    Random = 32,
    QWERTY = 64,
    Help = 128,

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
        if (word == "test"       ) return Mode::TestUser;
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
        for (int rep = 0; rep < repetitions; ++rep) {
            switch (mode) {
                case Mode::TestAll: {
                    testAll();
                    break;
                }
                case Mode::TestUser: {
                    testUser();
                    break;
                }
                case Mode::QWERTY: {
                    constexpr char qwerty[] {
                        "1234567890"
                        "qwertyuiop"
                        "asdfghjkl;"
                        "zxcvbnm,./"
                    };
                    test("qwerty", qwerty);
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
    constexpr char configuration[] = R"""(
Configuration
-----
Valid fields are
 - `size` (required, >512): How many keyboards per generation.
 - `survivors` (default = sqrt(0.5 * size)): How many of those keyboards are used to create the next generation.
 - `cache`: A directory to place cached corpora in. If not included, turns off caching.
 - `generations` (default = 30): How many generations to run until finishing the evolution.
 - `plateauLength` (default = 2): After this many generations without improvement, stop evolving.
 - `rounds` (required): When using the `evolve` mode, decides how many keyboards to generate.
 - `output` (default = "./output"): When using the `evolve` mode, decides where to put generated keyboard files.
 - `seed` (default = random): The seed which drives the pRNG of the program (can be written in hex).
 - `exportTables`: If provided, creates two files in this directory with the corpus and metric used.
 - `corpus` (required): A file or glob pattern to read text from. If this is a pattern, all matching files will be read
                        as text and used to train the keyboards. Otherwise, it will read the entire file.
 - `movable` (default = all movable): Which keys are allowed to be modified on the QWERTY keyboard.
                                      this should be 30 ones or zeros (1 = movable, 0 = left in place).

Of these 10 fields, only `corpus`, `size`, `rounds`, and `output` must be provided.
The fields `size` and `surviving` will both get rounded up to the nearest power of two.
)""";
    constexpr char usage[] = R"""(
USAGE
-----
The first argument to the executable must be the path to your configuration file.
Any other arguments will be treated as modes to run.

There are 8 possibles modes:
 - `help`: Show the possible modes/configurable fields.
 - `perf`: Profile various functions in the code (mostly here for debugging reasons).
 - `rand`: Compute the average score of a random keyboard.
 - `qwerty`: Compute the score of the QWERTY layout.
 - `evolve`: Evolve new keyboards and place them in the configured output directory.
 - `lock`: Compute the cost of leaving each single key in its place.
 - `test`: Test user keyboards given through stdin.
 - `test-common`: Test a few alternative keyboards and QWERTY.

Any mode can be repeated multiple times by following it with a number (e.g. "rand 100" runs "rand" 100 times).
For example:
`dist.exe configuration.txt qwerty rand 100` will do the following:
 1. Load the configuration from `configuration.txt`
 2. Print the score for QWERTY.
 3. Print the average score of `size` random keyboards 100 times.
)""";
    printf(configuration + 1);
    printf("\n-- More --  \r");
    system("pause 1>NUL");
    printf("\r            \r");
    printf(usage + 1);
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
