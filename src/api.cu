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

    bool match(const std::string &key, const std::string &value) {
        if (seen.find(key) != seen.end()) {
            printf("Field '%s' was defined more than once. The previous definition(s) will be ignored\n", key.c_str());
        }
        seen.insert(key);
        if (key == "movable") {
            std::set<int> lmov;

            for (const char c : value) {
                if (const int i = letterUtils.positionOf(c); i != -1) {
                    if (lmov.find(i) != lmov.end()) {
                        printf("Field 'movable' contains '%c' multiple times. Ignoring duplicates...\n", c);
                    }
                    lmov.insert(i);
                }
            }
            if (lmov.size() <= KEYS) {
                bool arr[KEYS] = {};
                for (const int i : lmov) {
                    arr[i] = true;
                }
                cudaMemcpyToSymbol(movable, arr, sizeof(arr), 0, cudaMemcpyHostToDevice);
            } else {
                seen.extract(key);
                printf("Movable mask is the wrong size.\nExpected: at most %d\nFound: %lld.\n", KEYS, lmov.size());
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
        } else if (key == "duplicates") {
            population::allowDuplicates = value == "on";
        } else {
            printf("Unknown key: '%s'\n", key.c_str());
            seen.extract(key);
            return false;
        }
        return true;
    }

    void parse(const char* configuration) {
        std::ifstream cnf(configuration);

        if (!cnf.is_open()) {
            printf("Config could not be opened: '%s'\n", configuration);
            exit(1);
        }

        while (!cnf.eof()) {
            if (const char start = cnf.get(); !isalnum(start) || start == '#') {
                if (start != '\n') cnf.ignore(1024, '\n');
                continue;
            }
            cnf.seekg(-1, std::ifstream::cur);

            std::string key, value;
            key.reserve(16);
            value.reserve(256);

            char buffer[1024];
            cnf.get(buffer, 1024, '=');
            cnf.get();

            key = buffer;
            while (key.back() == ' ') key.pop_back();

            while (true) {
                cnf.get(buffer, 1024, '\n');
                cnf.get();
                const char* start = buffer;
                if (value.empty()) {
                    while (*start == ' ') start++;
                }

                value.append(start);
                if (value.back() != '\\') break;
                value.pop_back();
            }
            while (value.back() == ' ') value.pop_back();
            match(key, value);
        }
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
        parse(configuration);

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

            const auto [hi, lo] = seed.checksum();
            SEED = hi + lo;
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

        printf(  "Config: {                    ");
        printf("\n    corpus:          %s file(s) (%s)", F3(corpora.size()), formatFileSize(textBytes).c_str());
        printf("\n    movable keys:    %d / %d chars", totalMovable, KEYS);
        printf("\n    population:      %s / %s", F3(size.surviving), F3(size.pop));
        printf("\n    duplicates:      %s", population::allowDuplicates ? "on" : "off");
        printf("\n    cache:           %ls", cacheDirectory.empty() ? L"off" : cacheDirectory.c_str());
        printf("\n    tables:          %ls", exportTables.empty() ? L"not exported" : exportTables.c_str());
        printf("\n    output:          %ls", Record::output.c_str());
        printf("\n    evolution:       %s rounds of %s (plateau length = %s)", F3(rounds), F3(generations), F3(Record::plateauLength));
        printf("\n    seed:            %llu", SEED);
        printf("\n}");
        printf("\n");

        if (configured("cache") && !is_directory(cacheDirectory)) {
            printf("Creating cache directory.\n");
            create_directory(cacheDirectory);
        }

        const std::shared_ptr text = initText(corpora);
        const mtype* metric = precomputeMetric(text);

        if (!exportTables.empty()) {
            if (!is_directory(exportTables)) {
                printf("Creating directory to export tables to...");
                create_directory(exportTables);
            }

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
        printf("Trial %d / %d...\r", try_, tries);
        group.init();
        for (int j = 0; j < 25; ++j) {
            group.advance();
        }
        if (all == nullptr) {
            all = group.snapshot();
        } else {
            std::unique_ptr s = group.snapshot();
            all->add(s);
        }
    }
    return all;
}

// todo: dont rearrange but just move pointer for kbs which will be overwritten anyway
// todo: multiple scored in single kernel pass

struct Mode {
    std::string word;
    int repeat;

    Mode(const std::string &word, const int repeat) : word(word), repeat(repeat) {}
};

class ArgParser {
    static std::string getFlag(const std::string &word) {
        for (const char* valid : {"perf", "lock", "evolve", "test", "test-all", "rand", "help"}) {
            if (word == valid) return word;
        }
        if (testable(word)) return word;

        printf("Invalid mode given: '%s'\n", word.c_str());
        return "help";
    }

public:
    std::vector<Mode> modes{};
    std::set<std::string> all;

    explicit ArgParser(const char* argv[], const int argc) {
        for (int i = 2; i < argc; ++i) {
            std::string mode = getFlag(argv[i]);
            
            int rep = 1;
            if (i != argc - 1) {
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

            modes.emplace_back(mode, rep);
            all.insert(mode);
        }
    }
    bool hasArgs() const { return modes.size() > 0; }

    bool hasMode(const std::string &mode) const {
        return all.find(mode) != all.end();
    }
    bool hasMode(const std::initializer_list<std::string> &arr) const {
        for (const std::string &s : arr) {
            if (hasMode(s)) return true;
        }
        return false;
    }
};

void run(const Config &cnf, const ArgParser &args) {
    pop_t P = 0, K = 0;
    if (args.hasMode({"evolve", "perf", "lock", "rand"})) {
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
            if (mode == "test-all") {
                testAll();
            } else if (mode == "test") {
                testUser();
            } else if (testable(mode)) {
                test(mode);
            } else if (mode == "perf") {
                runPerfMethods(group);
                runPerfRound(group, cnf.generations, cnf.rounds);
            } else if (mode == "rand") {
                group.init();
                group.scoreAndSort();
                printf("Average score for %s randomized organisms: %s\n", F3(group.n), F3p(group.averageScore(), 4));
            } else if (mode == "lock") {
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
                    printf("Locked %c: [%f, %f]\n", KEYS_LOWER[i], s->best->stats.score / best, s->worst->stats.score / worst);
                }
            } else if (mode == "evolve") {
                Record rec(&group, cnf.generations, cnf.rounds);
                rec.get();
            } else if (mode == "help") {}
            else {
                printf("What the fuck happened.\n");
                exit(0);
            }
        }
    }
}


void printHelpMessage() {
    constexpr char message[] = R"""(
CONFIGURATION
-----
This program needs a configuration file to properly run. This is just a text file with a few fields that need to be filled out.
Each field is declared with the name and a value separated by an equals sign. Every field should be placed on different lines.

For example, the line `corpus = ./text/` declares that the field `corpus` should be `"./text/"`.
A line can be commented out by starting it with '#'. Ending a line with a backslash ('\') continues the line.

A field can be one of the following types of values:
 1. A boolean value represented by "on" or "off".
 2. An integer represented by a series of digits (anything besides the digits 0-9 will be ignored).
 3. A string of arbitrary text.
 4. A path-like string (e.g. "./folder/file.txt" or "C:/folder/*.txt").

Valid fields are
 - `size` (required, >512): How many keyboards per generation.
 - `survivors` (default = sqrt(0.5 * size)): How many of those keyboards are used to create the next generation.
 - `cache`: A directory to place cached corpora in. If not included, turns off caching.
 - `generations` (default = 30): How many generations to run until finishing the evolution.
 - `plateauLength` (default = 2): After this many generations without improvement, stop evolving.
 - `rounds` (required): When using the `evolve` mode, decides how many keyboards to generate.
 - `duplicates` (default = on): Whether to tolerate duplicate keyboards in each generation. Deduplication is slow,
                                and offers only marginal improvements, so I suggest leaving duplicates on.
 - `output` (default = "./output"): When using the `evolve` mode, decides where to put generated keyboard files.
 - `seed` (default = random): The seed which drives the pRNG of the program (can be written in hex).
 - `exportTables`: If provided, creates two files in this directory with the corpus and metric used.
 - `corpus` (required): A file or glob pattern to read text from. If this is a pattern, all matching files will be read
                        as text and used to train the keyboards. Otherwise, it will read the entire file.
 - `movable` (default = all movable): Which keys are allowed to be modified on the QWERTY keyboard.
                                      this should be a list of every key which is allowed to move (e.g. "abcdef,.").

Of these 10 fields, only `corpus`, `size`, `rounds`, and `output` must be provided.
The fields `size` and `surviving` will both get rounded up to the nearest power of two.


USAGE
-----
The first argument to the executable must be the path to your configuration file.
Any other arguments will be treated as modes to run.

There are 7 possibles modes:
 - `help`: Show the possible modes/configurable fields.
 - `perf`: Profile various functions in the code (mostly here for debugging reasons).
 - `rand`: Compute the average score of a random keyboard.
 - `evolve`: Evolve new keyboards and place them in the configured output directory.
 - `lock`: Compute the cost of leaving each single key in its place.
 - `test`: Test user keyboards given through stdin.
 - `test-all`: Test a few alternative keyboards and QWERTY.

Additionally, using the name of a common keyboard layout will compute the score of that layout on this metric.
Valid layouts are:
 - qwerty
 - alphabet
 - dvorak
 - colemak
 - carpalx
 - arensito
 - asset
 - capewell

Any mode can be repeated multiple times by following it with a number (e.g. "rand 100" runs "rand" 100 times).
For example:
`dist.exe configuration.txt qwerty rand 100` will do the following:
 1. Load the configuration from `configuration.txt`
 2. Print the score for QWERTY.
 3. Print the average score of `size` random keyboards 100 times.
)""";
    printf(message + 1);
}


int main(const int argc, const char* argv[]) {
    if (argc < 2) {  // executable config ...
        printf("Invalid number of arguments. Expected >= 1, Got: %d\n", argc - 1);
        printf("\nUsage:\n%ls config.txt [mode1 [mode2 [...]]\n", std::filesystem::path(argv[0]).filename().c_str());
        return 1;
    }

    {
        const ArgParser args(argv, argc);
        if (args.hasMode("help")) {
            printHelpMessage();
            printf("Press any key to continue...");
            system("pause 1>NUL");
            system("cls");
        }

        // if (!args.hasArgs()) { printf("No valid modes supplied.\n"); return 0; }

        const Config cnf(argv[1]);

        run(cnf, args);
    }

    printf("\nFinished successfully.\n");
    // printf("\nFinished successfully. Press any key to exit.");
    // system("pause 1>NUL");
    return 0;
}
