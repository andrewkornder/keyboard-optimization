#include <set>
#include <filesystem>

#include <metric.cuh>
#include <glob.cuh>
#include <config.cuh>
#include "test.cuh"
#include "record.cuh"
#include "general.cuh"



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
                Record rec(&group, cnf);
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
 - `showOutput` (default = off): Whether to show each generated keyboard while running `evolve`.

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
    printf("%s", message + 1);
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
