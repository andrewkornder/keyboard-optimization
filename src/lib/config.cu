#include "config.cuh"

#include <glob.cuh>
#include <metric.cuh>
#include <record.cuh>
#include <md5.cuh>
#include <rng.cuh>

uint64_t parseNum(const std::string &value) {
    uint64_t x = 0;
    for (const char p : value) {
        if ('0' <= p && p <= '9') {
            x = 10 * x + (p - '0');
        }
    }
    return x;
}
uint64_t parseHex(const std::string &value) {
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


Config::Config(const char* configuration) {
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

    sprintf(buffer,
        "Config: {"                                                  "\n"
        "    corpus:          %s file(s) (%s)"                       "\n"
        "    movable keys:    %d / %d chars"                         "\n"
        "    population:      %s / %s"                               "\n"
        "    duplicates:      %s"                                    "\n"
        "    cache:           %ls"                                   "\n"
        "    tables:          %ls"                                   "\n"
        "    output:          %ls"                                   "\n"
        "    evolution:       %s rounds of %s (plateau length = %s)" "\n"
        "    showOutput:      %s"                                    "\n"
        "    seed:            %llu"                                  "\n"
        "}"                                                          "\n",
        F3(corpora.size()), formatFileSize(textBytes).c_str(),
        totalMovable, KEYS,
        F3(size.surviving), F3(size.pop),
        population::allowDuplicates ? "on" : "off",
        cacheDirectory.empty() ? L"off" : cacheDirectory.c_str(),
        exportTables.empty() ? L"not exported" : exportTables.c_str(),
        output.c_str(),
        F3(rounds), F3(generations), F3(plateauLength),
        showOutput ? "on" : "off",
        SEED
    );
    printf("%s\n", buffer);

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

bool Config::match(const std::string& key, const std::string& value) {
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
        plateauLength = parseNum(value);
    } else if (key == "rounds") {
        rounds = parseNum(value);
    } else if (key == "seed") {
        if (value[0] == '0' && value[1] == 'x') {
            SEED = parseHex(value);
        } else {
            SEED = parseNum(value);
        }
    } else if (key == "output") {
        output = value;
    } else if (key == "corpus") {
        corpus = value;
    } else if (key == "showOutput") {
        showOutput = value == "on";
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

void Config::parse(const char* configuration) {
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
