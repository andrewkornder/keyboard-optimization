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

    if (!configured("locked")) {
        bool lmov[KEYS];
        memset(lmov, true, KEYS);
        cudaMemcpyToSymbol(movable, lmov, sizeof(lmov), 0, cudaMemcpyHostToDevice);
        char ldef[KEYS];
        for (int i = 0; i < KEYS; ++i) {
            ldef[i] = i;
        }
        cudaMemcpyToSymbol(defaultKeyboard, ldef, sizeof(ldef), 0, cudaMemcpyHostToDevice);
    }

    int totalMovable = 0;
    {
        bool mov[KEYS];
        cudaMemcpyFromSymbol(mov, movable, sizeof(bool) * KEYS, 0, cudaMemcpyDeviceToHost);
        for (int i = 0; i < KEYS; ++i) {
            totalMovable += mov[i];
            hashData << (mov[i] ? '1' : '0');
        }
        hashData << '|';
        char def[KEYS];
        cudaMemcpyFromSymbol(def, defaultKeyboard, sizeof(char) * KEYS, 0, cudaMemcpyDeviceToHost);
        for (int i = 0; i < KEYS; ++i) {
            hashData << (int) def[i] << ',';
        }
        hashData << '|';
    }

    if (configured("size") && !configured("surviving")) {
        size.surviving = 1 << (ceilLog2(size.pop) - 1) / 2;
    }

    hashData << size.surviving << ',' << size.pop << '|';

    if (!configured("seed")) {
        MD5 seed;
        seed.update((uint64_t) time(nullptr));
        seed.update((uint64_t) &seen);
        seed.update((uint64_t) clock());
        seed.update(0x7f18ee808626fcb9ULL);

        const auto [hi, lo] = seed.checksum();
        SEED = hi + lo;
    }
    hashData << SEED << '|';

    uint64_t textBytes = 0;
    std::vector<std::filesystem::path> corpora;
    {
        hashData << corpus << '|';
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

    hashData << (int) showOutput << '|';
    hashData << (int) population::allowDuplicates << '|';
    hashData << output << '|';
    hashData << cacheDirectory << '|';
    hashData << exportTables << '|';
    hashData << plateauLength << '|';
    hashData << generations << '|';
    hashData << rounds;
    {
        MD5 md5;
        md5.update(hashData.str());
        hashData.clear();
        hash = md5.hex();
    }

    sprintf(buffer,
        "Config: {"                                                  "\n"
        "    hash:            0x%s"                                  "\n"
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
        hash.c_str(),
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

    if (!exportTables.empty() && !is_directory(exportTables)) {
        printf("Creating directory to export tables to...");
        create_directory(exportTables);
    }

    const std::shared_ptr text = initText(exportTables / "text.chc", corpora);
    const mtype* metric = precomputeMetric(text);

    if (!exportTables.empty()) {
        std::ofstream file(exportTables / "metric.chc");
        letterUtils.applyOnIndices<textWindow, KEYS>([&file, metric](const int index, const char keys[textWindow]) {
            for (int i = 0; i < textWindow; ++i) {
                file << (keys[i] == -1 ? ' ' : KEYS_LOWER[keys[i]]);
            }
            file << '|' << metric[index].cost << '\n';
        });
    }
}

bool validKeyboard(const char* keyboard) {
    bool filled[KEYS] = {};
    bool seen[KEYS] = {};

    for (int i = 0; i < KEYS; ++i) {
        if (const char k = keyboard[i]; 0 <= k && k < KEYS) {
            filled[k] = true;
            seen[i] = true;
        }
    }
    for (int i = 0; i < KEYS; ++i) {
        if (!filled[i] || !seen[i]) {
            return false;
        }
    }
    return true;
}

void Config::match(const std::string& key, const std::string& value) {
    if (seen.find(key) != seen.end()) {
        printf("Field '%s' was defined more than once. Any definition(s) after the first will be ignored\n", key.c_str());
        return;
    }
    seen.insert(key);
    if (key == "locked") {
        std::vector<int> lmov;

        for (const char c : value) {
            if (c == '_') {
                lmov.push_back(-1);
            } else if (const int i = letterUtils.positionOf(c); i != -1) {
                lmov.push_back(i);
            }
        }
        if (lmov.size() == KEYS) {
            bool filled[KEYS] = {};
            bool mask[KEYS];
            char defaultKB[KEYS];

            for (int i = 0; i < KEYS; ++i) {
                defaultKB[i] = -1;
                mask[i] = true;
            }

            for (int i = 0; i < KEYS; ++i) {
                if (const int letter = lmov[i]; letter != -1) {
                    mask[letter] = false;
                    defaultKB[letter] = i;
                    filled[i] = true;
                }
            }

            for (int letter = 0, pos = 0; letter < KEYS; ++letter) {
                if (defaultKB[letter] != -1) continue;
                while (filled[pos]) ++pos;
                defaultKB[letter] = pos++;
            }

            if (!validKeyboard(defaultKB)) {
                printf("Field `locked` was invalid.\n");
                exit(1);
            }

            cudaMemcpyToSymbol(defaultKeyboard, defaultKB, sizeof(defaultKB), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(movable, mask, sizeof(mask), 0, cudaMemcpyHostToDevice);
        } else {
            seen.extract(key);
            printf("Field `locked` is the wrong size.\nExpected: %d\nFound: %lld.\n", KEYS, lmov.size());
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
        if (value == "random") {
            seen.erase(key);
        } else if (value.size() > 2 && value[0] == '0' && value[1] == 'x') {
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
    }
}

void Config::parse(const char* configuration) {
    std::ifstream cnf(configuration);

    if (!cnf.is_open()) {
        printf("Config could not be opened: '%s'\n", configuration);
        exit(1);
    }

    while (!cnf.eof()) {
        char first = cnf.get();
        while (isspace(first)) {
            first = cnf.get();
        }
        if (cnf.eof()) break;
        if (first == '#') {
            cnf.ignore(1024, '\n');
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
