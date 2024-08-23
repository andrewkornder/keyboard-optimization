#include "text.cuh"
#include <filesystem>
#include <md5.cuh>
#include <unordered_map>


typedef std::filesystem::path path_t;
typedef std::unordered_map<uint64_t, count_t> Counter;

std::string getHashedFile(MD5 hash) {
    const path_t p = cacheDirectory / hash.hex();
    return p.string();
}


constexpr uint64_t loadDimension = 3;
constexpr uint64_t loadBase = 1 << 16;

std::string getHashedFile(const path_t &path) {
    MD5 hash;
    hash.update(path.string());
    // shitty mistake causes me to subtract one instead of reloading all text
    hash.update((char) (loadDimension - 1));
    return getHashedFile(hash);
}

std::string getHashedFile(const path_t &root, std::vector<path_t> paths) {
    MD5 hash;
    hash.update(root.string()).update(loadDimension);

    std::sort(paths.begin(), paths.end());
    for (auto &path : paths) {
        hash.update(path.string());
    }
    return getHashedFile(hash);
}

class BufferedFile {
    constexpr static int size = 32ull << 20;
    std::wifstream file;
    wchar_t* buffer;
    int index = 0;
    int has = 0;

public:
    explicit BufferedFile(const std::string &path, const int mode = std::wifstream::in) : file(path, mode) {
        buffer = new wchar_t[size];
    }
    ~BufferedFile() { delete buffer; }

    wchar_t get() {
        if (eof()) return (wchar_t) -1;

        if (index == has) {
            file.read(buffer, size);
            has = file.gcount();
            index = 0;
        }
        return buffer[index++];
    }
    bool eof() const {
        return index == has && file.eof();
    }
};

void saveCache(const std::string &path, const Counter &counts);
void loadNewText(const std::string &saveTo, const std::string &readFrom, Counter &output) {
    Counter counter;
    counter.reserve(500'000);

    constexpr uint64_t N = ipow(loadBase, loadDimension);

    BufferedFile file(readFrom);

    int pastEnd = 0;
    uint64_t code = 0;
    while (!file.eof() || ++pastEnd < loadDimension) {
        code = file.get() | (code << sizeof(wchar_t) * 8);
        if constexpr (N != 0) {
            code = code % N;
        }
        ++counter[code];
    }

    saveCache(saveTo, counter);
    for (const auto & [key, value] : counter) {
        output[key] += value;
    }
}

void saveCache(const std::string &path, const Counter &counts) {
    std::ofstream file(path);

    std::vector<uint64_t> values;
    values.reserve(counts.size());

    for (const auto& p : counts) {
        values.push_back(p.first);
    }

    std::sort(values.begin(), values.end(), [&counts](const uint64_t a, const uint64_t b) {
        return counts.at(a) > counts.at(b);
    });

    for (const uint64_t key : values) {
        int letters[loadDimension];
        letterUtils.getCharsAtIndexUnsigned<loadDimension, loadBase - 1>(key, letters);
        bool first = true;
        for (const int q : letters) {
            if (!first) file << ',';
            first = false;
            file << q;
        }
        file << '|' << counts.at(key) << '\n';
    }
}

bool loadCached(const std::string &path, Counter &counter) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    char line[64] = {};
    file.get(line, 64, '\n');
    file.seekg(std::ifstream::beg);


    while (!file.eof()) {
        file.get(line, 64, '\n');
        file.get();

        const char* ptr = line;
        if (*ptr > '9' || *ptr < '0') {
            break;
        }

        count_t cp[loadDimension + 1] = {};
        for (int i = 0; i < loadDimension + 1; ++i) {
            count_t x = 0;
            while ('0' <= *ptr && *ptr <= '9') {
                x = 10 * x + (*ptr - '0');
                ++ptr;
            }
            ++ptr;
            cp[i] = x - (i != loadDimension);
        }
        uint64_t code = letterUtils.getIndexAtChars<loadDimension, loadBase - 1>(cp);
        counter[code] += cp[loadDimension];
    }
    return true;
}



std::unique_ptr<FinishedText> initText(const path_t &corpus) {
    std::vector<path_t> corpora;

    if (is_directory(corpus)) {
        for (const auto &entry : std::filesystem::recursive_directory_iterator(corpus)) {
            if (!entry.is_directory()) {
                corpora.push_back(entry.path());
            }
        }
    } else if (exists(corpus)) {
        corpora.push_back(corpus);
    } else {
        printf("Corpus could not be found: '%ls'.\n", corpus.c_str());
        exit(1);
    }

    Counter counts;
    if (const std::string totalCache = getHashedFile(corpus, corpora) + ".sum"; !loadCached(totalCache, counts)) {
        for (auto &path : corpora) {
            const clock_t start = clock();
            printf("Loading corpus: '%ls'...", path.c_str());
            if (const std::string singleCache = getHashedFile(path) + ".chc"; !loadCached(singleCache, counts)) {
                loadNewText(singleCache, path.string(), counts);
            }
            const clock_t elapsed = clock() - start;
            printf("\rLoaded corpus '%ls' in %s ms.\n", path.c_str(), F3(elapsed));
        }
        saveCache(totalCache, counts);
    }

    count_t* array = new count_t[FinishedText::N]();

    for (const auto [key, value] : counts) {
        int letters[loadDimension];
        letterUtils.getCharsAtIndexUnsigned<loadDimension, loadBase - 1>(key, letters);
        constexpr int skip = loadDimension - textWindow;
        for (int i = skip; i < loadDimension; ++i) {
            letters[i] = letterUtils.positionOf(letters[i]);
        }
        const uint64_t i = letterUtils.getIndexAtChars<textWindow, KEYS>(letters + skip);
        array[i] += value;
    }

    std::unique_ptr text = std::make_unique<FinishedText>(array);
    delete array;

    return text;
}


void mapKeys(const char* arr, char* out) {
    for (int i = 0, k = 0; k < KEYS; ++i) {
        if (const int j = letterUtils.positionOf(arr[i]); j != -1) {
            out[j] = k++;
        }
    }
}

__host__ __device__ void printArr(const char* arr) {
    for (int i = 0; i < KEYS; ++i) {
        printf("%02d,", arr[i]);
        if (i % 10 == 9) printf("\n");
    }
}


__host__ void printArrQ(const char* arr) {
    char letters[KEYS];
    for (int i = 0; i < KEYS; ++i) {
        letters[arr[i]] = i;
    }

    for (int i = 0; i < KEYS; ++i) {
        printf("%c", KEYS_LOWER[letters[i]]);
        if (i % 10 == 9) printf("\n");
    }
}