#include "text.cuh"
#include <filesystem>
#include <md5.cuh>
#include <unordered_map>


constexpr int lmin = '!', lmax = '~' + 1;
constexpr uint64_t loadBase = lmax - lmin;
constexpr uint64_t loadDimension = 3;
constexpr uint64_t validNgrams = ipow(loadBase + 1, loadDimension);


std::string getHashedFile(MD5 hash) {
    const std::filesystem::path p = cacheDirectory / hash.hex();
    return p.string();
}


std::string getHashedFile(const std::filesystem::path& path) {
    if (cacheDirectory.empty()) return "";

    MD5 hash;
    // constexpr uint64_t SALT = 0x4ada93c00a8bf5abULL;
    // hash.update(SALT);
    hash.update(path.string());
    hash.update(loadDimension);
    return getHashedFile(hash);
}

std::string getHashedFile(std::vector<std::filesystem::path> &paths) {
    if (cacheDirectory.empty()) return "";

    MD5 hash;
    hash.update(loadDimension);

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
    ~BufferedFile() { delete[] buffer; }

    wchar_t get() {
        if (eof()) return (wchar_t) -1;

        if (index == has) {
            file.read(buffer, size);

            has = file.gcount();
            read += has;
            index = 0;
        }
        return buffer[index++];
    }
    bool eof() const {
        return index == has && file.eof();
    }
    std::streamsize read = 0;
};


void saveCache(const std::string &path, const count_t* counts);
void loadNewText(const std::string &saveTo, const std::string &readFrom, count_t* output) {
    count_t* counter = new count_t[validNgrams]();
    // counter.reserve(500'000);

    BufferedFile file(readFrom);

    int pastEnd = 0;
    uint64_t code = 0;
    while (!file.eof() || ++pastEnd < loadDimension) {
        unsigned int ch = file.get() - lmin;
        // sneaky underflow to do the equivalent of `ch < lmin ? lmax : ch;
        ch = ch >= lmax ? lmax : ch;
        code = ch + code * (loadBase + 1);
        if constexpr (validNgrams != 0) {
            code = code % validNgrams;
        }
        ++counter[code];
    }

    // if the cache file is bigger than the actual file, just read it from file next time
    if (!cacheDirectory.empty() && file.read > 8 * validNgrams) {
        saveCache(saveTo, counter);
    }

    for (int key = 0; key < validNgrams; ++key) {
        output[key] += counter[key];
    }

    delete[] counter;
}

void saveCache(const std::string &path, const count_t* counts) {
    std::ofstream file(path);

    std::vector<uint64_t> values;
    values.reserve(validNgrams / 2);

    for (int key = 0; key < validNgrams; ++key) {
        if (counts[key] > 0) {
            values.push_back(key);
        }
    }

    std::sort(values.begin(), values.end(), [counts](const uint64_t a, const uint64_t b) {
        return counts[a] > counts[b];
    });

    for (const uint64_t key : values) {
        int letters[loadDimension];
        letterUtils.getCharsAtIndexUnsigned<loadDimension, loadBase>(key, letters);
        bool first = true;
        for (const int q : letters) {
            if (!first) file << ',';
            first = false;
            file << q;
        }
        file << '|' << counts[key] << '\n';
    }
}

bool loadCached(const std::string &path, count_t* counter) {
    if (cacheDirectory.empty()) return false;

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
        const uint64_t code = letterUtils.getIndexAtChars<loadDimension, loadBase>(cp);
        counter[code] += cp[loadDimension];
    }
    return true;
}


std::unique_ptr<FinishedText> initText(std::vector<std::filesystem::path> &corpora) {
    count_t* counts = new count_t[validNgrams];
    if (const std::string totalCache = getHashedFile(corpora) + ".sum"; !loadCached(totalCache, counts)) {
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

    constexpr int skip = loadDimension - textWindow;
    for (int key = 0; key < validNgrams; ++key) {
        const count_t value = counts[key];
        int letters[loadDimension];
        letterUtils.getCharsAtIndexUnsigned<loadDimension, loadBase>(key, letters);
        for (int i = skip; i < loadDimension; ++i) {
            letters[i] = letterUtils.positionOf(lmin + letters[i]);
        }
        const uint64_t i = letterUtils.getIndexAtChars<textWindow, KEYS>(letters + skip);
        array[i] += value;
    }
    array[0] -= corpora.size() * skip;

    std::unique_ptr text = std::make_unique<FinishedText>(array);
    delete[] array;
    delete[] counts;

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