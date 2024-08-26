#include "text.cuh"
#include <filesystem>
#include <md5.cuh>
#include <unordered_map>


constexpr char lmin = '!', lmax = '~' + 1;
constexpr char loadBase = lmax - lmin;
constexpr char loadDimension = 3;
constexpr uint64_t validNgrams = ipow(loadBase + 1, loadDimension);


std::string getHashedFile(MD5 hash) {
    const std::filesystem::path p = cacheDirectory / hash.b64();
    return p.string();
}


constexpr uint64_t textVersion = 0xBCC0000000000001;
std::string getHashedFile(const std::filesystem::path& path) {
    if (cacheDirectory.empty()) return "";

    MD5 hash;
    // constexpr uint64_t SALT = 0x4ada93c00a8bf5abULL;
    // hash.update(SALT);
    hash.update(path.string());
    hash.update(loadDimension);
    // old version should be incompatible
    hash.update(textVersion);
    return getHashedFile(hash);
}

std::string getHashedFile(const std::vector<std::filesystem::path> &paths) {
    if (cacheDirectory.empty()) return "";

    MD5 hash;
    hash.update(loadDimension);

    for (auto &path : paths) {
        hash.update(path.string());
    }
    return getHashedFile(hash);
}

class BufferedFile {
    constexpr static int size = 16ull << 20;
    std::wifstream file;
    wchar_t* buffer;
    int index = 0;
    int has = 0;

public:
    explicit BufferedFile(const std::string &path, const int mode = std::wifstream::in) : file(path, mode) {
        buffer = new wchar_t[size];

        const std::streampos fsize = file.tellg();
        file.seekg(0, std::ifstream::end);

        count = file.tellg() - fsize;
        file.seekg(0, std::ifstream::beg);
    }
    ~BufferedFile() { delete[] buffer; }

    __inline__ wchar_t get() {
        if (eof()) return (wchar_t) -1;

        if (index == has) {
            file.read(buffer, size);

            has = file.gcount();
            read += has;
            index = 0;
            new_ = true;
        } else {
            new_ = false;
        }
        return buffer[index++];
    }

    bool eof() const {
        return index == has && file.eof();
    }

    bool new_ = false;
    std::streamsize count = 0;
    std::streamsize read = 0;
};


void saveCache(const std::string &path, const count_t* counts, uint64_t maximumSize);
void loadNewText(const std::string &saveTo, const std::string &readFrom, count_t* output) {
    count_t* counter = new count_t[validNgrams]();

    BufferedFile file(readFrom);

    int pastEnd = 0;
    uint64_t code = 0;

    constexpr char spinner[4] = {'|', '/', '-', '\\'};
    int spinnerIdx = 0;

    while (!file.eof() || ++pastEnd < loadDimension) {
        if (file.new_) {
            printf("\r[%c] Loading '%s'... %.2f%%", spinner[spinnerIdx], readFrom.c_str(), 100. * file.read / file.count);
            spinnerIdx = (spinnerIdx + 1) % 4;
        }
        int ch = file.get() - lmin;
        if (ch < 0 || loadBase <= ch) {
            ch = -1;
        }
        code = 1 + ch + code * (loadBase + 1);
        if constexpr (validNgrams != 0) {
            code = code % validNgrams;
        }
        ++counter[code];
    }

    if (!cacheDirectory.empty()) {
        saveCache(saveTo, counter, file.read);
    }

    for (int key = 0; key < validNgrams; ++key) {
        output[key] += counter[key];
    }

    delete[] counter;
}

void saveCache(const std::string &path, const count_t* counts, const uint64_t maximumSize) {
    count_t max = 0;

    uint64_t existing = 0;
    for (int i = 0; i < validNgrams; ++i) {
        const count_t v = counts[i];
        existing += v > 0;
        max = max < v ? v : max;
    }

    char bytesPerCount = 1;
    while (max >>= 8) {
        ++bytesPerCount;
    }

    if ((loadDimension + bytesPerCount) * existing > maximumSize) {
        // printf("\nNot saving to cache because heuristic check failed\n");
        return;
    }

    std::ofstream file(path, std::ofstream::binary);
    if (!file.is_open()) {
        printf("\nFailed to open file \"%s\"\n", path.c_str());
        return;
    }
    file << bytesPerCount;
    file << loadDimension;
    file << loadBase;
    file << lmin << lmax;
    file << "CHC";

    for (int i = 0; i < sizeof(uint64_t); ++i) {
        file << (char) (existing >> 8 * i & 0xff);
    }

    std::vector<uint64_t> values;
    values.reserve(existing);

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
        letterUtils.getCharsAtIndex<loadDimension, loadBase>(key, letters);
        for (const int q : letters) {
            file << (char) (1 + q);
        }
        count_t cnt = counts[key];
        for (int i = 0; i < bytesPerCount; ++i) {
            file << (char) (cnt & 0xff);
            cnt >>= 8;
        }
#if 1
        if (cnt != 0) {
            printf("\nWrong value for bytesPerCount (%d) tested against 0x%llx\n", bytesPerCount, counts[key]);
            file.close();
            std::filesystem::remove(path);
            exit(1);
        }
#endif
    }
}

bool loadCached(const std::string &path, count_t* counter) {
    if (cacheDirectory.empty()) return false;

    std::ifstream file(path, std::ofstream::binary);
    if (!file.is_open()) {
        return false;
    }

    char start[9] = {};
    file.read(start, 8);

    const char bytesPerCount = start[0];
    const char dimension = start[1];
    const char base = start[2];
    const char lmin_ = start[3], lmax_ = start[4];

    if (base != loadBase) return false;
    if (lmax_ != lmax) return false;
    if (lmin_ != lmin) return false;
    if (dimension < loadDimension) return false;
    if (strcmp(start + 5, "CHC") != 0) return false;

    uint64_t length = 0;
    for (int i = 0; i < sizeof(uint64_t); ++i) {
        length |= (uint64_t) file.get() << 8 * i;
    }

    for (uint64_t n = 0; n < length; ++n) {
        int cp[10] = {};
        for (int i = 0; i < dimension; ++i) {
            cp[i] = file.get() - 1;
        }

        count_t cnt = 0;
        for (int i = 0; i < bytesPerCount; ++i) {
            cnt |= (count_t) file.get() << 8 * i;
        }
        const uint64_t code = letterUtils.getIndexAtChars<loadDimension, loadBase>(cp + dimension - loadDimension);
        counter[code] += cnt;
    }
    return true;
}

void exportText(const std::filesystem::path& to, const count_t* array) {
    if (to.empty()) return;

    std::ofstream file(to);

    std::vector<uint64_t> values;
    values.reserve(validNgrams);

    for (int i = 0; i < validNgrams; ++i) {
        if (array[i] > 0) {
            values.push_back(i);
        }
    }

    std::sort(values.begin(), values.end(), [array](const uint64_t a, const uint64_t b) {
        return array[a] > array[b];
    });

    for (const uint64_t key : values) {
        int letters[loadDimension];
        letterUtils.getCharsAtIndex<loadDimension, loadBase>(key, letters);
        for (const int q : letters) {
            file << (char) (q + lmin);
        }
        file << ": " << array[key] << '\n';
    }
}


std::unique_ptr<FinishedText> initText(const std::filesystem::path &exportTo, std::vector<std::filesystem::path> &corpora) {
    count_t* counts = new count_t[validNgrams]();

    const clock_t start = clock();
    uint64_t bytes = 0;

    for (int i = 0; i < corpora.size(); ++i) {
        corpora[i] = canonical(corpora[i]);
    }
    std::sort(corpora.begin(), corpora.end(), [](const std::filesystem::path &a, const std::filesystem::path &b) {
        const std::string &p = a.string(), &q = b.string();
        if (p.size() != q.size()) {
            return p.size() > q.size();
        }
        return p > q;
    });

    clock_t lastExport = clock();
    exportText(exportTo, counts);

    if (const std::string totalCache = getHashedFile(corpora) + ".sum"; !loadCached(totalCache, counts)) {
        for (auto &path : corpora) {
            const clock_t startSingle = clock();

            printf("Loading '%ls'...", path.c_str());

            std::string src;

            uint64_t read;
            if (const std::string singleCache = getHashedFile(path) + ".chc"; !loadCached(singleCache, counts)) {
                loadNewText(singleCache, path.string(), counts);

                read = file_size(path);
                src = "file";
            } else {
                read = std::filesystem::file_size(singleCache);
                src = "cache";
            }

            if (const clock_t now = clock(); now - lastExport > CLOCKS_PER_SEC * 2) {
                lastExport = now;
                exportText(exportTo, counts);
            }
            bytes += read;

            const clock_t elapsed = clock() - startSingle;
            printf("\rLoaded '%ls' from %s in %s ms.", path.c_str(), src.c_str(), F3(elapsed));
            printf(" (%s/s)   \n", formatFileSize(read / (double) elapsed * CLOCKS_PER_SEC).c_str());
        }
        if (corpora.size() > 1) {
            saveCache(totalCache, counts, bytes);
        }
    } else {
        bytes += std::filesystem::file_size(totalCache);
    }

    exportText(exportTo, counts);

    const clock_t elapsed = clock() - start;
    printf("Loaded all corpora in %s ms.", F3(elapsed));
    printf(" (%s/s)     \n", formatFileSize(bytes / (double) elapsed * CLOCKS_PER_SEC).c_str());

    count_t* array = new count_t[FinishedText::N]();

    constexpr int skip = loadDimension - textWindow;
    for (int key = 0; key < validNgrams; ++key) {
        const count_t value = counts[key];
        int letters[loadDimension];
        letterUtils.getCharsAtIndex<loadDimension, loadBase>(key, letters);
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
