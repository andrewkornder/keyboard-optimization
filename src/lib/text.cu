// ReSharper disable CppDFAUnreachableCode
// ReSharper disable CppDFAConstantConditions
// ReSharper disable CppDFAConstantFunctionResult
#include <fstream>

#include "text.cuh"

#include <sstream>
#include <filesystem>


void mapKeys(const char* arr, char* out) {
    for (int i = 0; i < KEYS; ++i) {
        out[letterUtils.keyboardPosition[arr[i]]] = i;
    }
}

void writeCache(const wchar_t* path, const UnfinishedText &text) {
    std::ofstream file(path);

    char code[maxTextWindow] = {};
    bool first = true;
    for (int i = 0; i < text.N; ++i) {
        if (text.ngrams[i] == 0) continue;

        if (!first) file << '\n';
        first = false;
        letterUtils.getCharsAtIndex<maxTextWindow, letterUtils.totalPrintable>(i, code);
        for (int j = 0; j < maxTextWindow; ++j) {
            file << (code[j] == -1 ? ' ' : letterUtils.codeToPrintable[code[j]]);
        }
        file << '|' << text.ngrams[i];
    }
}

void saveToCache(const wchar_t* path, const UnfinishedText &t) {
    if (!cachedText) return;

    std::wstringstream stream;
    stream << cacheDirectory << clock() << time(nullptr) << ".textc";

    const std::wstring out = stream.str();
    writeCache(out.c_str(), t);

    std::wofstream cache(cachedIndexFile, std::ofstream::out | std::ofstream::app);
    if (!cache.is_open()) {
        printf("Failed to save to cache\n");
        return;
    }

    cache << path << '|';
    cache << out << '\n';
    cache.flush();
    cache.close();
}

class BufferedFile {
    constexpr static int size = 1 << 25;
    std::ifstream file;
    char* buffer;
    int index = 0;
    int has = 0;

public:
    BufferedFile(const wchar_t* path, const int mode) : file(path, mode) {
        buffer = new char[size];
    }
    ~BufferedFile() { delete buffer; }

    char get() {
        if (eof()) return -1;

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


void loadText(const wchar_t* path, const UnfinishedText &out) {
    int time = clock();

    BufferedFile file(path, std::ifstream::in | std::ifstream::binary);

    const UnfinishedText text;
    count_t code = 0;

    int pastEnd = 0;
    while (true) {
        if (file.eof()) {
            if (++pastEnd == maxTextWindow) break;
        }

        const unsigned char q = file.get();
        const int c = 1 + letterUtils.printableToCode[q];
        code = (c + code * letterUtils.totalPrintable) % text.N;

        ++text.ngrams[code];
    }

    count_t total = 0;
    for (int i = 0; i < text.N; ++i) {
        out.ngrams[i] += text.ngrams[i];
        total += text.ngrams[i];
    }

    time = clock() - time;
    printf("\rLoaded %s characters of training data in %s ms. (%s MB/s)\n", F3(total), F3(time),
        F3(CLOCKS_PER_SEC * total / (1024. * 1024. * time)));

    saveToCache(path, text);

    out.ngrams[0] -= textOverlap;
    // out.count -= textOverlap;
}

bool compareString(const wchar_t* a, const wchar_t* b) {
    const std::wstring x(a);
    const std::wstring y(b);
    return x.compare(y) == 0;
}

bool inCache(const wchar_t* path, wchar_t* buf) {
    if (!cachedText) return false;

    std::wifstream file(cachedIndexFile, std::ifstream::in);
    if (!file.is_open()) {
        printf("Could not open cache file.\n");
        return false;
    }
    while (!file.eof()) {
        wchar_t temp[256];

        file.get(temp, 256, '|');
        file.get();
        file.get(buf, 256, '\n');
        file.get();

        if (compareString(temp, path)) {
            return true;
        }
    }
    return false;
}

bool loadCache(const UnfinishedText &out, const wchar_t* location) {
    if (!cachedText) return false;

    std::ifstream file(location, std::ifstream::in);
    if (!file.is_open()) {
        printf("Failed to open %ls\n", location);
        return false;
    }

    int time = clock();
    const UnfinishedText cache;

    while (!file.eof()) {
        char line[256];
        file.get(line, 256);
        file.get();

        char code[maxTextWindow];
        for (int i = 0; i < maxTextWindow; ++i) {
            code[i] = letterUtils.printableToCode[line[i]];
        }
        const int idx = letterUtils.indexOf<maxTextWindow, letterUtils.totalPrintable>(code);
        cache.ngrams[idx] += atoll(line + maxTextWindow + 1);
    }

    count_t total = 0;
    for (int i = 0; i < out.N; ++i) {
        total += cache.ngrams[i];
        out.ngrams[i] += cache.ngrams[i];
    }

    time = clock() - time;
    printf("Loaded %s characters from cache in %d ms.\n", F3(total), time);
    return true;
}

bool addText(const std::filesystem::path &path, const UnfinishedText &out) {
    if (path.extension().string().substr(0, 6) == ".textc") {
        return loadCache(out, path.c_str());
    }

    const std::filesystem::path can = weakly_canonical(path);
    const wchar_t* fp = can.c_str();

    bool cached = false;
    if (wchar_t cachedLocation[256]; inCache(fp, cachedLocation)) {
        cached = loadCache(out, cachedLocation);
    }
    if (!cached) {
        loadText(fp, out);
    }
    return cached;
}

void writeText(const UnfinishedText &text) {
    writeCache(L"text.chc", text);
}


std::unique_ptr<FinishedText<textWindow>> initText(const wchar_t* saveTo, const std::vector<std::filesystem::path> &paths) {
    UnfinishedText counts;
    std::unique_ptr<FinishedText<textWindow>> out;

    if (saveTo != nullptr && saveTo[0] > 0 && std::filesystem::exists(saveTo)) {
        printf("Loading text from previous run: '%ls'\n", saveTo);
        loadCache(counts, saveTo);
        saveTo = nullptr;
        out = std::make_unique<FinishedText<textWindow>>(counts);
    } else {
        int cacheHits = 0;
        int totalAdded = 0;

        int nLength = 0;
        {
            int n = paths.size();
            while (n > 0) {
                nLength++;
                n /= 10;
            }
        }

        for (int idx = 0; idx < paths.size(); idx++) {
            writeText(counts);

            const std::filesystem::path &s = paths[idx];
            printf("[%*d / %lld] Adding '%ls' to text...\n", nLength, idx + 1, paths.size(),
                s.c_str());
            cacheHits += addText(s, counts);

            totalAdded++;
        }
        out = std::make_unique<FinishedText<textWindow>>(counts);
        printf("Total files read: %s (%s cached)\n", F3(totalAdded), F3(cacheHits));
        printf("Total chars read: %s\n", F3(out->count));
    }

    writeText(counts);

    if (out->count == 0) {
        printf("Failed to read any characters of text.\n");
        exit(2);
    }

    if (saveTo != nullptr) {
        printf("Saving text to '%ls'.\n", saveTo);
        writeCache(saveTo, counts);
    }

    return out;
}

std::unique_ptr<FinishedText<textWindow>> initText(const char* file) {
    std::vector<std::filesystem::path> paths;
    paths.push_back(file);
    return initText(nullptr, paths);
}

std::unique_ptr<FinishedText<textWindow>> initText(const char* root, const std::vector<std::string> &exts) {
    std::vector<std::filesystem::path> paths;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(root)) {
        const std::filesystem::path s = entry.path();
        std::filesystem::path ext = s.extension();

        for (const std::string &cmp : exts) {
            if (ext == cmp) {
                paths.push_back(s);
                break;
            }
        }
    }

    wchar_t saveTo[512] = {};
    if (cachedText) {
        wchar_t repl[1024];
        wchar_t* rp = repl;
        for (const char* p = root; *p != '\0'; ++rp, ++p) {
            if (*p == ':' || *p == '/' || *p == '\\') *rp = '_';
            else *rp = *p;
        }
        *rp = '\0';

        std::wstringstream stream;

        stream << cacheDirectory << repl << "{";

        for (int i = 0; i < exts.size(); ++i) {
            if (i) stream << ',';
            stream << exts[i].c_str() + 1;
        }
        stream << "}x" << paths.size() << '[' << KEYS << "].text" << textWindow << 'c' << sizeof(text_t);

        const std::wstring out = stream.str();
        memcpy(saveTo, out.c_str(), out.length() * sizeof(wchar_t));
    }
    return initText(saveTo, paths);
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
        printf("%c", QC0[letters[i]]);
        if (i % 10 == 9) printf("\n");
    }
}
