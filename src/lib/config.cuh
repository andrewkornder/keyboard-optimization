#ifndef CONFIG_CUH
#define CONFIG_CUH
#include <common.cuh>
#include <filesystem>
#include <set>

struct Config {
private:
    bool match(const std::string &key, const std::string &value);
    void parse(const char* configuration);

    std::set<std::string> seen{};

    std::filesystem::path exportTables;
    std::string corpus;
    std::vector<std::string> textExtensions{};

    bool configured(const char* key) {
        return seen.find(key) != seen.end();
    }

public:
    explicit Config(const char* configuration);

    char buffer[32 * 1024] = {};
    struct {
        pop_t pop = 1 << 22;
        pop_t surviving = 0;
    } size;
    int generations = 30;
    int rounds = 1000;

    int plateauLength = 2;
    bool showOutput = false;
    std::filesystem::path output;
};

#endif //CONFIG_CUH
