#include <lib/def.cuh>
#include <lib/text.cuh>
#include <common.cuh>
#include <iostream>
#include <metric.cuh>
#include <unordered_map>
#include <vector>

class TestGroup {
    std::vector<std::string> names;
    std::vector<stats> scores;

public:
    void combine(const TestGroup &other) {
        for (int i = 0; i < other.names.size(); ++i) {
            names.push_back(other.names[i]);
            scores.push_back(other.scores[i]);
        }
    }

    void add(const std::string &name, const stats score) {
        names.push_back(name);
        scores.push_back(score);
    }

    static void show(const std::string &string, const stats &out) {
        printf("%s:\n", string.c_str());
        printf("    %-13s = %s\n", "score", F3(out.score));
        if constexpr (stats::attrs > 0) {
            const text_t* arr = (text_t*) ((char*) &out + sizeof(out.score));
            for (int i = 0; i < stats::attrs; ++i) {
                printf("    %-13s = %s\n", stats::names[i], F3(arr[i]));
            }
        }
        printf("\n");
    }

    void print() const {
        const int n = scores.size();
        std::vector seen(n, false);

        for (int i = 0; i < n; ++i) {
            int best = -1;
            score_t cmp = 0;
            for (int j = 0; j < n; ++j) {
                if (seen[j]) continue;

                if (best == -1 || CMP(scores[j].score < cmp)) {
                    best = j;
                    cmp = scores[j].score;
                }
            }
            show(names[best], scores[best]);
            seen[best] = true;
        }
    }
};

struct TestCollection {
    std::unordered_map<std::string, std::vector<char>> lut{};

    TestCollection() {
        add(
            "dist-w&p",
            "1234567890"
            "j.glykcub;"
            "ietrhdnosa"
            "/xwmzpf,vq"
        );
        add(
            "dist-full",
            "1234567890"
            "jbglykcu,;"
            "istrhdnoea"
            "/vwmzpf.xq"
        );
        add(
            "okp1",
            "1234567890"
            "/oupqxlbfw"
            "eainkmhtsc"
            ";.,yzjrdvg"
        );
        add(
            "okp2",
            "1234567890"
            ".uofqjmdbg"
            "ieanylrhts"
            ",;/pzwcvkx"
        );
        add(
            "qwerty",
            "1234567890"
            "qwertyuiop"
            "asdfghjkl;"
            "zxcvbnm,./"
        );
        add(
            "alphabet",
            "1234567890"
            "abcdefghij"
            "klmnopqrs;"
            "tuvwxyz,./"
        );
        add(
            "dvorak",
            "1234567890"
            "/,.pyfgcrl"
            "aoeuidhtns"
            ";qjkxbmwvz"
        );
        add(
            "colemak",
            "1234567890"
            "qwfpgjluy;"
            "arstdhneio"
            "zxcvbkm,./"
        );
        add(
            "carpalx",
            "1234567890"
            "qgmlwbyuv;"
            "dstnriaeoh"
            "zxcfjkp,./"
        );
        add(
            "arensito",
            "1234567890"
            "ql,p/;fudk"
            "arenbgsito"
            "zw.hjvcymx"
        );
        add(
            "asset",
            "1234567890"
            "qwjfgypul;"
            "asetdhnior"
            "zxcvbkm,./"
        );
        add(
            "capewell",
            "1234567890"
            ".ywdfjpluq"
            "aersgbtnio"
            "xzcv;kwh,/"
        );
    }

    void add(const std::string &name, const char* array) {
        std::vector<char> &arr = lut[name];
        arr.clear();
        for (const char* ptr = array; *ptr != '\0'; ++ptr) {
            arr.push_back(*ptr);
        }
    }

    void remove(const std::string &name) {
        lut.erase(name);
    }

    void test(TestGroup &tg) const {
        for (const auto &[name, _] : lut) {
            test(tg, name);
        }
    }

    void test(TestGroup &tg, const std::string &name) const {
        if (lut.find(name) == lut.end()) {
            return;
        }

        char t[KEYS];
        const std::vector<char> &arr = lut.at(name);

        int k = 0;
        for (const char c : arr) {
            if (const int j = letterUtils.positionOf(c); j != -1) {
                t[j] = k++;
            }
        }
        if (k != KEYS) {
            printf("Keyboard layout '%s' must have %d valid keys: Found %d.\n", name.c_str(), k, KEYS);
        } else {
            tg.add(name, score(t));
        }
    }
} tests;


void testAll() {
    TestGroup t;
    tests.test(t);
    t.print();
}


bool testable(const std::string &name) {
    return tests.lut.find(name) != tests.lut.end();
}
void test(const std::string &name) {
    TestGroup t;
    tests.test(t, name);
    t.print();
}

void testUser() {
    printf("To test a keyboard, enter a string of text with exactly %d letters included in"
           "the following string:\n%s\n%s\nTo exit the testing loop, enter \"stop\".\n\n" , KEYS, KEYS_LOWER, KEYS_UPPER);
    while (true) {
        char keyboard[KEYS];
        int seen = 0;

        printf("Enter a keyboard or \"stop\"\n");
        while (seen != KEYS) {
            char buffer[1024];
            std::cin.getline(buffer, 1024);
            if (std::string(buffer) == "stop") {
                printf("Exiting.\n");
                goto exit;
            }
            for (int i = 0; buffer[i] != '\0'; ++i) {
                if (letterUtils.positionOf(buffer[i]) != -1) {
                    if (seen < KEYS) keyboard[seen++] = buffer[i];
                    else seen++;
                }
            }
            if (seen > KEYS) {
                printf("Too many keys found. Expected %d, found at least %d.\n", KEYS, seen);
                break;
            }
        }

        if (seen != KEYS) continue;

        printf("Your keyboard:\n");
        for (int i = 0; i < KEYS; ++i) {
            printf("%c", keyboard[i]);
            if (i % 10 == 9) printf("\n");
        }
        printf("\n");

        TestGroup tg;
        tests.add("result", keyboard);
        tests.test(tg, "result");
        tg.print();
    }
    exit:
    tests.remove("result");
}