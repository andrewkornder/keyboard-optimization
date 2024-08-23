#include <lib/def.cuh>
#include <lib/text.cuh>
#include <common.cuh>
#include <metric.cuh>
#include <vector>


stats test(const char* kb) {
    char t[KEYS];
    mapKeys(kb, t);

    return score(t);
}

void show(const std::string &string, const stats &out) {
    const text_t* arr = (text_t*) ((char*) &out + sizeof(out.score));

    printf("%s:\n", string.c_str());
    printf("    %-13s = %s\n", "score", F3(out.score));
    for (int i = 0; i < stats::attrs; ++i) {
        printf("    %-13s = %s\n", stats::names[i], F3(arr[i]));
    }
    printf("\n");
}


class TestGroup {
    std::vector<const char*> names;
    std::vector<stats> scores;

public:
    void combine(const TestGroup &other) {
        for (int i = 0; i < other.names.size(); ++i) {
            names.push_back(other.names[i]);
            scores.push_back(other.scores[i]);
        }
    }

    void add(const char* name, const char* kb) {
        names.push_back(name);
        scores.push_back(test(kb));
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
#define TEST_(_name,s) { \
    constexpr char arr_[] = {s}; \
    t.add(#_name, arr_); \
}
#define TEST_T(name, s) TEST_(test{##name##},s)
#define TEST_N(name, s) TEST_(name,s)
TestGroup testNew_() {
    TestGroup t;

    TEST_T(dist-w&p,
        "1234567890"
        "j.glykcub;"
        "ietrhdnosa"
        "/xwmzpf,vq"
    );
    TEST_T(dist-full,
        "1234567890"
        "jbglykcu,;"
        "istrhdnoea"
        "/vwmzpf.xq"
    );
    TEST_T(okp1,
        "1234567890"
        "/oupqxlbfw"
        "eainkmhtsc"
        ";.,yzjrdvg"
    );
    TEST_T(okp2,
        "1234567890"
        ".uofqjmdbg"
        "ieanylrhts"
        ",;/pzwcvkx"
    );
    return t;
}
void testNew() {
    testNew_().print();
}

void testQWERTY() {
    TestGroup t;
    TEST_(qwerty,
        "1234567890"
        "qwertyuiop"
        "asdfghjkl;"
        "zxcvbnm,./"
    )
    t.print();
}

TestGroup testOther_() {
    TestGroup t;
    TEST_(qwerty,
        "1234567890"
        "qwertyuiop"
        "asdfghjkl;"
        "zxcvbnm,./"
    )
    TEST_(alphabet,
        "1234567890"
        "abcdefghij"
        "klmnopqrs;"
        "tuvwxyz,./"
    )
    TEST_(dvorak,
        "1234567890"
        "/,.pyfgcrl"
        "aoeuidhtns"
        ";qjkxbmwvz"
    )
    TEST_(colemak,
        "1234567890"
        "qwfpgjluy;"
        "arstdhneio"
        "zxcvbkm,./"
    )
    TEST_(carpalx,
        "1234567890"
        "qgmlwbyuv;"
        "dstnriaeoh"
        "zxcfjkp,./"
    )
    TEST_(arensito,
        "1234567890"
        "ql,p/;fudk"
        "arenbgsito"
        "zw.hjvcymx"
    )
    TEST_(asset,
        "1234567890"
        "qwjfgypul;"
        "asetdhnior"
        "zxcvbkm,./"
    )
    TEST_(capewell,
        "1234567890"
        ".ywdfjpluq"
        "aersgbtnio"
        "xzcv;kwh,/"
    )
    // TEST_(dickens,
    // "1 2 3 4 5 6 7 8 9 0 q"
    // "y p o u - v d l c w x"
    // "i n e a , m h t s r \""
    // "( ) ; . _ k f g b '"
    // "/ =             z j
    // )
    return t;
}
void testOther() {
    testOther_().print();
}
void testAll() {
    TestGroup t = testNew_();
    t.combine(testOther_());
    t.print();
}