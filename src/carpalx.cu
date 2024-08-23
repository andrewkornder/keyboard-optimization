#include <algorithm>
#include <common.cuh>

static constexpr int fingers[KEYS] {
    0,1,2,3,3,6,6,7,8,9,
    0,1,2,3,3,6,6,7,8,9,
    0,1,2,3,3,6,6,7,8,9,
};
static constexpr double pathCosts[1000] {
    0. , 0.3, 0.6, 0.9, 0. , 0. , 1.8, 0. , 0. , 0. , 0.3, 0.6, 0.9,
    1.2, 0. , 0. , 2.1, 0. , 0. , 0. , 0.6, 0.9, 1.2, 1.5, 0. , 0. ,
    2.4, 0. , 0. , 0. , 0.9, 0. , 1.5, 1.8, 0. , 0. , 2.7, 0. , 0. ,
    0. , 1.2, 0. , 1.8, 2.1, 0. , 0. , 3. , 0. , 0. , 0. , 1.5, 0. ,
    2.1, 2.4, 0. , 0. , 3.3, 0. , 0. , 0. , 1.8, 0. , 2.4, 2.7, 0. ,
    0. , 3.6, 0. , 0. , 0. , 2.1, 0. , 2.7, 3. , 0. , 0. , 3.9, 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1.6, 1.9,
    2.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1.9, 2.2, 2.5, 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 2.2, 2.5, 2.8, 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 2.5, 2.8, 3.1, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 2.8,
    3.1, 3.4, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 3.1, 3.4, 3.7, 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 3.4, 3.7, 4. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 3.7, 4. , 4.3, 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 2. , 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 0. ,
    0. , 0. , 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 0. , 0. , 2.6,
    2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 0. , 0. , 2.9, 0. , 3.5, 3.8,
    4.1, 4.4, 4.7, 5. , 0. , 0. , 3.2, 0. , 3.8, 4.1, 4.4, 0. , 5. ,
    5.3, 0. , 0. , 3.5, 0. , 4.1, 4.4, 4.7, 0. , 5.3, 5.6, 0. , 0. ,
    3.8, 0. , 4.4, 4.7, 5. , 0. , 5.6, 5.9, 0. , 0. , 4.1, 0. , 4.7,
    5. , 5.3, 5.6, 5.9, 6.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.
};

inline int getPath2(const char c1, const char c2, const char c3) {
    constexpr int hand[] = {
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    };

    const bool h1 = hand[c1];
    const bool h2 = hand[c2];
    const bool h3 = hand[c3];

    const int row1 = c1 / 10;
    const int row2 = c2 / 10;
    const int row3 = c3 / 10;

    const int f1 = fingers[c1];
    const int f2 = fingers[c2];
    const int f3 = fingers[c3];

    int handFlag;
    if (h1 == h3) {
        if (h2 == h3) {
            handFlag = 2;
        } else {
            handFlag = 1;
        }
    } else {
        handFlag = 0;
    }

    int fingerFlag;
    if (f1 > f2) {
        if (f2 > f3) {
            fingerFlag = 0;
        } else if (f2 == f3) {
            if (c2 == c3) {
                fingerFlag = 1;
            } else {
                fingerFlag = 6;
            }
        } else if (f3 == f1) {
            fingerFlag = 4;
        } else if (f1 > f3 && f3 > f2) {
            fingerFlag = 2;
        } else {
            fingerFlag = 3;
        }
    } else if (f1 < f2) {
        if (f2 < f3) {
            fingerFlag = 0;
        } else if (f2 == f3) {
            if (c2 == c3) {
                fingerFlag = 1;
            } else {
                fingerFlag = 6;
            }
        } else if (f3 == f1) {
            fingerFlag = 4;
        } else if (f1 < f3 && f3 < f2) {
            fingerFlag = 2;
        } else {
            fingerFlag = 3;
        }
    } else {
        if (f1 != f3) {
            if (c1 == c2) {
                fingerFlag = 1;
            } else {
                fingerFlag = 6;
            }
        } else {
            if (c1 != c2 && c2 != c3 && c1 != c3) {
                fingerFlag = 7;
            } else {
                fingerFlag = 5;
            }
        }
    }
    int rowFlag;

    int order[] = {row1 - row2, row1 - row3, row2 - row3};
    std::sort(order, order + 3, [](const int a, const int b) {
        if (abs(a) == abs(b)) return b > a;
        return abs(b) < abs(a);
    });

    const int drmax = order[0];
    const int drmax_abs = abs(drmax);
    
    if (row1 < row2) {
        if (row3 == row2) {
            rowFlag = 1;
        } else if (row2 < row3) {
            rowFlag = 4;
        } else if (drmax_abs == 1) {
            rowFlag = 3;
        } else {
            if (drmax < 0) {
                rowFlag = 7;
            } else {
                rowFlag = 5;
            }
        }
    } else if (row1 > row2) {
        if (row3 == row2) {
            rowFlag = 2;
        } else if (row2 > row3) {
            rowFlag = 6;
        } else if (drmax_abs == 1) {
            rowFlag = 3;
        } else {
            if (drmax < 0) {
                rowFlag = 7;
            } else {
                rowFlag = 5;
            }
        }
    } else {
        if (row2 > row3) {
            rowFlag = 2;
        } else if (row2 < row3) {
            rowFlag = 1;
        } else {
            rowFlag = 0;
        }
    }
    const int f = handFlag * 100 + rowFlag * 10 + fingerFlag;
    return f;
}


inline score_t effort(const char c1, const char c2, const char c3) {
    constexpr double kb = 0.3555, kp = 0.6423, ks = 0.4268, k1 = 1, k2 = 0.367, k3 = 0.235;
    constexpr double base[]{
        2.0, 2.0, 2.0, 2.0, 2.5, 3.0, 2.0, 2.0, 2.0, 2.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 2.0, 2.0, 2.0, 3.5, 2.0, 2.0, 2.0, 2.0, 2.0,
    };
    constexpr double row[] = {
        0.5, 0, 1
    };
    constexpr double fingerPenalty[] = {
        1, 0.5, 0, 0, 0
    };

    const double be1 = base[c1];
    const double be2 = base[c2];
    const double be3 = base[c3];

    const int row1 = c1 / 10;
    const int row2 = c2 / 10;
    const int row3 = c3 / 10;

    const int f1 = fingers[c1];
    const int f2 = fingers[c2];
    const int f3 = fingers[c3];

    const double pe1 = row[row1] * 1.3088 + fingerPenalty[f1] * 2.5948;
    const double pe2 = row[row2] * 1.3088 + fingerPenalty[f2] * 2.5948;
    const double pe3 = row[row3] * 1.3088 + fingerPenalty[f3] * 2.5948;

    double triad_effort =
        kb * k1 * be1 * (1 + k2 * be2 * (1 + k3 * be3)) +
        kp * k1 * pe1 * (1 + k2 * pe2 * (1 + k3 * pe3));

    triad_effort += ks * pathCosts[getPath2(c1, c2, c3)];
    return triad_effort;
}

inline bool save() {
    constexpr int k = 1000;
    constexpr int m = KEYS;
    constexpr int n = ipow(m, 3);
    
    uchar3* arr = new uchar3[k * n]; // (path, ngram)
    int* count = new int[k]();

    constexpr char ordering[m] = {
        'a','b','c','d','e','f','g','h','i','j',
        'k','l','m','n','o','p','q','r','s','t',
        'u','v','w','x','y','z',';',',','.','/',
    };
    constexpr char ignore[m] = {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,1,1,1,
    };
    
    char orderLUT[m] = {};
    for (int i = 0; i < m; ++i) {
        int j = 0;
        for (; j < m; ++j) {
            if (ordering[j] == KEYS_LOWER[i]) break;
        }
        orderLUT[i] = j;
    }
    
    for (int i = 0; i < n; ++i) {
        const unsigned char c = i % m;
        const unsigned char b = i / m % m;
        const unsigned char a = i / m / m % m;
        if (ignore[a] || ignore[b] || ignore[c]) continue;
        
        const int j = getPath2(a, b, c);

        const int size = count[j];
        arr[j * n + size] = uchar3{a, b, c};
        count[j]++;
    }

    std::ofstream file("paths");
    for (int i = 0; i < k; ++i) {
        const int s = count[i];
        if (s == 0) continue;

        std::sort(&arr[i * n], &arr[i * n + s], [orderLUT](const uchar3 &a, const uchar3 &b) {
            if (orderLUT[a.x] != orderLUT[b.x]) return orderLUT[a.x] < orderLUT[b.x];
            if (orderLUT[a.y] != orderLUT[b.y]) return orderLUT[a.y] < orderLUT[b.y];
            return orderLUT[a.z] < orderLUT[b.z];
        });

        char I[4] = {};
        I[0] = '0' + i / 100 % 10;
        I[1] = '0' + i /  10 % 10;
        I[2] = '0' + i /   1 % 10;

        file << I;
        for (int j = 0; j < s; ++j) {
            const uchar3 code = arr[i * n + j];
            file << ' ' << KEYS_LOWER[code.x] << KEYS_LOWER[code.y] << KEYS_LOWER[code.z];
        }
        file << '\n';
    }

    delete[] arr;
    delete[] count;
    return true;
}
