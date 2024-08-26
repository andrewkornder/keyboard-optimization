/*
 * Copyright (C) 2010 Google Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MD5_CUH
#define MD5_CUH

#include <array>
#include <vector>
#include <xstring>

class MD5 {
public:
    MD5();

    MD5& update(const std::vector<uint8_t>& input) {
        return update(input.data(), input.size());
    }
    MD5& update(const std::string &input) {
        return update(input.c_str(), input.size());
    }

    MD5& update(const char x) {
        return update(&x, 1);
    }
    MD5& update(const uint32_t x) {
        return update((uint8_t*) &x, sizeof(uint32_t));
    }
    MD5& update(const uint64_t x) {
        return update((uint8_t*) &x, sizeof(uint64_t));
    }

    MD5& update(const char* input, const size_t length) {
        return update((const uint8_t*) input, length);
    }

    MD5& update(const uint8_t* input, size_t length);

    // Size of the SHA1 hash
    static constexpr size_t hashSize = 16;

    // type for computing MD5 hash
    struct Digest64 {uint64_t hi{}; uint64_t lo{};};

    // checksum has a side effect of resetting the state of the object.
private:
    typedef std::array<uint8_t, hashSize> Digest;
    void checksum(Digest&);

public:
    std::string hex();
    std::string b64();
    Digest64 checksum();

private:
    uint32_t m_buf[4];
    uint32_t m_bits[2];
    uint8_t m_in[64];
};

#endif // MD5_CUH
