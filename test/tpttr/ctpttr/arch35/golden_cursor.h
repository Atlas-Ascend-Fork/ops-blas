/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace complex_triangular_test {

struct ComplexBits {
    uint32_t real;
    uint32_t imag;
};
static_assert(sizeof(ComplexBits) == 8U, "complex64 bit representation mismatch");

inline uint32_t FloatBits(float value)
{
    static_assert(sizeof(float) == sizeof(uint32_t), "float32 representation required");
    uint32_t bits = 0;
    std::copy_n(reinterpret_cast<const unsigned char*>(&value), sizeof(bits), reinterpret_cast<unsigned char*>(&bits));
    return bits;
}

// Independent CPU reference: traverse columns and increment a packed cursor.
// It does not call production packed-offset or partitioning functions.
class GoldenCursor {
public:
    GoldenCursor(const ComplexBits* input, uint64_t n, uint64_t lda, bool upper)
        : input_(input), n_(n), lda_(lda), upper_(upper)
    {}

    bool Selected(uint64_t position) const
    {
        const uint64_t row = position % lda_;
        const uint64_t column = position / lda_;
        return column < n_ && row < n_ && (upper_ ? row <= column : row >= column);
    }

    ComplexBits Next() { return input_[packedCursor_++]; }

private:
    const ComplexBits* input_;
    uint64_t n_;
    uint64_t lda_;
    bool upper_;
    uint64_t packedCursor_ = 0;
};

} // namespace complex_triangular_test
